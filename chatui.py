"""LibraMind Chat: talk to LibraMind Mini in your browser.

  python chatui.py                                  (downloads the model from Hugging Face the first time)
  python chatui.py --model path/to/LibraMind        (a local copy of the model folder)
  python chatui.py --port 8060 --no-browser

Opens http://127.0.0.1:8052. Ctrl+C in this window stops it.

The system prompt box is sent with every message, so you can give the model a role or turn it into a
character at any point; the change applies to the next reply. Replies stream in as they're written.
Stop, regenerate, edit-and-resend and a "reply as JSON" mode (for game NPC actions) are on the page.
Conversations are kept in your browser if you reload.

Needs an NVIDIA GPU (about 2-3 GB of VRAM).
"""
import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from chat_format import ChatEncoder
from inference import GrammarFactory, generate
from model import LM, ModelConfig

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "alby13/LibraMindMini"  # Hugging Face repo with model.safetensors, config.json and tokenizer.json
GPU = threading.Lock()  # one reply at a time
STATE = {"model": None}
NPC_SCHEMA = {
    "type": "object",
    "properties": {
        "line": {"type": "string", "maxLength": 200},
        "emotion": {"enum": ["happy", "angry", "sad", "neutral", "surprised", "afraid"]},
        "action": {"enum": ["none", "give_item", "attack", "open_shop", "leave", "follow_player"]},
    },
    "required": ["line", "emotion", "action"],
    "additionalProperties": False,
}


# ----------------------------------------------------------------------------- model loading

def model_folder(name):
    """A local folder holding the model files, downloading it from Hugging Face if `name` is a repo id."""
    path = Path(name)
    if (path / "model.safetensors").exists():
        return path
    if path.exists():
        sys.exit(f"{path} has no model.safetensors")
    from huggingface_hub import snapshot_download
    print(f"Downloading {name} from Hugging Face (about 1.1 GB, once)...")
    return Path(snapshot_download(name, allow_patterns=["model.safetensors", "config.json", "tokenizer.json"]))


def load_model(name, display_name):
    folder = model_folder(name)
    cfg = json.loads((folder / "config.json").read_text())
    model = LM(ModelConfig(**{**cfg, "grad_ckpt": False}))
    model.load_state_dict(load_file(str(folder / "model.safetensors")))
    model = model.cuda().to(torch.bfloat16).eval()
    tok_path = str(folder / "tokenizer.json")
    info = dict(name=display_name, params=sum(p.numel() for p in model.parameters()), max_len=model.config.max_seq_len)
    STATE.update(model=model, enc=ChatEncoder(Tokenizer.from_file(tok_path)), tok_path=tok_path, grammar=None, info=info)
    return info


def fit_context(enc, messages, limit):
    """Drop the oldest exchanges (always keeping the system prompt) until the prompt fits in `limit` tokens.
    Returns (prompt ids, number of messages dropped)."""
    keep = 1 if messages and messages[0]["role"] == "system" else 0
    dropped = 0
    prompt = enc.encode_prompt(messages)
    while len(prompt) > limit and len(messages) > keep + 1:
        cut = 2 if len(messages) > keep + 2 else 1
        messages, dropped = messages[:keep] + messages[keep + cut:], dropped + cut
        prompt = enc.encode_prompt(messages)
    return prompt[-limit:], dropped


# ----------------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")

    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == "/":
                data = (HERE / "chatui.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return self.wfile.write(data)
            if url.path == "/api/status":
                return self._json(dict(info=STATE.get("info"), busy=GPU.locked()))
            if url.path == "/api/npc_schema":
                return self._json(NPC_SCHEMA)
            self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = self._body()
            if url.path == "/api/count":
                return self._json(dict(tokens=len(STATE["enc"].encode_prompt(self._messages(body)))))
            if url.path == "/api/chat":
                return self._chat(body)
            self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    @staticmethod
    def _messages(body):
        msgs = [m for m in body.get("messages", []) if m.get("role") in ("user", "assistant") and m.get("content") is not None]
        system = (body.get("system") or "").strip()
        return ([{"role": "system", "content": system}] if system else []) + msgs

    def _chat(self, body):
        """Stream the reply as lines of JSON; closing the connection (Stop) ends generation."""
        model, enc = STATE["model"], STATE["enc"]
        limit = STATE["info"]["max_len"]
        max_new = max(16, min(int(body.get("max_new", 512)), limit // 2))
        prompt, dropped = fit_context(enc, self._messages(body), limit - max_new)
        schema = body.get("schema")
        matcher = None
        if schema is not None:
            if STATE["grammar"] is None:
                STATE["grammar"] = GrammarFactory(STATE["tok_path"], enc.im_end, model.config.vocab_size)
            matcher = STATE["grammar"].json(schema or None)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        class Cancelled(Exception):
            pass

        def send(obj):
            try:
                self.wfile.write((json.dumps(obj) + "\n").encode())
                self.wfile.flush()
            except OSError as e:
                raise Cancelled from e

        ids, shown = [], {"n": 0}

        def on_token(_, t):  # decode everything so far and send the new part (multi-byte characters wait)
            ids.append(t)
            text = enc.tok.decode(ids)
            if not text.endswith("�") and len(text) > shown["n"]:
                send({"delta": text[shown["n"]:]})
                shown["n"] = len(text)

        with GPU:
            try:
                send({"start": True, "prompt_tokens": len(prompt), "dropped": dropped})
                t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    generate(model, [prompt], max_new, float(body.get("temperature", 0.4)), float(body.get("top_p", 0.9)),
                             rep_penalty=float(body.get("rep_penalty", 1.1)), stop_ids=[enc.im_end],
                             matchers=[matcher] if matcher else None, vocab=enc.tok.get_vocab_size(), on_token=on_token)
                text = enc.tok.decode(ids)
                if len(text) > shown["n"]:
                    send({"delta": text[shown["n"]:]})
                send(dict(done=True, text=text, tokens=len(ids), seconds=round(time.time() - t0, 2),
                          reason="length" if len(ids) >= max_new else "end",
                          context=len(prompt) + len(ids), max_len=limit))
            except Cancelled:
                pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local folder (default: %(default)s)")
    p.add_argument("--name", default="LibraMind Mini", help="name shown in the page header")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices on your network")
    p.add_argument("--port", type=int, default=8052)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        sys.exit("LibraMind needs an NVIDIA GPU with CUDA (its Gated DeltaNet layers run as Triton GPU kernels).")
    print(f"Loading {args.model}...")
    info = load_model(args.model, args.name)
    print(f"  {info['params'] / 1e6:.0f}M parameters, {info['max_len']:,}-token context")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"LibraMind Chat at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
