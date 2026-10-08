"""LibraMind Math Chat: the chat app with a calculator, so LibraMind Mini gets its arithmetic right.

  python mathchat.py                                (downloads the model from Hugging Face the first time)
  python mathchat.py --model path/to/LibraMind      (a local copy of the model folder)

Opens http://127.0.0.1:8055. Ctrl+C in this window stops it.

LibraMind can't do arithmetic reliably in its head, but it can call functions. In the default "Auto" mode:
  - messages that don't involve math are normal chat;
  - a question that IS a calculation ("What is 347 x 582?", "12% of 50") must start with a call to the
    calculate tool (grammar-forced); the expression is evaluated exactly and LibraMind answers from the result;
  - a word problem gets no tool: LibraMind reasons step by step in its own words, and every calculation it
    writes ("16 - 3 - 4 =") gets the exact result filled in before it can guess (an inline calculator).
"Let it decide" offers the tool and lets the model choose; "Off" turns the calculator off. The calculator only
evaluates numbers and + - * / // % ** ( ) and a few math functions (see calctools.py); nothing is run as code.
Each calculation is shown above the answer.
"""
import argparse
import json
import re
import sys
import time
import webbrowser
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

import torch

import chatui
from calctools import (EXPLICIT, TOOL, calculate, inline_calculator, needs_calculator, parse_calls, strip_calls,
                       system_prompt, tool_message)
from chatui import DEFAULT_MODEL, GPU, HERE, STATE, fit_context, load_model
from inference import GrammarFactory, generate

MAX_ROUNDS = 6  # calculator calls per reply
TAG = "<tool_call>"
HASH_LINE = re.compile(r"^\s*####[^\n]*\n?", re.M)  # "#### 85": answer format from the math training data


def clean_final(text):
    """Drop the "#### 85" line; if that leaves a bare "The answer is:", move the number up into it."""
    m = re.search(r"^\s*####\s*([^\n]*)", text, re.M)
    out = HASH_LINE.sub("", text).strip()
    if m and m.group(1).strip() and re.search(r"answer is:?\s*$", out, re.I):
        out = f"{out} {m.group(1).strip()}"
    return out


class Handler(chatui.Handler):
    def do_GET(self):
        if urlparse(self.path).path == "/":
            data = (HERE / "mathchat.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(data)
        return super().do_GET()

    def _chat(self, body):
        """Stream one reply, running calculations in between. Events: start, delta (answer text), calc
        (one calculation), done (final text + all calculations)."""
        model, enc = STATE["model"], STATE["enc"]
        if STATE["grammar"] is None:
            STATE["grammar"] = GrammarFactory(STATE["tok_path"], enc.im_end, model.config.vocab_size)
        grammar = STATE["grammar"]
        limit = STATE["info"]["max_len"]
        max_new = max(16, min(int(body.get("max_new", 512)), limit // 2))
        temperature, top_p = float(body.get("temperature", 0.4)), float(body.get("top_p", 0.9))
        rep_penalty = float(body.get("rep_penalty", 1.1))
        raw = [m for m in body.get("messages", []) if m.get("role") in ("user", "assistant") and m.get("content") is not None]
        history = [{"role": m["role"], "content": m["content"]} for m in raw]
        extra = (body.get("system") or "").strip()
        mode = body.get("calculator", "auto")
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        use_calc = mode in ("auto", "offer") and needs_calculator(last_user)
        # auto: a question that is an expression gets a calculator call, and the first reply must be one
        # (grammar-forced; left to decide, LibraMind answers mid-conversation math in its head). A word problem
        # gets no tool: as a tool user LibraMind squeezes the whole problem into one wrong expression; instead it
        # reasons as usual and the inline calculator fixes each "<arithmetic> =" it writes.
        use_tool = use_calc and (mode == "offer" or bool(EXPLICIT.search(last_user)))
        force_first = use_tool and mode == "auto"
        plain = ([{"role": "system", "content": extra}] if extra else []) + history
        if use_tool:
            # earlier answers go back with the calculator calls behind them (call, result, answer, as in the
            # function-calling training data); with bare answers, the model copies the habit of answering directly
            with_calls = []
            for m in raw:
                calls = [c for c in (m.get("calcs") or []) if isinstance(c, dict) and c.get("expression")
                         and not c.get("inline")]
                if m["role"] == "assistant" and calls:
                    with_calls.append({"role": "assistant", "content": "\n".join(
                        "<tool_call>\n" + json.dumps({"name": "calculate", "arguments": {"expression": c["expression"]}})
                        + "\n</tool_call>" for c in calls)})
                    with_calls.append(tool_message([calculate(c["expression"]) for c in calls]))
                with_calls.append({"role": m["role"], "content": m["content"]})
            convo = [{"role": "system", "content": system_prompt(extra)}] + with_calls
        else:
            convo = list(plain)

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

        calcs = []

        def gen(msgs, forced=False, stream=True):
            """One model reply. Streams only the part before any <tool_call> (calls are shown as calculations).
            For math messages, every "<arithmetic> =" written in free text gets the exact result inserted."""
            prompt, dropped = fit_context(enc, msgs, limit - max_new)
            ids, shown = [], {"n": 0}

            def inline_calc(r):
                r = dict(r, inline=True)
                calcs.append(r)
                send({"calc": r})
            hook = inline_calculator(enc.tok, on_calc=inline_calc) if use_calc and not forced else None

            def on_token(_, t):
                ids.append(t)
                if not stream:
                    return
                text = enc.tok.decode(ids)
                if text.endswith("�"):
                    return  # wait for multi-byte characters to complete
                visible = text.split(TAG)[0]
                held = max((k for k in range(1, len(TAG)) if visible.endswith(TAG[:k])), default=0)
                visible = visible[:len(visible) - held]  # don't flash the start of a "<tool_call>"
                if len(visible) > shown["n"]:
                    send({"delta": visible[shown["n"]:]})
                    shown["n"] = len(visible)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                generate(model, [prompt], max_new, temperature, top_p, rep_penalty=rep_penalty, stop_ids=[enc.im_end],
                         matchers=[grammar.tool_calls([TOOL])] if forced else None, vocab=enc.tok.get_vocab_size(),
                         on_token=on_token, insert=hook)
            return enc.tok.decode(ids), len(ids), len(prompt), dropped

        with GPU:
            try:
                t0, tokens, final = time.time(), 0, None
                prompt0, dropped0 = fit_context(enc, convo, limit - max_new)
                send({"start": True, "prompt_tokens": len(prompt0), "dropped": dropped0, "calculator": use_calc})
                rounds = MAX_ROUNDS if use_tool else 0
                for rnd in range(rounds + 1):
                    forced = force_first and rnd == 0
                    text, n, plen, _ = gen(convo, forced=forced, stream=not forced)
                    tokens += n
                    exprs = parse_calls(text) if use_tool else []
                    if exprs and any("error" in calculate(e) for e in exprs):
                        # words instead of numbers: redo this step with the expression constrained to math
                        text, n, plen, _ = gen(convo, forced=True, stream=False)
                        tokens += n
                        exprs = parse_calls(text)
                    if exprs and rnd < rounds:
                        results = [calculate(e) for e in exprs]
                        for r in results:
                            calcs.append(r)
                            send({"calc": r})
                        convo += [{"role": "assistant", "content": text}, tool_message(results)]
                        continue
                    final = clean_final(strip_calls(text))
                    break
                if not final:  # a call with no usable expression: answer without the calculator
                    final, n, plen, _ = gen(plain)
                    tokens += n
                send(dict(done=True, text=final, calcs=calcs, calculator=use_calc, tokens=tokens,
                          seconds=round(time.time() - t0, 2), reason="end", context=plen + n, max_len=limit))
            except Cancelled:
                pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local folder (default: %(default)s)")
    p.add_argument("--name", default="LibraMind Mini", help="name shown in the page header")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices on your network")
    p.add_argument("--port", type=int, default=8055)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        sys.exit("LibraMind needs an NVIDIA GPU with CUDA (its Gated DeltaNet layers run as Triton GPU kernels).")
    print(f"Loading {args.model}...")
    info = load_model(args.model, args.name)
    print(f"  {info['params'] / 1e6:.0f}M parameters, {info['max_len']:,}-token context")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"LibraMind Math Chat at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
