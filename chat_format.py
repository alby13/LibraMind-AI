"""Chat template shared by data preparation, fine-tuning and the chat program (ChatML style).

  <|bos|><|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n{reply}<|im_end|>\\n

Messages are encoded piece by piece, so training and inference tokenize identically.
Loss mask: 1 on assistant reply tokens and the <|im_end|> that closes them (so the model learns to stop).
"""
import numpy as np
from tokenizers import Tokenizer

BOS, IM_START, IM_END = "<|bos|>", "<|im_start|>", "<|im_end|>"
ROLES = ("system", "user", "assistant", "tool")


def chat_tokenizer(base_tokenizer_path):
    """The pretraining tokenizer plus the chat special tokens (appended as new ids)."""
    tok = Tokenizer.from_file(str(base_tokenizer_path))
    tok.add_special_tokens([IM_START, IM_END])
    return tok


class ChatEncoder:
    def __init__(self, tok):
        self.tok = tok
        self.bos, self.im_start, self.im_end = (tok.token_to_id(t) for t in (BOS, IM_START, IM_END))
        self.newline = tok.encode("\n", add_special_tokens=False).ids
        self.headers = {r: tok.encode(f"{r}\n", add_special_tokens=False).ids for r in ROLES}

    def encode_conversation(self, messages):
        """messages: list of {"role", "content"}. Returns (ids uint16, loss mask uint8)."""
        ids, mask = [self.bos], [0]
        for m in messages:
            head = [self.im_start] + self.headers[m["role"]]
            body = self.tok.encode(m["content"], add_special_tokens=False).ids + [self.im_end]
            train = 1 if m["role"] == "assistant" else 0
            ids += head + body + self.newline
            mask += [0] * len(head) + [train] * len(body) + [0] * len(self.newline)
        return np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8)

    def encode_prompt(self, messages):
        """Conversation so far plus the assistant header, ready for generation."""
        ids, _ = self.encode_conversation(messages)
        return np.concatenate([ids, [self.im_start] + self.headers["assistant"]]).astype(np.int64)

    def encode_reply(self, text):
        """An assistant reply as the model would produce it: its tokens followed by <|im_end|>."""
        return np.array(self.tok.encode(text, add_special_tokens=False).ids + [self.im_end], dtype=np.int64)


def decode_conversation(tok, ids):
    """Turn a tokenized conversation back into [{"role", "content"}] messages."""
    text = tok.decode([int(i) for i in ids], skip_special_tokens=False).replace(BOS, "")
    msgs = []
    for part in text.split(IM_START)[1:]:
        role, _, body = part.partition("\n")
        msgs.append({"role": role, "content": body.split(IM_END)[0]})
    return msgs
