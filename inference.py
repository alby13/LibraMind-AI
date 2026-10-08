"""Fast generation: cached incremental decoding, batched sampling, optional grammar constraints.

Each sequence's state is small and fixed per token: attention layers keep their keys/values, DeltaNet
layers keep their recurrent state (flash-linear-attention cache), and Block AttnRes only mixes within
a token, so it needs no cache. A prompt is processed once (prefill) and each new token then costs one
cheap step instead of re-running the whole conversation.

  outs = generate(model, [prompt_ids, ...], max_new=200, stop_ids=[enc.im_end])
  matcher = GrammarFactory(tokenizer_path, im_end_id).json(schema)   # output guaranteed to parse
"""
import torch
import torch.nn.functional as F

from model import DeltaNet, GatedAttention, attnres_mix, inv_rms, rms


class _GDNStates:
    """The minimal cache interface flash-linear-attention layers expect (len / [] / update)."""

    def __init__(self, n):
        self.layers = [None] * n

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def update(self, layer_idx, recurrent_state=None, conv_state=None, **_):
        cur = self.layers[layer_idx]
        if cur is None:  # first call (prefill): keep the tensors
            self.layers[layer_idx] = dict(recurrent_state=recurrent_state, conv_state=conv_state)
            return
        # later calls: write into the existing buffers, so a replayed CUDA graph carries the state forward
        pairs = [(cur["recurrent_state"], recurrent_state)] + list(zip(cur["conv_state"] or (), conv_state or ()))
        for dst, src in pairs:
            if src is not None and dst.data_ptr() != src.data_ptr():
                dst.copy_(src)


class Cache:
    """Decoding state for a batch: attention keys/values (preallocated to `capacity`), DeltaNet states,
    and each row's current length (rows may have different lengths)."""

    def __init__(self, model, batch, capacity):
        dev, dt = model.embed.weight.device, torch.bfloat16
        self.pos = torch.zeros(batch, dtype=torch.long, device=dev)
        self.max_pos, self.capacity = 0, capacity
        self.kv = {j: (torch.zeros(batch, s.nkv, capacity, s.hd, device=dev, dtype=dt),
                       torch.zeros(batch, s.nkv, capacity, s.hd, device=dev, dtype=dt))
                   for j, s in enumerate(model.sublayers) if isinstance(s, GatedAttention)}
        self.gdn = _GDNStates(len(model.sublayers))

    @staticmethod
    def stack(caches, extra):
        """Combine single-sequence caches of different lengths into one batch with room for `extra` tokens."""
        out = Cache.__new__(Cache)
        out.pos = torch.cat([c.pos for c in caches])
        out.max_pos = max(c.max_pos for c in caches)
        out.capacity = out.max_pos + extra
        out.kv = {}
        for j, (K0, _) in caches[0].kv.items():
            K = K0.new_zeros(len(caches), K0.size(1), out.capacity, K0.size(3))
            V = torch.zeros_like(K)
            for i, c in enumerate(caches):
                n = c.max_pos
                K[i, :, :n], V[i, :, :n] = c.kv[j][0][0, :, :n], c.kv[j][1][0, :, :n]
            out.kv[j] = (K, V)
        out.gdn = _GDNStates(len(caches[0].gdn))
        for i, layer in enumerate(caches[0].gdn.layers):
            if layer is not None:
                conv = layer["conv_state"]
                out.gdn.layers[i] = dict(
                    recurrent_state=torch.cat([c.gdn.layers[i]["recurrent_state"] for c in caches]),
                    conv_state=None if conv is None else tuple(
                        torch.cat([c.gdn.layers[i]["conv_state"][k] for c in caches]) for k in range(len(conv))))
        return out


def _attention(sub, x, K, V, pos):
    """GatedAttention over cached keys/values; x holds t new tokens per row starting at position pos[row]."""
    B, t, _ = x.shape
    q, gate = sub.q_proj(x).split(sub.nh * sub.hd, -1)
    q = q.view(B, t, sub.nh, sub.hd)
    k, v = sub.kv_proj(x).view(B, t, 2, sub.nkv, sub.hd).unbind(2)
    positions = pos[:, None] + torch.arange(t, device=x.device)            # [B, t]
    cos, sin = (b[0, :, 0][positions].unsqueeze(2).to(x.dtype) for b in (sub.cos, sub.sin))

    def rope(z):
        z1, z2 = z.chunk(2, -1)
        return torch.cat([z1 * cos - z2 * sin, z1 * sin + z2 * cos], -1)

    q, k = rope(rms(q)), rope(rms(k))
    rows = torch.arange(B, device=x.device)[:, None].expand(B, t)
    K[rows, :, positions] = k.to(K.dtype)
    V[rows, :, positions] = v.to(V.dtype)
    keys = K.repeat_interleave(sub.nh // sub.nkv, dim=1)
    vals = V.repeat_interleave(sub.nh // sub.nkv, dim=1)
    mask = torch.arange(K.size(2), device=x.device)[None, None, :] <= positions[:, :, None]  # causal per row
    y = F.scaled_dot_product_attention(q.transpose(1, 2), keys.to(q.dtype), vals.to(q.dtype), attn_mask=mask[:, None])
    y = y.transpose(1, 2).reshape(B, t, sub.nh * sub.hd)
    return sub.o_proj(y * torch.sigmoid(gate))


@torch.no_grad()
def step(model, idx, cache):
    """Feed t new tokens per row (idx [B, t]); returns next-token logits [B, vocab] for the last position."""
    t = idx.size(1)
    if cache.max_pos + t > cache.capacity:
        raise ValueError(f"cache full ({cache.capacity} tokens)")
    x0 = rms(model.embed(idx))
    blocks, blocks_inv, partial = [x0], [inv_rms(x0)], None
    for j, sub in enumerate(model.sublayers):
        srcs = blocks if partial is None else blocks + [partial]
        inv = blocks_inv if partial is None else blocks_inv + [inv_rms(partial)]
        h = rms(attnres_mix(srcs, inv, model.attnres_queries[j]))
        if isinstance(sub, GatedAttention):
            out = _attention(sub, h, *cache.kv[j], cache.pos)
        elif isinstance(sub, DeltaNet):
            sub.gdn.layer_idx = j
            out = sub.gdn(h, past_key_values=cache.gdn, use_cache=True)[0]
        else:
            out = sub(h)
        out = out.float()
        partial = out if partial is None else partial + out
        if (j + 1) % model.block_size == 0:
            blocks.append(partial)
            blocks_inv.append(inv_rms(partial))
            partial = None
    srcs = blocks if partial is None else blocks + [partial]
    inv = blocks_inv if partial is None else blocks_inv + [inv_rms(partial)]
    h = rms(attnres_mix(srcs, inv, model.attnres_queries[-1]))
    cache.pos += t
    cache.max_pos += t
    return model._logits(h[:, -1])


def prefill(model, prompt, extra):
    """Process one prompt (list/array of ids); returns (cache with room for `extra` tokens, last logits [1, V])."""
    ids = torch.as_tensor(prompt, dtype=torch.long, device=model.embed.weight.device)[None]
    cache = Cache(model, 1, ids.size(1) + extra)
    return cache, step(model, ids, cache)


def sample(logits, temperature=0.7, top_p=0.9, top_k=0, recent=None, rep_penalty=1.0, bitmask=None, vocab=None):
    """Pick one token per row. recent: [B, n] recently generated ids (-1 = none) for the repetition penalty."""
    logits = logits.float()
    if vocab is not None and vocab < logits.size(1):
        logits[:, vocab:] = -float("inf")  # padding rows of the embedding are not real tokens
    if recent is not None and rep_penalty != 1.0:
        seen = torch.zeros(logits.size(0), logits.size(1) + 1, device=logits.device, dtype=torch.bool)
        seen.scatter_(1, torch.where(recent < 0, logits.size(1), recent), True)
        seen = seen[:, :-1]
        logits = torch.where(seen, torch.where(logits > 0, logits / rep_penalty, logits * rep_penalty), logits)
    if bitmask is not None:  # llguidance bitmask: bit i of word w allows token 32*w + i
        bits = (bitmask.to(logits.device)[:, :, None] >> torch.arange(32, device=logits.device)) & 1
        logits = logits.masked_fill(bits.reshape(bits.size(0), -1)[:, :logits.size(1)] == 0, -float("inf"))
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    probs = torch.softmax(logits, -1)
    if top_p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        sp = sp.masked_fill(sp.cumsum(-1) - sp > top_p, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    return torch.multinomial(probs, 1)[:, 0]


@torch.no_grad()
def generate(model, prompts, max_new=256, temperature=0.7, top_p=0.9, top_k=0, rep_penalty=1.0, stop_ids=(),
             matchers=None, batch_size=32, vocab=None, on_token=None, cuda_graph=True, insert=None):
    """Sample continuations for many prompts, `batch_size` at a time. Returns a list of token-id lists.
    matchers: optional per-prompt llguidance matchers (None = unconstrained) that force a grammar.
    on_token(row_index, token_id): optional callback, e.g. for streaming a single prompt.
    insert(row_index, tokens_so_far): optional callback returning token ids to write next instead of sampling
    (e.g. an exact calculator result after "12 * 7 ="), or None. Not applied to grammar-forced rows."""
    from llguidance.torch import allocate_token_bitmask, fill_next_token_bitmask
    results = [None] * len(prompts)
    stop = torch.tensor(list(stop_ids) or [-1], device=model.embed.weight.device)
    for start in range(0, len(prompts), batch_size):
        idx = list(range(start, min(start + batch_size, len(prompts))))
        pre = [prefill(model, prompts[i], max_new) for i in idx]
        cache = Cache.stack([c for c, _ in pre], max_new) if len(pre) > 1 else pre[0][0]
        logits = torch.cat([l for _, l in pre])
        del pre
        B = len(idx)
        rows_m = [matchers[i] if matchers else None for i in idx]
        bitmask = allocate_token_bitmask(B, logits.size(1)) if any(rows_m) else None
        out = [[] for _ in range(B)]
        queued = [[] for _ in range(B)]  # tokens from `insert` still to be written
        done = torch.zeros(B, dtype=torch.bool, device=logits.device)
        recent = torch.full((B, 64), -1, dtype=torch.long, device=logits.device)
        graph = _DecodeGraph(model, cache, B) if cuda_graph else None
        for n in range(max_new):
            if bitmask is not None:
                bitmask.fill_(-1)  # all tokens allowed ...
                for r, m in enumerate(rows_m):
                    if m is not None and not m.is_stopped():
                        fill_next_token_bitmask(m, bitmask, r)  # ... except where a grammar forbids them
            nxt = sample(logits, temperature, top_p, top_k, recent, rep_penalty, bitmask, vocab)
            nxt = torch.where(done, stop[0].clamp(min=0), nxt)
            toks = nxt.tolist()
            if any(queued):  # inserted tokens replace what was sampled
                for r in range(B):
                    if queued[r] and not done[r]:
                        toks[r] = queued[r].pop(0)
                nxt = torch.tensor(toks, dtype=nxt.dtype, device=nxt.device)
            for r in range(B):
                if done[r]:
                    continue
                if rows_m[r] is not None:
                    rows_m[r].consume_token(toks[r])
                if toks[r] in stop_ids:
                    done[r] = True
                    continue
                out[r].append(toks[r])  # keep it even if it completes the grammar (e.g. the final "}")
                if on_token:
                    on_token(idx[r], toks[r])
                if rows_m[r] is not None and rows_m[r].is_stopped():
                    done[r] = True
                elif insert and rows_m[r] is None and not queued[r]:
                    queued[r] = list(insert(idx[r], out[r]) or [])
            recent = torch.cat([recent[:, 1:], nxt[:, None]], 1)
            if bool(done.all()) or n == max_new - 1:
                break
            logits = graph.step(nxt) if graph else step(model, nxt[:, None], cache)
        for r, i in enumerate(idx):
            results[i] = out[r]
    return results


class _DecodeGraph:
    """Records the one-token decode step as a CUDA graph and replays it: one launch instead of ~3,500
    small kernel launches per token (launch overhead, not the GPU, dominates at this model size).
    The first steps run normally (they also warm up the Triton kernels); if recording fails, it
    quietly keeps running the normal way."""

    WARMUP = 2

    def __init__(self, model, cache, batch):
        self.model, self.cache, self.n, self.graph = model, cache, 0, None
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device=model.embed.weight.device)
        self.failed = False

    def step(self, nxt):
        self.n += 1
        if self.failed or self.n <= self.WARMUP:
            return step(self.model, nxt[:, None], self.cache)
        if self.cache.max_pos + 1 > self.cache.capacity:
            raise ValueError(f"cache full ({self.cache.capacity} tokens)")
        self.tok.copy_(nxt[:, None])
        if self.graph is None:
            try:
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    self.out = step(self.model, self.tok, self.cache)  # recorded, not run
                self.cache.max_pos -= 1  # the recording pass bumped it; the replay below is the real step
            except Exception as e:  # noqa: BLE001 - fall back to plain steps
                print(f"(CUDA graph unavailable, using normal decoding: {type(e).__name__}: {str(e)[:120]})")
                self.failed, self.graph = True, None
                return step(self.model, nxt[:, None], self.cache)
        self.graph.replay()
        self.cache.max_pos += 1
        return self.out


class GrammarFactory:
    """Builds llguidance matchers for our tokenizer: guaranteed-valid JSON or tool calls."""

    def __init__(self, tokenizer_path, stop_id, n_vocab):
        """n_vocab: the model's (padded) output size, so masks line up with its logits."""
        import llguidance.hf
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        hf = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(str(tokenizer_path)))
        self.lltok = llguidance.hf.from_tokenizer(hf, n_vocab=n_vocab, eos_token=stop_id)

    def _matcher(self, grammar):
        from llguidance import LLMatcher
        m = LLMatcher(self.lltok, grammar)
        if m.is_error():
            raise ValueError(m.get_error())
        return m

    COMPACT = {"whitespace_flexible": False, "item_separator": ", ", "key_separator": ": "}

    def json(self, schema=None):
        """Any valid JSON value matching `schema` (a dict; None = any JSON object). Compact whitespace, so a
        reply can't wander off into blank space and run out of tokens before the JSON is closed."""
        from llguidance import LLMatcher
        return self._matcher(LLMatcher.grammar_from_json_schema({**(schema or {"type": "object"}), "x-guidance": self.COMPACT}))

    def tool_calls(self, tools):
        """One or more <tool_call> blocks whose JSON names a listed function with schema-valid arguments.
        tools: OpenAI-style [{"type": "function", "function": {"name", "parameters"}}, ...]. Parameter lists in
        the shorthand some datasets use ({"arg": {"type": "str, optional"}}) are converted to JSON Schema. If a
        schema still can't be compiled, only the function name is enforced."""
        from llguidance import LLMatcher
        fns = [t.get("function", t) for t in tools]
        try:
            return self._tool_matcher(fns, strict_args=True)
        except ValueError:
            return self._tool_matcher(fns, strict_args=False)

    def _tool_matcher(self, fns, strict_args):
        import json as _json
        from llguidance import LLMatcher
        options = []
        for fn in fns:
            params = to_json_schema(fn.get("parameters")) if strict_args else {"type": "object"}
            options.append({"type": "object", "properties": {"name": {"const": fn["name"]}, "arguments": params},
                            "required": ["name", "arguments"], "additionalProperties": False})
        schema = _json.dumps({"anyOf": options, "x-guidance": self.COMPACT})
        lark = (f'start: call ("\\n" call)*\n'
                f'call: "<tool_call>\\n" body "\\n</tool_call>"\n'
                f'body: %json {schema}\n')
        return self._matcher(LLMatcher.grammar_from_lark(lark))


_SHORT_TYPES = {"str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "number",
                "number": "number", "bool": "boolean", "boolean": "boolean", "list": "array", "array": "array",
                "dict": "object", "object": "object"}


def to_json_schema(params):
    """Tool parameters as JSON Schema, accepting either real JSON Schema or the {"arg": {"type": "str, optional"}}
    shorthand. Unknown types become 'any value'. Only declared arguments are allowed."""
    if not params:
        return {"type": "object", "properties": {}, "additionalProperties": False}
    if params.get("type") == "object" or "properties" in params:
        out = dict(params)
        out.setdefault("additionalProperties", False)
        return out
    props, required = {}, []
    for name, spec in params.items():
        spec = spec if isinstance(spec, dict) else {}
        raw = str(spec.get("type", "")).lower()
        base = raw.split(",")[0].strip().split("[")[0]
        props[name] = {"type": _SHORT_TYPES[base]} if base in _SHORT_TYPES else {}
        if "optional" not in raw:
            required.append(name)
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}
