"""Hybrid language model: Gated DeltaNet + gated attention (3:1), Block Attention Residuals.

Architecture follows Qwen3.5 (3 Gated DeltaNet layers per gated full-attention layer,
sigmoid output gate on attention, QK-norm) with Kimi's Block Attention Residuals replacing
the plain residual stream. arch="dense" swaps every DeltaNet layer for gated attention,
which gives a like-for-like Transformer baseline.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_layer: int = 24
    d_model: int = 1024
    arch: str = "hybrid"      # "hybrid": every `attn_every`-th layer is attention, rest DeltaNet; "dense": all attention
    attn_every: int = 4
    n_head: int = 16          # attention query heads
    n_kv_head: int = 4
    head_dim: int = 128
    gdn_heads: int = 16
    gdn_head_dim: int = 128
    ffn_dim: int = 3584
    attnres_blocks: int = 8   # Block AttnRes: the 2*n_layer sublayers are split into this many blocks
    rope_theta: float = 10000.0
    max_seq_len: int = 2048
    softcap: float = 15.0
    grad_ckpt: bool = False   # recompute sublayers in backward to save VRAM


PRESETS = {
    # smoke tests only
    "tiny": dict(n_layer=4, d_model=256, n_head=4, n_kv_head=2, head_dim=64, gdn_heads=4, gdn_head_dim=64,
                 ffn_dim=768, attnres_blocks=4),
    # ~140M non-embedding: architecture bake-off size
    "small": dict(n_layer=12, d_model=768, n_head=12, n_kv_head=3, head_dim=128, gdn_heads=12, gdn_head_dim=128,
                  ffn_dim=2688, attnres_blocks=8),
    # ~500M non-embedding: Qwen3.5-0.8B layout with a 32k vocab
    "base": dict(n_layer=24, d_model=1024, n_head=16, n_kv_head=4, head_dim=128, gdn_heads=16, gdn_head_dim=128,
                 ffn_dim=3584, attnres_blocks=8),
}


def rms(x):
    return F.rms_norm(x, (x.size(-1),))


def inv_rms(x, eps=1e-6):
    return torch.rsqrt(x.float().pow(2).mean(-1) + eps)


class GatedAttention(nn.Module):
    """Causal GQA attention with QK-norm, RoPE and Qwen's elementwise sigmoid output gate."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.nh, self.nkv, self.hd = c.n_head, c.n_kv_head, c.head_dim
        self.q_proj = nn.Linear(c.d_model, 2 * self.nh * self.hd, bias=False)  # query and gate
        self.kv_proj = nn.Linear(c.d_model, 2 * self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, c.d_model, bias=False)
        inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        freqs = torch.outer(torch.arange(c.max_seq_len).float(), inv_freq)
        self.register_buffer("cos", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, :, None, :], persistent=False)

    def rope(self, x):
        T = x.size(1)
        cos, sin = self.cos[:, :T].to(x.dtype), self.sin[:, :T].to(x.dtype)
        x1, x2 = x.chunk(2, -1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)

    def forward(self, x):
        B, T, _ = x.shape
        q, gate = self.q_proj(x).split(self.nh * self.hd, -1)
        q = q.view(B, T, self.nh, self.hd)
        k, v = self.kv_proj(x).view(B, T, 2, self.nkv, self.hd).unbind(2)
        q, k = self.rope(rms(q)), self.rope(rms(k))
        if self.nkv != self.nh:
            k = k.repeat_interleave(self.nh // self.nkv, dim=2)
            v = v.repeat_interleave(self.nh // self.nkv, dim=2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.o_proj(y * torch.sigmoid(gate))


class DeltaNet(nn.Module):
    """Gated DeltaNet (linear attention with a delta-rule memory) from flash-linear-attention."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        from fla.layers import GatedDeltaNet
        self.gdn = GatedDeltaNet(hidden_size=c.d_model, head_dim=c.gdn_head_dim, num_heads=c.gdn_heads,
                                 expand_v=1.0, mode="chunk", use_gate=True, use_short_conv=True, conv_size=4)

    def forward(self, x):
        return self.gdn(x)[0]


class SwiGLU(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(c.d_model, c.ffn_dim, bias=False)
        self.w_up = nn.Linear(c.d_model, c.ffn_dim, bias=False)
        self.w_down = nn.Linear(c.ffn_dim, c.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def attnres_mix(sources, source_inv_rms, query):
    """Block AttnRes: softmax over sources of <query, RMSNorm(source)>, then a weighted sum of sources."""
    logits = torch.stack([(s @ query.to(s.dtype)).float() * r for s, r in zip(sources, source_inv_rms)])
    weights = logits.softmax(0)
    h = weights[0, ..., None] * sources[0]
    for w, s in zip(weights[1:], sources[1:]):
        h = h + w[..., None] * s
    return h


class LM(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.sublayers = nn.ModuleList()
        for i in range(c.n_layer):
            is_attn = c.arch == "dense" or (i % c.attn_every == c.attn_every - 1)
            self.sublayers.append(GatedAttention(c) if is_attn else DeltaNet(c))
            self.sublayers.append(SwiGLU(c))
        n_sub = len(self.sublayers)
        self.block_size = math.ceil(n_sub / c.attnres_blocks)
        # one pseudo-query per sublayer plus one for the final read-out; zero init = uniform average at start
        self.attnres_queries = nn.Parameter(torch.zeros(n_sub + 1, c.d_model))
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        c = self.config
        nn.init.normal_(self.embed.weight, std=1.0)
        nn.init.normal_(self.lm_head.weight, std=0.001)
        s = 3 ** 0.5 * c.d_model ** -0.5
        for m in self.sublayers:
            if isinstance(m, GatedAttention):
                nn.init.uniform_(m.q_proj.weight, -s, s)
                nn.init.uniform_(m.kv_proj.weight, -s, s)
                nn.init.zeros_(m.o_proj.weight)
            elif isinstance(m, SwiGLU):
                nn.init.uniform_(m.w_gate.weight, -s, s)
                nn.init.uniform_(m.w_up.weight, -s, s)
                nn.init.zeros_(m.w_down.weight)
            else:  # DeltaNet keeps FLA's init for its gates/decay; match the rest to the attention layers
                g = m.gdn
                for lin in (g.q_proj, g.k_proj, g.v_proj, g.g_proj):
                    nn.init.uniform_(lin.weight, -s, s)
                nn.init.zeros_(g.o_proj.weight)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() - self.lm_head.weight.numel() if non_embedding else n

    def _run(self, sub, x):
        if self.config.grad_ckpt and self.training:
            return checkpoint(sub, x, use_reentrant=False)
        return sub(x)

    def hidden(self, idx):
        """Final normalized hidden states [B, T, d_model]."""
        x0 = rms(self.embed(idx))
        blocks, blocks_inv = [x0], [inv_rms(x0)]   # completed block sums (block 0 = token embedding)
        partial = None                             # running sum of sublayer outputs in the current block
        for j, sub in enumerate(self.sublayers):
            srcs = blocks if partial is None else blocks + [partial]
            inv = blocks_inv if partial is None else blocks_inv + [inv_rms(partial)]
            h = attnres_mix(srcs, inv, self.attnres_queries[j])
            out = self._run(sub, rms(h)).float()
            partial = out if partial is None else partial + out
            if (j + 1) % self.block_size == 0:
                blocks.append(partial)
                blocks_inv.append(inv_rms(partial))
                partial = None
        srcs = blocks if partial is None else blocks + [partial]
        inv = blocks_inv if partial is None else blocks_inv + [inv_rms(partial)]
        return rms(attnres_mix(srcs, inv, self.attnres_queries[-1]))

    def forward(self, idx, targets=None):
        h = self.hidden(idx)
        if targets is None:
            return self._logits(h)
        # Loss in chunks, recomputing each chunk's logits in backward, so the full
        # (tokens x vocab) fp32 logit tensor never has to sit in VRAM.
        # Targets of -1 (prompts, padding during fine-tuning) are ignored.
        h, targets = h.flatten(0, 1), targets.flatten()
        loss = 0.0
        for i in range(0, h.size(0), self.LOSS_CHUNK):
            loss = loss + checkpoint(self._chunk_loss, h[i:i + self.LOSS_CHUNK], targets[i:i + self.LOSS_CHUNK],
                                     use_reentrant=False)
        return loss / (targets >= 0).sum().clamp(min=1)

    LOSS_CHUNK = 4096

    def _logits(self, h):
        logits = self.lm_head(h).float()
        if self.config.softcap:
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
        return logits

    def _chunk_loss(self, h, targets):
        return F.cross_entropy(self._logits(h), targets, reduction="sum", ignore_index=-1)

    def token_logprobs(self, idx, targets):
        """Log-probability of each target token, [B, T] (0 where the target is -1). Chunked like the loss."""
        h, t = self.hidden(idx).flatten(0, 1), targets.flatten()
        parts = [checkpoint(self._chunk_logprobs, h[i:i + self.LOSS_CHUNK], t[i:i + self.LOSS_CHUNK], use_reentrant=False)
                 for i in range(0, h.size(0), self.LOSS_CHUNK)]
        return torch.cat(parts).view(targets.shape)

    def _chunk_logprobs(self, h, targets):
        lp = torch.log_softmax(self._logits(h), -1).gather(1, targets.clamp(min=0)[:, None])[:, 0]
        return lp * (targets >= 0)

    def param_groups(self):
        """Split parameters: hidden matrices go to Muon, everything else to AdamW."""
        muon, other = [], []
        for name, p in self.sublayers.named_parameters():
            (muon if p.ndim == 2 and min(p.shape) >= 64 else other).append(p)
        other.append(self.attnres_queries)
        return dict(muon=muon, embed=[self.embed.weight], head=[self.lm_head.weight], other=other)


def resize_vocab(state_dict, new_rows):
    """Grow the embedding and output matrices for added tokens; new rows start at the mean of the old ones."""
    for key in ("embed.weight", "lm_head.weight"):
        w = state_dict[key]
        if w.size(0) < new_rows:
            extra = w.float().mean(0, keepdim=True).expand(new_rows - w.size(0), -1).to(w.dtype)
            state_dict[key] = torch.cat([w, extra])
    return state_dict


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=50):
    """Simple sampling that re-runs the full context each step (no cache; fine for short samples)."""
    for _ in range(max_new_tokens):
        ctx = idx[:, -model.config.max_seq_len:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ctx)[:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
        idx = torch.cat([idx, nxt], 1)
    return idx
