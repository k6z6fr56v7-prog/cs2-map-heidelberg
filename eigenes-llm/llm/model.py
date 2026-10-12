"""Das Sprachmodell: ein Decoder-only-Transformer im Llama-Stil.

Die Modulnamen entsprechen bewusst exakt denen von Hugging Face
`LlamaForCausalLM` (model.embed_tokens, model.layers.N.self_attn.q_proj, ...).
So ist der state_dict direkt exportierbar, ohne Umbenennen.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


def rope_tables(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # gleiche Konvention wie HF (rotate_half)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D), cos/sin: (T, D)
    return (x * cos + rotate_half(x) * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, dropout: float = 0.0):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.dropout = dropout

    def forward(self, x, cos, sin, past: Optional[tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        past_len = 0
        if past is not None:
            past_len = past[0].shape[2]
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        drop = self.dropout if self.training else 0.0
        if past_len == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, dropout_p=drop)
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        else:
            # Mehrere neue Tokens bei vorhandenem Cache: explizite Kausalmaske
            S = past_len + T
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=past_len)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y), present


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, dropout: float = 0.0):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.self_attn = Attention(cfg, dropout)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin, past=None):
        a, present = self.self_attn(self.input_layernorm(x), cos, sin, past)
        x = x + self.resid_drop(a)
        x = x + self.resid_drop(self.mlp(self.post_attention_layernorm(x)))
        return x, present


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig, dropout: float = 0.0):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg, dropout) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig, dropout: float = 0.0):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.model = Backbone(cfg, dropout)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.grad_checkpointing = False
        self.apply(self._init_weights)
        # Residual-Projektionen kleiner initialisieren (stabileres Training tiefer Netze)
        scale = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, include_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not include_embeddings:
            n -= self.model.embed_tokens.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past: Optional[KVCache] = None,
        use_cache: bool = False,
    ):
        B, T = idx.shape
        start = past[0][0].shape[2] if past else 0
        if start + T > self.cfg.max_seq_len:
            raise ValueError(f"Sequenz zu lang: {start + T} > max_seq_len {self.cfg.max_seq_len}")
        cos = self.rope_cos[start:start + T].to(idx.device)
        sin = self.rope_sin[start:start + T].to(idx.device)

        x = self.model.embed_tokens(idx)
        presents: KVCache = []
        for i, layer in enumerate(self.model.layers):
            layer_past = past[i] if past else None
            if self.grad_checkpointing and self.training and not use_cache:
                x, present = checkpoint(layer, x, cos, sin, None, use_reentrant=False)
            else:
                x, present = layer(x, cos, sin, layer_past)
            if use_cache:
                presents.append(present)
        x = self.model.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
            return logits, loss, None
        # Bei Inferenz reicht das letzte Token
        logits = self.lm_head(x[:, -1:, :]) if use_cache else self.lm_head(x)
        return logits, None, (presents if use_cache else None)

    def optimizer(self, lr: float, weight_decay: float, betas: tuple[float, float], device_type: str):
        decay, no_decay = [], []
        seen = set()
        for _, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        extra = {"fused": True} if device_type == "cuda" else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)
