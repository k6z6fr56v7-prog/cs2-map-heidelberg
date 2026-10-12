"""Muon-Optimizer: lernt pro Rechenschritt mehr als AdamW.

Idee (Keller Jordan 2024, "Muon"; skaliert von Moonshot AI 2025, "Moonlight"):
Fuer die grossen Gewichtsmatrizen im Transformer wird der Momentum-Schritt
"orthogonalisiert", d.h. alle Richtungen der Matrix werden gleich stark
aktualisiert statt nur die paar dominanten. Das geht naeherungsweise mit
fuenf Newton-Schulz-Iterationen (nur Matrixmultiplikationen, schnell auf der GPU).

Embeddings, Normen und das (geteilte) Ausgabe-Embedding laufen weiter mit AdamW.
Die Skalierung 0.2 * sqrt(max(Zeilen, Spalten)) aus Moonlight sorgt dafuer,
dass Muon dieselbe Lernrate und denselben Weight-Decay wie AdamW nutzen kann.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Naeherung fuer U @ V^T aus der Singulaerwertzerlegung G = U S V^T."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """Ein Optimizer mit zwei Arten von Parametergruppen: use_muon=True/False."""

    def __init__(self, param_groups, lr: float = 1e-3, betas=(0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0, momentum: float = 0.95, ns_steps: int = 5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        ns_steps=ns_steps, use_muon=False)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            if group["use_muon"]:
                mom = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(g)
                    g = g.add(buf, alpha=mom)  # Nesterov
                    u = newton_schulz(g, group["ns_steps"])
                    u.mul_(0.2 * math.sqrt(max(p.size(0), p.size(1))))
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(u, alpha=-lr)
            else:
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    state["exp_avg"].lerp_(g, 1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bc1 = 1 - beta1 ** t
                    bc2 = 1 - beta2 ** t
                    denom = (state["exp_avg_sq"] / bc2).sqrt_().add_(group["eps"])
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(state["exp_avg"], denom, value=-lr / bc1)
        return loss


def build_optimizer(model: torch.nn.Module, kind: str, lr: float, weight_decay: float,
                    betas: tuple[float, float], device_type: str) -> torch.optim.Optimizer:
    muon, decay, no_decay = [], [], []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_matrix = p.dim() == 2 and ".layers." in f".{name}"
        if kind == "muon" and is_matrix:
            muon.append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    if kind == "adamw":
        extra = {"fused": True} if device_type == "cuda" else {}
        return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=betas, **extra)
    if kind != "muon":
        raise ValueError(f"Unbekannter Optimizer: {kind}")
    groups = [
        {"params": muon, "use_muon": True, "weight_decay": weight_decay},
        {"params": decay, "use_muon": False, "weight_decay": weight_decay},
        {"params": no_decay, "use_muon": False, "weight_decay": 0.0},
    ]
    return MuonAdamW([g for g in groups if g["params"]], lr=lr, betas=betas)
