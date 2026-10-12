"""Deutsch-Test: misst, wie gut ein Modell ist, damit man Trainings vergleichen kann.

Jede Aufgabe ist ein Satzanfang mit mehreren moeglichen Fortsetzungen, von
denen genau eine richtig ist (Grammatik, Wissen, Logik). Das Modell "waehlt"
die Fortsetzung, die es fuer am wahrscheinlichsten haelt. Das funktioniert schon
beim reinen Textmodell, ganz ohne Chat-Training (wie bei HellaSwag/MMLU).

Bewertet wird die Wahrscheinlichkeit pro Zeichen, damit lange Antworten nicht
benachteiligt werden. Zum Vergleich steht die Trefferquote durch Raten dabei.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPT
from .tokenizer import Tokenizer

TESTDATEI = Path(__file__).resolve().parent.parent / "daten" / "test" / "deutsch_test.jsonl"


def load_items(path: Path = TESTDATEI) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@torch.no_grad()
def option_logprob(model: GPT, tok: Tokenizer, kontext: str, option: str) -> float:
    device = next(model.parameters()).device
    ctx = [tok.bos_id] + tok.encode(kontext)
    full = [tok.bos_id] + tok.encode(kontext + option)
    if full[:len(ctx)] != ctx:  # Tokengrenze verschoben: Option getrennt tokenisieren
        full = ctx + tok.encode(option)
    full = full[-model.cfg.max_seq_len:]
    n_opt = len(full) - len(ctx)
    x = torch.tensor([full[:-1]], device=device)
    logits, _, _ = model(x)
    logp = F.log_softmax(logits[0].float(), dim=-1)
    targets = torch.tensor(full[1:], device=device)
    token_lp = logp[torch.arange(len(targets), device=device), targets]
    return float(token_lp[-n_opt:].sum())


def multiple_choice(model: GPT, tok: Tokenizer, items: Optional[list[dict]] = None) -> dict:
    items = items or load_items()
    was_training = model.training
    model.eval()
    per_cat: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0])  # richtig, gesamt, zufall
    for it in items:
        scores = [option_logprob(model, tok, it["kontext"], o) / max(len(o), 1) for o in it["optionen"]]
        c = per_cat[it["kategorie"]]
        c[0] += int(int(np.argmax(scores)) == it["richtig"])
        c[1] += 1
        c[2] += 1 / len(it["optionen"])
    if was_training:
        model.train()
    result = {k: {"richtig": v[0], "gesamt": v[1], "quote": v[0] / v[1], "zufall": v[2] / v[1]}
              for k, v in sorted(per_cat.items())}
    tot = sum(v[0] for v in per_cat.values()), sum(v[1] for v in per_cat.values()), sum(v[2] for v in per_cat.values())
    result["gesamt"] = {"richtig": tot[0], "gesamt": tot[1], "quote": tot[0] / tot[1], "zufall": tot[2] / tot[1]}
    return result


@torch.no_grad()
def perplexity(model: GPT, data_path: Path, n_tokens: int = 200_000) -> Optional[float]:
    if not data_path.exists() or data_path.stat().st_size == 0:
        return None
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    device = next(model.parameters()).device
    T = model.cfg.max_seq_len
    n = min(len(data) - 1, n_tokens)
    model.eval()
    total, count = 0.0, 0
    for start in range(0, n - T, T):
        chunk = torch.from_numpy(data[start:start + T + 1].astype(np.int64)).to(device)
        _, loss, _ = model(chunk[None, :-1], chunk[None, 1:])
        total += loss.item() * T
        count += T
    return math.exp(total / count) if count else None


def report(result: dict, ppl: Optional[float] = None, title: str = "") -> str:
    lines = [f"  Deutsch-Test {title}".rstrip()]
    for k, v in result.items():
        bar = "#" * round(v["quote"] * 20)
        lines.append(f"    {k:10s} {v['richtig']:3d}/{v['gesamt']:<3d} = {v['quote'] * 100:5.1f} %  "
                     f"(Raten: {v['zufall'] * 100:4.1f} %)  {bar}")
    if ppl is not None:
        lines.append(f"    Perplexitaet auf Validierungstext: {ppl:.1f} (kleiner = besser)")
    return "\n".join(lines)


def evaluate_model(model: GPT, tok: Tokenizer, out_dir: Optional[Path] = None,
                   val_path: Optional[Path] = None) -> dict:
    res = multiple_choice(model, tok)
    ppl = perplexity(model, val_path) if val_path else None
    summary = {"test": res, "perplexitaet": ppl}
    if out_dir:
        (out_dir / "bewertung.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
