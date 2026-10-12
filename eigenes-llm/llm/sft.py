"""Chat-Fine-Tuning (SFT): aus dem Textmodell wird ein Assistent.

Das Modell lernt nur auf den Antworten des Assistenten (die Fragen werden in
der Loss-Berechnung ausgeblendet). Unterstuetzte Datenformate (JSONL oder
Hugging-Face-Datensatz):

    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}
    {"instruction": ..., "input": ..., "output": ...}           (Alpaca)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch

from .tokenizer import ASSISTANT, SYSTEM, USER, Tokenizer

STANDARD_DATENSATZ = "FreedomIntelligence/alpaca-gpt4-deutsch"
IDENTITAET = Path(__file__).resolve().parent.parent / "daten" / "sft" / "identitaet.jsonl"

ROLLEN = {
    "human": "user", "user": "user", "prompter": "user", "question": "user",
    "gpt": "assistant", "assistant": "assistant", "bot": "assistant", "model": "assistant",
    "system": "system",
}


def normalize(row: dict) -> Optional[list[dict]]:
    """Bringt verschiedene Datensatzformate auf eine Liste von {role, content}."""
    if "messages" in row and isinstance(row["messages"], list):
        msgs = [{"role": ROLLEN.get(str(m.get("role", "")).lower(), ""), "content": m.get("content") or ""}
                for m in row["messages"]]
    elif "conversations" in row and isinstance(row["conversations"], list):
        msgs = [{"role": ROLLEN.get(str(m.get("from", m.get("role", ""))).lower(), ""),
                 "content": m.get("value", m.get("content")) or ""} for m in row["conversations"]]
    elif "instruction" in row and "output" in row:
        frage = row["instruction"].strip()
        if row.get("input"):
            frage += "\n\n" + row["input"].strip()
        msgs = [{"role": "user", "content": frage}, {"role": "assistant", "content": row["output"]}]
    else:
        return None
    msgs = [m for m in msgs if m["role"] and m["content"].strip()]
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    return msgs


def tokenize_chat(tok: Tokenizer, msgs: list[dict], max_len: int) -> Optional[tuple[list[int], list[int]]]:
    """Gibt (ids, labels) zurueck; labels = -100 ueberall ausser auf Antworten."""
    ids = [tok.bos_id]
    labels = [-100]
    for m in msgs:
        content = m["content"].strip()
        if m["role"] == "system":
            part = tok.encode(f"{SYSTEM}\n{content}\n")
            ids += part
            labels += [-100] * len(part)
        elif m["role"] == "user":
            part = tok.encode(f"{USER}\n{content}\n{ASSISTANT}\n")
            ids += part
            labels += [-100] * len(part)
        else:
            part = tok.encode(content) + [tok.eos_id]
            ids += part
            labels += part
    if len(ids) > max_len:
        ids, labels = ids[:max_len], labels[:max_len]
    if all(l == -100 for l in labels[1:]):
        return None
    return ids, labels


def iter_rows(source: str) -> Iterator[dict]:
    p = Path(source)
    if p.exists():
        files = [p] if p.is_file() else sorted(p.glob("*.jsonl")) + sorted(p.glob("*.json"))
        for f in files:
            text = f.read_text(encoding="utf-8")
            if f.suffix == ".json" and text.lstrip().startswith("["):
                yield from json.loads(text)
            else:
                for line in text.splitlines():
                    if line.strip():
                        yield json.loads(line)
        return
    from datasets import load_dataset

    for row in load_dataset(source, split="train"):
        yield row


def prepare(
    sources: Iterable[str],
    tok: Tokenizer,
    max_len: int,
    out_path: Path,
    identity_repeat: int = 20,
    max_examples: Optional[int] = None,
    max_chars: int = 6000,
    name: str = "Eigen-LM",
    ersteller: str = "Leo",
) -> dict:
    examples: list[tuple[list[int], list[int]]] = []
    skipped = 0
    for src in sources:
        n_src = 0
        for row in iter_rows(src):
            msgs = normalize(row)
            if not msgs or sum(len(m["content"]) for m in msgs) > max_chars:
                skipped += 1
                continue
            res = tokenize_chat(tok, msgs, max_len)
            if res is None:
                skipped += 1
                continue
            examples.append(res)
            n_src += 1
            if max_examples and n_src >= max_examples:
                break
        print(f"  {src}: {n_src:,} Beispiele")
    # Validierung nur aus echten Datensatz-Beispielen (die wiederholten
    # Identitaets-Beispiele wuerden sonst auch in der Validierung landen)
    rng = random.Random(0)
    rng.shuffle(examples)
    n_val = min(500, len(examples) // 50) if len(examples) >= 10 else 0
    val, examples = examples[:n_val], examples[n_val:]
    if identity_repeat and IDENTITAET.exists():
        ident = []
        for row in iter_rows(str(IDENTITAET)):
            msgs = normalize(row)
            for m in msgs or []:
                m["content"] = m["content"].replace("{name}", name).replace("{ersteller}", ersteller)
            res = tokenize_chat(tok, msgs, max_len) if msgs else None
            if res:
                ident.append(res)
        examples += ident * identity_repeat
        print(f"  Identitaet: {len(ident)} Beispiele x {identity_repeat}")
    if not examples:
        raise SystemExit("Keine verwertbaren Chat-Beispiele gefunden.")
    rng.shuffle(examples)
    if not val:  # sehr kleine Datensaetze: Validierung = Stichprobe aus dem Training
        val = examples[: max(1, len(examples) // 10)]
    data = {"train": examples, "val": val, "max_len": max_len}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    n_tok = sum(len(i) for i, _ in examples)
    print(f"  Gesamt: {len(examples):,} Trainings- und {len(val):,} Validierungsbeispiele, {n_tok / 1e6:.1f} Mio. Tokens, {skipped:,} uebersprungen")
    return {"beispiele": len(examples), "tokens": n_tok}


class ChatData:
    def __init__(self, path: Path, pad_id: int, seed: int = 0):
        d = torch.load(path, weights_only=False)
        self.train, self.val = d["train"], d["val"]
        self.pad_id = pad_id
        self.rng = random.Random(seed)
        self.train_tokens = sum(len(i) for i, _ in self.train)

    def batch(self, split: str, batch_size: int, device):
        pool = self.train if split == "train" else self.val
        items = [pool[self.rng.randrange(len(pool))] for _ in range(batch_size)]
        # Eingabe = ids[:-1], Ziel = labels[1:] (naechstes Token vorhersagen)
        L = max(len(i) for i, _ in items) - 1
        x = torch.full((batch_size, L), self.pad_id, dtype=torch.long)
        y = torch.full((batch_size, L), -100, dtype=torch.long)
        for b, (ids, labels) in enumerate(items):
            n = len(ids) - 1
            x[b, :n] = torch.tensor(ids[:-1])
            y[b, :n] = torch.tensor(labels[1:])
        return x.to(device), y.to(device)
