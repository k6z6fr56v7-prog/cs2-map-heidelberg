"""Daten: Text herunterladen, Tokenizer-Stichprobe schreiben, tokenisieren, laden.

Ablauf auf der Festplatte (alles unter dem Datenordner, Standard `arbeit/`):

    raw/teil_00000.jsonl ...     ein Dokument pro Zeile: {"text": "..."}
    tokenizer_stichprobe.txt     Absaetze, eine Zeile pro Absatz (fuer SentencePiece)
    tokenizer.model              der trainierte Tokenizer
    train.bin / val.bin          alle Tokens als uint16, Dokumente getrennt durch BOS/EOS
    daten.json                   Statistik
"""

from __future__ import annotations

import json
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .tokenizer import Tokenizer

QUELLEN = {
    "fineweb2": {
        "hf": ("HuggingFaceFW/fineweb-2", "deu_Latn"),
        "info": "Deutscher Webtext, gefiltert (FineWeb-2). Gross und abwechslungsreich.",
    },
    "wikipedia": {
        "hf": ("wikimedia/wikipedia", "20231101.de"),
        "info": "Deutsche Wikipedia. Sauber, sachlich, etwas einseitig im Stil.",
    },
}

MIN_DOC_CHARS = 200
TEIL_GROESSE_ZEICHEN = 100_000_000  # ~100 MB pro raw-Datei


def _hf_stream(name: str) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Bitte zuerst installieren: pip install datasets") from e
    repo, cfg = QUELLEN[name]["hf"]
    ds = load_dataset(repo, cfg, split="train", streaming=True)
    for row in ds:
        yield row["text"]


def _local_stream(folder: Path) -> Iterator[str]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".txt", ".md", ".jsonl"})
    if not files:
        raise SystemExit(f"Keine .txt/.md/.jsonl-Dateien in {folder} gefunden.")
    for p in files:
        if p.suffix.lower() == ".jsonl":
            with p.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line).get("text", "")
        else:
            text = p.read_text(encoding="utf-8", errors="replace")
            # Lange Dateien (z.B. Buecher) an Leerzeilen in handliche Dokumente teilen
            chunk: list[str] = []
            size = 0
            for para in text.split("\n\n"):
                chunk.append(para)
                size += len(para)
                if size > 20_000:
                    yield "\n\n".join(chunk)
                    chunk, size = [], 0
            if chunk:
                yield "\n\n".join(chunk)


def text_stream(source: str) -> Iterator[str]:
    if source in QUELLEN:
        return _hf_stream(source)
    path = Path(source)
    if path.exists():
        return _local_stream(path)
    raise SystemExit(f"Unbekannte Quelle '{source}'. Moeglich: {', '.join(QUELLEN)} oder ein Ordner mit Textdateien.")


def download(source: str, work: Path, max_chars: int, sample_chars: int = 300_000_000) -> dict:
    """Laedt Text bis `max_chars` Zeichen und schreibt raw-Dateien + Tokenizer-Stichprobe."""
    raw = work / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for old in raw.glob("teil_*.jsonl"):
        old.unlink()

    sample_path = work / "tokenizer_stichprobe.txt"
    sample = sample_path.open("w", encoding="utf-8")
    teil, teil_chars, total_chars, docs, sample_written = 0, 0, 0, 0, 0
    out = (raw / f"teil_{teil:05d}.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    last_print = 0.0
    try:
        for text in text_stream(source):
            text = text.strip()
            if len(text) < MIN_DOC_CHARS:
                continue
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            docs += 1
            teil_chars += len(text)
            total_chars += len(text)
            if sample_written < sample_chars:
                for para in text.split("\n"):
                    para = para.strip()
                    if len(para) >= 20:
                        sample.write(para + "\n")
                        sample_written += len(para)
            if teil_chars >= TEIL_GROESSE_ZEICHEN:
                out.close()
                teil += 1
                teil_chars = 0
                out = (raw / f"teil_{teil:05d}.jsonl").open("w", encoding="utf-8")
            now = time.time()
            if now - last_print > 5:
                last_print = now
                pct = 100 * total_chars / max_chars
                rate = total_chars / max(now - t0, 1e-6) / 1e6
                print(f"\r  {total_chars / 1e6:8.0f} / {max_chars / 1e6:.0f} Mio. Zeichen ({pct:5.1f} %)  "
                      f"{docs:,} Dokumente  {rate:.1f} Mio. Zeichen/s   ", end="", flush=True)
            if total_chars >= max_chars:
                break
    except KeyboardInterrupt:
        print("\n  Abgebrochen - bisher geladene Daten werden behalten.")
    finally:
        out.close()
        sample.close()
    print()
    stats = {"quelle": source, "dokumente": docs, "zeichen": total_chars, "teile": teil + 1}
    (work / "download.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


# ---------------------------------------------------------------- Tokenisieren

_worker_tok: Optional[Tokenizer] = None


def _init_worker(tok_path: str) -> None:
    global _worker_tok
    _worker_tok = Tokenizer(tok_path)


def _encode_file(args: tuple[str, int]) -> tuple[np.ndarray, np.ndarray, int]:
    path, val_every = args
    assert _worker_tok is not None
    tok = _worker_tok
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    train_parts, val_parts = [], []
    batch = 512
    doc_index = 0
    for i in range(0, len(texts), batch):
        for ids in tok.encode_batch(texts[i:i + batch]):
            arr = np.empty(len(ids) + 2, dtype=np.uint16)
            arr[0] = tok.bos_id
            arr[1:-1] = ids
            arr[-1] = tok.eos_id
            (val_parts if doc_index % val_every == val_every - 1 else train_parts).append(arr)
            doc_index += 1
    cat = lambda parts: np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)
    return cat(train_parts), cat(val_parts), doc_index


def tokenize(work: Path, tok_path: Path, workers: Optional[int] = None, val_every: int = 200) -> dict:
    files = sorted((work / "raw").glob("teil_*.jsonl"))
    if not files:
        raise SystemExit(f"Keine Rohdaten in {work / 'raw'} - zuerst 'daten' ausfuehren.")
    tok = Tokenizer(tok_path)
    if tok.vocab_size > 65535:
        raise SystemExit("Vokabular zu gross fuer uint16-Speicherung.")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    n_train = n_val = n_docs = 0
    t0 = time.time()
    with open(work / "train.bin", "wb") as ftrain, open(work / "val.bin", "wb") as fval:
        with Pool(workers, initializer=_init_worker, initargs=(str(tok_path),)) as pool:
            for i, (tr, va, nd) in enumerate(pool.imap(_encode_file, [(str(f), val_every) for f in files])):
                tr.tofile(ftrain)
                va.tofile(fval)
                n_train += len(tr)
                n_val += len(va)
                n_docs += nd
                print(f"\r  Datei {i + 1}/{len(files)}  {n_train / 1e6:,.1f} Mio. Tokens  "
                      f"({time.time() - t0:.0f} s)   ", end="", flush=True)
    print()
    stats = {
        "train_tokens": n_train,
        "val_tokens": n_val,
        "dokumente": n_docs,
        "vocab_size": tok.vocab_size,
        "tokenizer": str(tok_path.name),
    }
    (work / "daten.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


# ---------------------------------------------------------------- Laden


class TokenData:
    """Zieht zufaellige Ausschnitte aus train.bin / val.bin (per Memory-Map, kaum RAM)."""

    def __init__(self, work: Path, seq_len: int, seed: int = 0):
        self.seq_len = seq_len
        self.train = np.memmap(work / "train.bin", dtype=np.uint16, mode="r")
        val_path = work / "val.bin"
        has_val = val_path.exists() and val_path.stat().st_size > 0
        self.val = np.memmap(val_path, dtype=np.uint16, mode="r") if has_val else np.empty(0, dtype=np.uint16)
        if len(self.train) <= seq_len + 1:
            raise SystemExit("Zu wenig Trainingsdaten fuer diese Kontextlaenge.")
        if len(self.val) <= seq_len + 1:
            # Sehr kleine Datensaetze: Validierung aus dem Ende von train nehmen
            self.val = self.train[-max(len(self.train) // 20, seq_len + 2):]
        self.rng = random.Random(seed)

    def batch(self, split: str, batch_size: int, device):
        import torch

        data = self.train if split == "train" else self.val
        hi = len(data) - self.seq_len - 1
        starts = [self.rng.randint(0, hi) for _ in range(batch_size)]
        chunk = np.stack([data[s:s + self.seq_len + 1].astype(np.int64) for s in starts])
        t = torch.from_numpy(chunk)
        x, y = t[:, :-1], t[:, 1:]
        if getattr(device, "type", str(device)) == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
