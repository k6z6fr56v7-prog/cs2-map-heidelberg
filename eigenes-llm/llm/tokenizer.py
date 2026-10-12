"""Tokenizer: SentencePiece-BPE mit Byte-Fallback (wie bei Llama 2).

Warum SentencePiece? llama.cpp (und damit Ollama/LM Studio) kann diesen
Tokenizer-Typ direkt aus `tokenizer.model` uebernehmen.

Wichtige Entscheidung: `add_dummy_prefix=False`. Dadurch ergibt das
Tokenisieren von Textstuecken einzeln exakt dasselbe wie das Tokenisieren des
zusammengesetzten Texts (solange an Sonderzeichen oder Zeilenumbruechen
getrennt wird). Das braucht das Chat-Fine-Tuning, damit Training und
Inferenz in Ollama identische Token sehen.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import sentencepiece as spm

SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
CHAT_TOKENS = [SYSTEM, USER, ASSISTANT]


def train_tokenizer(
    text_files: Iterable[Path],
    out_prefix: Path,
    vocab_size: int = 16000,
    max_sentences: int = 3_000_000,
    threads: int = 8,
) -> Path:
    files = [str(p) for p in text_files]
    if not files:
        raise FileNotFoundError("Keine Textdateien fuer das Tokenizer-Training gefunden.")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        _train(files, out_prefix, vocab_size, max_sentences, threads)
    except RuntimeError as e:
        # Bei wenig Text kann SentencePiece nicht beliebig viele Tokens bilden
        m = re.search(r"value <= (\d+)", str(e))
        if not m:
            raise
        smaller = int(m.group(1))
        print(f"  Hinweis: zu wenig Text fuer {vocab_size} Tokens - verwende {smaller}.")
        _train(files, out_prefix, smaller, max_sentences, threads)
    return out_prefix.with_suffix(".model")


def _train(files: list[str], out_prefix: Path, vocab_size: int, max_sentences: int, threads: int) -> None:
    spm.SentencePieceTrainer.train(
        input=files,
        model_prefix=str(out_prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=0.9995,
        byte_fallback=True,
        split_digits=True,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True,
        normalization_rule_name="identity",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        user_defined_symbols=CHAT_TOKENS,
        input_sentence_size=max_sentences,
        shuffle_input_sentence=True,
        max_sentence_length=16384,
        num_threads=threads,
        minloglevel=2,
    )


class Tokenizer:
    def __init__(self, model_path: Path | str):
        self.path = Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Tokenizer nicht gefunden: {self.path}")
        self.sp = spm.SentencePieceProcessor(model_file=str(self.path))
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.vocab_size = self.sp.get_piece_size()
        self.system_id = self.sp.piece_to_id(SYSTEM)
        self.user_id = self.sp.piece_to_id(USER)
        self.assistant_id = self.sp.piece_to_id(ASSISTANT)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = self.sp.encode(text)
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self.sp.encode(texts)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    def id_to_piece(self, i: int) -> str:
        return self.sp.id_to_piece(i)


class StreamDecoder:
    """Dekodiert Token fuer Token, ohne kaputte UTF-8-Zeichen auszugeben.

    Umlaute bestehen bei Byte-Fallback teils aus mehreren Byte-Token. Wir
    dekodieren deshalb immer die ganze Folge und geben nur den neuen,
    vollstaendigen Teil aus.
    """

    def __init__(self, tok: Tokenizer, lstrip: bool = False):
        self.tok = tok
        self.ids: list[int] = []
        self.emitted = ""
        self.lstrip = lstrip  # Leerzeichen am Antwortanfang weglassen

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):  # unvollstaendiges Mehrbyte-Zeichen
            return ""
        if self.lstrip:
            text = text.lstrip()
        new = text[len(self.emitted):]
        self.emitted = text
        return new
