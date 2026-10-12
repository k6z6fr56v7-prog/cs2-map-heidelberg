"""Modell- und Trainingskonfiguration.

Die Architektur entspricht exakt der Llama-Architektur (RMSNorm, RoPE, SwiGLU,
Grouped-Query-Attention). Dadurch laesst sich das fertige Modell ohne Umwege
nach Hugging Face / GGUF exportieren und in Ollama oder LM Studio laden.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    dim: int = 512                 # Breite der versteckten Schicht (hidden_size)
    n_layers: int = 8              # Anzahl Transformer-Bloecke
    n_heads: int = 8               # Attention-Koepfe fuer Queries
    n_kv_heads: int = 4            # Koepfe fuer Keys/Values (GQA), muss n_heads teilen
    ffn_dim: int = 1408            # Breite des SwiGLU-Feedforward
    max_seq_len: int = 512         # Kontextlaenge in Tokens
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True    # Ein- und Ausgabe-Embedding teilen (spart Parameter)
    bos_id: int = 1
    eos_id: int = 2

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def validate(self) -> None:
        if self.dim % self.n_heads:
            raise ValueError(f"dim ({self.dim}) muss durch n_heads ({self.n_heads}) teilbar sein")
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"n_heads ({self.n_heads}) muss durch n_kv_heads ({self.n_kv_heads}) teilbar sein")
        if self.head_dim % 2:
            raise ValueError("head_dim muss gerade sein (wegen RoPE)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class TrainConfig:
    # Batch: effektive Tokens pro Schritt = micro_batch * grad_accum * max_seq_len
    micro_batch: int = 16
    grad_accum: int = 4
    lr: float = 1e-3
    optimizer: str = "muon"        # "muon" (schneller lernend) oder "adamw" (klassisch)
    schedule: str = "wsd"          # "wsd" (verlaengerbar) oder "cosine"
    decay_frac: float = 0.2        # WSD: Anteil am Ende, in dem die Lernrate abfaellt
    min_lr_ratio: float = 0.0
    warmup_steps: int = 200
    max_steps: int = 10000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    save_every: int = 500
    sample_every: int = 500
    log_every: int = 10
    seed: int = 1337
    compile: bool = False
    grad_checkpointing: bool = False
    dropout: float = 0.0


@dataclass
class Preset:
    name: str
    beschreibung: str
    model: ModelConfig
    train: TrainConfig
    tokens_empfohlen: int          # wie viele Trainings-Tokens sinnvoll sind
    extra: dict = field(default_factory=dict)


PRESETS: dict[str, Preset] = {
    "test": Preset(
        name="test",
        beschreibung="Winzig, nur zum Ausprobieren ob alles laeuft (Minuten, auch auf CPU).",
        model=ModelConfig(dim=128, n_layers=4, n_heads=4, n_kv_heads=2, ffn_dim=352, max_seq_len=128),
        train=TrainConfig(micro_batch=16, grad_accum=1, lr=2e-3, warmup_steps=50, max_steps=600,
                          eval_every=100, save_every=200, sample_every=200),
        tokens_empfohlen=5_000_000,
    ),
    "mini": Preset(
        name="mini",
        beschreibung="~9 Mio. Parameter. Lernt Rechtschreibung und einfache Saetze. Ca. 1 Stunde auf einer RTX 3050.",
        model=ModelConfig(dim=256, n_layers=6, n_heads=4, n_kv_heads=4, ffn_dim=704, max_seq_len=256),
        train=TrainConfig(micro_batch=32, grad_accum=2, lr=1.5e-3, warmup_steps=200, max_steps=6000),
        tokens_empfohlen=150_000_000,
    ),
    "klein": Preset(
        name="klein",
        beschreibung="~32 Mio. Parameter. Zusammenhaengende Absaetze auf Deutsch. Ca. 6-10 Stunden (eine Nacht).",
        model=ModelConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=4, ffn_dim=1408, max_seq_len=512),
        train=TrainConfig(micro_batch=16, grad_accum=4, lr=1e-3, warmup_steps=300, max_steps=20000),
        tokens_empfohlen=600_000_000,
    ),
    "mittel": Preset(
        name="mittel",
        beschreibung="~88 Mio. Parameter (GPT-2-small-Klasse). Deutlich besser, braucht aber mehrere Naechte.",
        model=ModelConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, ffn_dim=2048, max_seq_len=1024),
        train=TrainConfig(micro_batch=8, grad_accum=8, lr=6e-4, warmup_steps=500, max_steps=40000,
                          grad_checkpointing=True),
        tokens_empfohlen=2_000_000_000,
    ),
}


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
