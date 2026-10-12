"""Text erzeugen: Modell laden, Sampling mit KV-Cache, Chat-Formatierung."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch

from .config import ModelConfig
from .model import GPT
from .tokenizer import ASSISTANT, SYSTEM, USER, StreamDecoder, Tokenizer


def pick_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_checkpoint(path: Path) -> Path:
    """Akzeptiert eine .pt-Datei oder einen Modellordner."""
    path = Path(path)
    if path.is_file():
        return path
    for name in ("modell.pt", "ckpt.pt"):
        if (path / name).exists():
            return path / name
    raise FileNotFoundError(f"Kein Modell (modell.pt / ckpt.pt) in {path} gefunden.")


def find_tokenizer(ckpt: Path) -> Path:
    for cand in (ckpt.parent / "tokenizer.model", ckpt.parent.parent / "tokenizer.model"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"tokenizer.model neben {ckpt} nicht gefunden.")


def load_model(path: Path, device: Optional[torch.device] = None) -> tuple[GPT, Tokenizer, dict]:
    device = device or pick_device()
    ckpt_path = find_checkpoint(path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    tok = Tokenizer(find_tokenizer(ckpt_path))
    return model, tok, ckpt


def _sample_next(
    logits: torch.Tensor,
    history: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    generator: Optional[torch.Generator] = None,
) -> int:
    logits = logits.float().clone()
    if repetition_penalty != 1.0 and history:
        recent = torch.tensor(sorted(set(history[-256:])), device=logits.device)
        vals = logits[recent]
        logits[recent] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        cut = cum - sorted_p > top_p
        sorted_p[cut] = 0
        probs = torch.zeros_like(probs).scatter_(0, sorted_i, sorted_p)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate(
    model: GPT,
    prompt_ids: list[int],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    stop_ids: Optional[set[int]] = None,
    on_token: Optional[Callable[[int], None]] = None,
    seed: Optional[int] = None,
) -> list[int]:
    """Erzeugt Tokens. Nutzt den KV-Cache; wird der Kontext voll, wird
    mit der zweiten Haelfte neu angesetzt (gleitendes Fenster)."""
    device = next(model.parameters()).device
    max_len = model.cfg.max_seq_len
    stop_ids = stop_ids or set()
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    ids = list(prompt_ids)[-(max_len - 1):]
    out: list[int] = []
    past = None
    feed = ids
    for _ in range(max_new_tokens):
        cache_len = past[0][0].shape[2] if past else 0
        if cache_len + len(feed) > max_len:
            # Kontext voll: mit dem juengsten Teil neu beginnen
            ids = ids[-(max_len // 2):]
            feed, past = ids, None
        x = torch.tensor([feed], dtype=torch.long, device=device)
        logits, _, past = model(x, past=past, use_cache=True)
        nxt = _sample_next(logits[0, -1], ids, temperature, top_k, top_p, repetition_penalty, gen)
        if nxt in stop_ids:
            break
        ids.append(nxt)
        out.append(nxt)
        feed = [nxt]
        if on_token:
            on_token(nxt)
    return out


def complete(model: GPT, tok: Tokenizer, text: str, stream: bool = True, **kw) -> str:
    dec = StreamDecoder(tok)
    cb = (lambda t: print(dec.push(t), end="", flush=True)) if stream else None
    ids = generate(model, tok.encode(text, bos=True), stop_ids={tok.eos_id}, on_token=cb, **kw)
    return tok.decode(ids)


# ---------------------------------------------------------------- Chat-Format


def render_chat(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Das Chat-Format. Muss mit sft.py und dem Ollama-Template uebereinstimmen.

    <|system|>\\n{system}\\n<|user|>\\n{frage}\\n<|assistant|>\\n{antwort}</s>
    """
    parts = []
    for m in messages:
        role, content = m["role"], m["content"].strip()
        if role == "system":
            parts.append(f"{SYSTEM}\n{content}\n")
        elif role == "user":
            parts.append(f"{USER}\n{content}\n{ASSISTANT}\n")
        elif role == "assistant":
            parts.append(f"{content}</s>")
    text = "".join(parts)
    if not add_generation_prompt and text.endswith(f"{ASSISTANT}\n"):
        text = text[: -len(f"{ASSISTANT}\n")]
    return text


def encode_chat(tok: Tokenizer, messages: list[dict]) -> list[int]:
    """Tokenisiert einen Chatverlauf fuer die Inferenz (Antwort des Assistenten folgt)."""
    ids = [tok.bos_id]
    for m in messages:
        role, content = m["role"], m["content"].strip()
        if role == "system":
            ids += tok.encode(f"{SYSTEM}\n{content}\n")
        elif role == "user":
            ids += tok.encode(f"{USER}\n{content}\n{ASSISTANT}\n")
        elif role == "assistant":
            ids += tok.encode(content) + [tok.eos_id]
    return ids
