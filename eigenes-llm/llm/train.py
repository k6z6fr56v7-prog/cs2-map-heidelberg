"""Training (Vortraining und Fine-Tuning teilen sich diese Schleife).

Besonderheiten fuer den Heim-PC:
- bf16 auf RTX-30xx und neuer, sonst fp16 mit GradScaler, auf CPU fp32
- Zeitbudget (`--stunden 6`): die Lernrate wird so geplant, dass das Training
  nach der angegebenen Zeit sauber endet
- Strg+C speichert einen Checkpoint, `--weiter` setzt fort
- Bei "out of memory" wird der Micro-Batch automatisch halbiert
"""

from __future__ import annotations

import contextlib
import csv
import math
import os
import shutil
import signal
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Optional

import torch

from .config import ModelConfig, TrainConfig
from .generate import generate
from .model import GPT
from .optim import build_optimizer
from .tokenizer import StreamDecoder, Tokenizer

BatchFn = Callable[[str, int, torch.device], tuple[torch.Tensor, torch.Tensor]]

BEISPIEL_ANFAENGE = [
    "Heidelberg ist eine Stadt",
    "Die wichtigste Erfindung des 20. Jahrhunderts",
    "Am Morgen ging sie",
]


def fmt_time(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def lr_at(progress: float, step: int, tc: TrainConfig) -> float:
    """Lernrate. WSD = Warmup, dann konstant (Stable), am Ende Abkuehlen (Decay).

    Vorteil von WSD: Der Stand vor dem Abkuehlen wird gespeichert. Von dort
    kann man das Training spaeter beliebig verlaengern (--verlaengern).
    Abkuehlen mit 1-sqrt-Form (Haegele et al. 2024).
    """
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    progress = min(max(progress, 0.0), 1.0)
    lo = tc.lr * tc.min_lr_ratio
    if tc.schedule == "cosine":
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return lo + (tc.lr - lo) * cos
    start = 1.0 - tc.decay_frac
    if progress < start:
        return tc.lr
    x = (progress - start) / max(tc.decay_frac, 1e-9)
    return lo + (tc.lr - lo) * (1 - math.sqrt(x))


def progress_at_start_beyond_stable(step: int, tc: TrainConfig, hours: Optional[float], elapsed: float) -> bool:
    p = step / tc.max_steps
    if hours:
        p = max(p, elapsed / (hours * 3600))
    return p >= 1 - tc.decay_frac


def setup_precision(device: torch.device):
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.autocast("cuda", dtype=torch.bfloat16), None, "bf16"
        return torch.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda"), "fp16"
    return contextlib.nullcontext(), None, "fp32"


def save_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"{p.name} ({p.total_memory / 2**30:.1f} GB VRAM)"
    return device.type.upper()


def train(
    model_cfg: ModelConfig,
    tc: TrainConfig,
    get_batch: BatchFn,
    out_dir: Path,
    tokenizer_path: Path,
    device: torch.device,
    hours: Optional[float] = None,
    resume: bool = False,
    extend: bool = False,
    init_weights: Optional[dict] = None,
    tokens_available: Optional[int] = None,
    run_name: str = "vortraining",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if tokenizer_path.resolve() != (out_dir / "tokenizer.model").resolve():
        shutil.copy(tokenizer_path, out_dir / "tokenizer.model")
    tok = Tokenizer(out_dir / "tokenizer.model")
    torch.manual_seed(tc.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = GPT(model_cfg, dropout=tc.dropout).to(device)
    model.grad_checkpointing = tc.grad_checkpointing

    step, elapsed_before, tokens_seen, best_val = 0, 0.0, 0, float("inf")
    ckpt_path = out_dir / "ckpt.pt"
    stable_path = out_dir / "ckpt_stabil.pt"
    load_path = stable_path if extend else ckpt_path
    if extend and not stable_path.exists():
        raise SystemExit(f"{stable_path} fehlt - Verlaengern geht nur nach einem WSD-Training.")
    ck = None
    if (resume or extend) and load_path.exists():
        ck = torch.load(load_path, map_location="cpu", weights_only=False)
        saved_tc = ck.get("train_cfg", {})
        # Optimizer und Batch so wie beim urspruenglichen Training (sonst passt der Zustand nicht)
        tc = replace(tc, micro_batch=saved_tc.get("micro_batch", tc.micro_batch),
                     grad_accum=saved_tc.get("grad_accum", tc.grad_accum),
                     optimizer=saved_tc.get("optimizer", "adamw"),
                     schedule=saved_tc.get("schedule", "cosine"),
                     decay_frac=saved_tc.get("decay_frac", tc.decay_frac))
    elif resume:
        print("  Kein Checkpoint gefunden - starte neu.")

    opt = build_optimizer(model, tc.optimizer, tc.lr, tc.weight_decay, (tc.beta1, tc.beta2), device.type)

    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        elapsed_before = ck.get("elapsed", 0.0)
        tokens_seen = ck.get("tokens_seen", 0)
        best_val = ck.get("best_val", float("inf"))
        saved_hours = ck.get("hours")
        saved_steps = ck.get("train_cfg", {}).get("max_steps", tc.max_steps)
        if extend:
            # Ohne neue Vorgabe: Gesamtdauer verdoppeln
            if saved_hours and not hours:
                hours = saved_hours * 2
            if not saved_hours and tc.max_steps <= saved_steps:
                tc = replace(tc, max_steps=saved_steps * 2)
            ziel = f"{hours:g} Stunden insgesamt" if hours else f"{tc.max_steps:,} Schritte insgesamt"
            print(f"  Verlaengere ab Schritt {step} (Stand vor dem Abkuehlen), neues Ziel: {ziel}")
        else:
            hours = hours or saved_hours
            print(f"  Setze fort bei Schritt {step} ({tokens_seen / 1e6:.1f} Mio. Tokens gesehen)")
    elif init_weights is not None:
        missing, unexpected = model.load_state_dict(init_weights, strict=False)
        if unexpected:
            print(f"  Hinweis: unerwartete Gewichte ignoriert: {unexpected[:3]}...")

    raw_model = model
    if tc.compile:
        print("  Kompiliere Modell (torch.compile)...")
        model = torch.compile(model)

    autocast, scaler, prec = setup_precision(device)
    T = model_cfg.max_seq_len

    def tokens_per_step() -> int:
        return tc.micro_batch * tc.grad_accum * T

    n_params = raw_model.num_params()
    print(f"  Geraet:      {describe_device(device)}, Praezision {prec}")
    print(f"  Modell:      {n_params / 1e6:.1f} Mio. Parameter "
          f"({raw_model.num_params(False) / 1e6:.1f} Mio. ohne Embeddings)")
    print(f"  Batch:       {tc.micro_batch} x {tc.grad_accum} x {T} = {tokens_per_step():,} Tokens pro Schritt")
    print(f"  Optimizer:   {tc.optimizer}, Lernrate {tc.lr:g}, Plan {tc.schedule}")
    if hours:
        print(f"  Zeitbudget:  {hours:g} Stunden (oder max. {tc.max_steps:,} Schritte)")
    else:
        print(f"  Schritte:    {tc.max_steps:,}  (~{tc.max_steps * tokens_per_step() / 1e6:,.0f} Mio. Tokens)")
    if tokens_available:
        epochs = tc.max_steps * tokens_per_step() / tokens_available
        if epochs > 4 and not hours:
            print(f"  Achtung: das sind ~{epochs:.1f} Durchlaeufe ueber die Daten - eher mehr Daten laden.")
    if device.type == "cpu":
        print("  Achtung: Training auf der CPU ist sehr langsam. Ist PyTorch mit CUDA installiert?")

    log_path = out_dir / f"{run_name}_log.csv"
    new_log = not log_path.exists() or not resume
    log_file = log_path.open("w" if new_log else "a", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["schritt", "loss", "val_loss", "lr", "tokens", "sekunden", "tokens_pro_s"])

    stop_requested = {"flag": False}

    def on_sigint(signum, frame):
        if stop_requested["flag"]:
            raise KeyboardInterrupt
        stop_requested["flag"] = True
        print("\n  Strg+C erkannt - speichere nach diesem Schritt. (Nochmal druecken = sofort abbrechen)")

    old_handler = signal.signal(signal.SIGINT, on_sigint)

    def checkpoint(final: bool = False, path: Optional[Path] = None) -> None:
        state = {
            "model": raw_model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "elapsed": elapsed_before + (time.time() - t_start),
            "tokens_seen": tokens_seen,
            "best_val": best_val,
            "model_cfg": model_cfg.to_dict(),
            "train_cfg": asdict(tc),
            "hours": hours,
            "run": run_name,
        }
        save_atomic(state, path or ckpt_path)
        if final:
            export_weights(out_dir / "modell.pt")

    def export_weights(path: Path) -> None:
        save_atomic({"model": raw_model.state_dict(), "model_cfg": model_cfg.to_dict(), "step": step,
                     "tokens_seen": tokens_seen, "val_loss": best_val, "run": run_name}, path)

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        losses = []
        for _ in range(tc.eval_batches):
            x, y = get_batch("val", tc.micro_batch, device)
            with autocast:
                _, loss, _ = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    def show_samples() -> None:
        raw_model.eval()
        for start in BEISPIEL_ANFAENGE[:2]:
            dec = StreamDecoder(tok)
            ids = generate(raw_model, tok.encode(start, bos=True), max_new_tokens=60, temperature=0.8,
                           top_k=40, stop_ids={tok.eos_id}, seed=step)
            text = "".join(dec.push(i) for i in ids).replace("\n", " ")
            print(f"    > {start}{text}")
        raw_model.train()

    model.train()
    t_start = time.time()
    t_log = time.time()
    tokens_at_log = tokens_seen
    loss_acc, loss_n = 0.0, 0
    oom_retries = 0
    stable_saved = extend or (tc.schedule == "wsd" and progress_at_start_beyond_stable(step, tc, hours,
                                                                                      elapsed_before))

    def progress() -> float:
        p = step / tc.max_steps
        if hours:
            p = max(p, (elapsed_before + time.time() - t_start) / (hours * 3600))
        return p

    try:
        while progress() < 1.0 and not stop_requested["flag"]:
            lr = lr_at(progress(), step, tc)
            for g in opt.param_groups:
                g["lr"] = lr
            try:
                for micro in range(tc.grad_accum):
                    x, y = get_batch("train", tc.micro_batch, device)
                    with autocast:
                        _, loss, _ = model(x, y)
                        loss = loss / tc.grad_accum
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    loss_acc += loss.item()
                if scaler:
                    scaler.unscale_(opt)
                if tc.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tc.grad_clip)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                loss_acc = 0.0
                torch.cuda.empty_cache()
                if tc.micro_batch == 1 or oom_retries > 6:
                    raise SystemExit("Grafikspeicher reicht nicht - kleineres Preset oder kuerzere Kontextlaenge waehlen.")
                oom_retries += 1
                tc = replace(tc, micro_batch=tc.micro_batch // 2, grad_accum=tc.grad_accum * 2)
                print(f"  Grafikspeicher voll -> Micro-Batch {tc.micro_batch}, Akkumulation {tc.grad_accum}")
                continue

            step += 1
            tokens_seen += tokens_per_step()
            loss_n += 1
            if tc.schedule == "wsd" and not stable_saved and progress() >= 1 - tc.decay_frac:
                stable_saved = True
                checkpoint(path=stable_path)
                print("  -- Stand vor dem Abkuehlen gesichert (ckpt_stabil.pt, fuer --verlaengern)")

            if step % tc.log_every == 0:
                now = time.time()
                dt = max(now - t_log, 1e-6)
                tps = (tokens_seen - tokens_at_log) / dt
                sec_per_step = dt / loss_n
                t_log, tokens_at_log = now, tokens_seen
                avg = loss_acc / loss_n
                loss_acc, loss_n = 0.0, 0
                elapsed = elapsed_before + now - t_start
                p = progress()
                eta = (tc.max_steps - step) * sec_per_step
                if hours:
                    eta = min(eta, hours * 3600 - elapsed)
                mem = f"  {torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if device.type == "cuda" else ""
                print(f"  Schritt {step:6d} | loss {avg:.3f} | lr {lr:.2e} | {tps / 1000:6.1f}k Tok/s | "
                      f"{p * 100:5.1f} % | {fmt_time(elapsed)} vergangen, ~{fmt_time(eta)} uebrig{mem}")
                log.writerow([step, f"{avg:.4f}", "", f"{lr:.3e}", tokens_seen, int(elapsed), int(tps)])
                log_file.flush()

            if step % tc.eval_every == 0:
                val = evaluate()
                ppl = math.exp(min(val, 20))
                better = val < best_val
                if better:
                    best_val = val
                    export_weights(out_dir / "modell.pt")
                print(f"  -- Validierung: loss {val:.3f} (Perplexitaet {ppl:.1f})"
                      f"{'  * bestes bisher, gespeichert' if better else ''}")
                log.writerow([step, "", f"{val:.4f}", "", tokens_seen, int(elapsed_before + time.time() - t_start), ""])
                log_file.flush()

            if step % tc.sample_every == 0:
                show_samples()
            if step % tc.save_every == 0:
                checkpoint()
    finally:
        signal.signal(signal.SIGINT, old_handler)
        log_file.close()

    val = evaluate()
    if val < best_val:
        best_val = val
    checkpoint(final=False)
    # modell.pt: das zuletzt trainierte Modell, wenn es nicht schlechter als das beste ist
    if val <= best_val + 1e-6:
        export_weights(out_dir / "modell.pt")
    elapsed = elapsed_before + time.time() - t_start
    print(f"\n  Fertig nach {step:,} Schritten, {fmt_time(elapsed)}. Validierungs-Loss {val:.3f} "
          f"(Perplexitaet {math.exp(min(val, 20)):.1f})")
    if stop_requested["flag"]:
        print("  Unterbrochen - mit --weiter laesst sich das Training fortsetzen.")
    print(f"  Gespeichert in {out_dir}")
    return out_dir / "modell.pt"
