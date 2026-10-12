"""Kommandozeile:  python -m llm <befehl> [optionen]

Befehle (in der ueblichen Reihenfolge):
    info          Pruefen, ob PyTorch die Grafikkarte findet
    vorbereiten   Text laden + Tokenizer trainieren + tokenisieren (alles in einem)
    trainieren    Vortraining des Sprachmodells
    generieren    Text fortsetzen lassen
    sft           Chat-Fine-Tuning (macht aus dem Textmodell einen Assistenten)
    chat          Im Terminal chatten
    web           Chat im Browser
    export        Nach GGUF exportieren fuer Ollama / LM Studio

Einzelschritte von 'vorbereiten':  daten, tokenizer, tokenisieren
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ZEICHEN_PRO_TOKEN = 4.3  # Schaetzwert fuer Deutsch bei 16k Vokabular


def _arbeit(args) -> Path:
    p = Path(args.arbeit)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_info(args) -> None:
    import platform

    import torch

    print(f"  Python      {platform.python_version()}  ({platform.system()} {platform.release()})")
    print(f"  PyTorch     {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  CUDA        ja, {torch.version.cuda}")
        print(f"  GPU         {p.name}, {p.total_memory / 2**30:.1f} GB VRAM, bf16: {torch.cuda.is_bf16_supported()}")
        x = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        import time
        t = time.time()
        for _ in range(20):
            x @ x
        torch.cuda.synchronize()
        tflops = 20 * 2 * 2048**3 / (time.time() - t) / 1e12
        print(f"  Leistung    ~{tflops:.1f} TFLOPS (fp32-Matrixmultiplikation)")
    else:
        print("  CUDA        NEIN - es wird nur die CPU genutzt.")
        print("              Loesung: setup_windows.ps1 ausfuehren (installiert PyTorch mit CUDA).")
    work = Path(args.arbeit)
    free = shutil.disk_usage(work if work.exists() else Path.cwd()).free / 2**30
    print(f"  Speicher    {free:.0f} GB frei im Arbeitsordner")
    if (work / "daten.json").exists():
        d = json.loads((work / "daten.json").read_text())
        print(f"  Daten       {d['train_tokens'] / 1e6:,.0f} Mio. Trainings-Tokens vorbereitet")
    else:
        print("  Daten       noch keine (python -m llm vorbereiten)")


def cmd_daten(args) -> None:
    from .config import PRESETS
    from .data import download

    tokens = args.tokens or PRESETS[args.preset].tokens_empfohlen
    max_chars = int(tokens * ZEICHEN_PRO_TOKEN)
    print(f"[1/3] Lade Text aus '{args.quelle}' (Ziel ~{tokens / 1e6:,.0f} Mio. Tokens = {max_chars / 1e9:.1f} GB Text)")
    stats = download(args.quelle, _arbeit(args), max_chars)
    print(f"  {stats['dokumente']:,} Dokumente, {stats['zeichen'] / 1e6:,.0f} Mio. Zeichen")


def cmd_tokenizer(args) -> None:
    from .tokenizer import Tokenizer, train_tokenizer

    work = _arbeit(args)
    sample = work / "tokenizer_stichprobe.txt"
    print(f"[2/3] Trainiere Tokenizer ({args.vocab} Tokens)...")
    path = train_tokenizer([sample], work / "tokenizer", vocab_size=args.vocab)
    tok = Tokenizer(path)
    demo = "Heidelberg liegt am Neckar und hat eine berühmte Altstadt."
    pieces = [tok.id_to_piece(i) for i in tok.encode(demo)]
    print(f"  Fertig: {path}")
    print(f"  Beispiel: {demo}\n  -> {len(pieces)} Tokens: {' | '.join(pieces)}")


def cmd_tokenisieren(args) -> None:
    from .data import tokenize

    work = _arbeit(args)
    print("[3/3] Tokenisiere alle Texte...")
    stats = tokenize(work, work / "tokenizer.model", workers=args.prozesse)
    print(f"  {stats['train_tokens'] / 1e6:,.1f} Mio. Trainings-Tokens, {stats['val_tokens'] / 1e6:,.2f} Mio. Validierung")


def cmd_vorbereiten(args) -> None:
    cmd_daten(args)
    cmd_tokenizer(args)
    cmd_tokenisieren(args)
    print("\nDaten fertig. Naechster Schritt:")
    print(f"  python -m llm trainieren --preset {args.preset}")


def _apply_overrides(tc, args):
    over = {}
    for key, attr in [("micro_batch", "micro_batch"), ("akkumulation", "grad_accum"), ("lr", "lr"),
                      ("schritte", "max_steps")]:
        val = getattr(args, key, None)
        if val is not None:
            over[attr] = val
    if getattr(args, "compile", False):
        over["compile"] = True
    return replace(tc, **over) if over else tc


def cmd_trainieren(args) -> None:
    from .config import PRESETS
    from .data import TokenData
    from .generate import pick_device
    from .train import train

    work = Path(args.arbeit)
    if not (work / "train.bin").exists():
        raise SystemExit("Keine Trainingsdaten gefunden - zuerst: python -m llm vorbereiten")
    meta = json.loads((work / "daten.json").read_text())
    preset = PRESETS[args.preset]
    mc = replace(preset.model, vocab_size=meta["vocab_size"])
    if args.kontext:
        mc = replace(mc, max_seq_len=args.kontext)
    tc = _apply_overrides(preset.train, args)
    out = Path(args.ausgabe) if args.ausgabe else work / f"modell_{args.preset}"
    data = TokenData(work, mc.max_seq_len, seed=tc.seed)
    print(f"Vortraining '{preset.name}': {preset.beschreibung}")
    print(f"  Daten:       {meta['train_tokens'] / 1e6:,.0f} Mio. Tokens")
    train(mc, tc, data.batch, out, work / "tokenizer.model", pick_device(args.geraet), hours=args.stunden,
          resume=args.weiter, tokens_available=meta["train_tokens"], run_name="vortraining")
    print("\nAusprobieren:")
    print(f"  python -m llm generieren --modell {out} --text \"Heidelberg ist\"")


def cmd_generieren(args) -> None:
    from .generate import complete, load_model, pick_device

    model, tok, _ = load_model(Path(args.modell), pick_device(args.geraet))
    for i in range(args.anzahl):
        print(f"\n--- {i + 1} ---\n{args.text}", end="")
        complete(model, tok, args.text, max_new_tokens=args.laenge, temperature=args.temperatur,
                 top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.wiederholung)
        print()


def cmd_sft(args) -> None:
    import torch

    from .config import TrainConfig
    from .generate import find_checkpoint, find_tokenizer, pick_device
    from .model import GPT  # noqa: F401  (Import prueft Abhaengigkeiten frueh)
    from .sft import STANDARD_DATENSATZ, ChatData, prepare
    from .tokenizer import Tokenizer
    from .config import ModelConfig
    from .train import train

    base = find_checkpoint(Path(args.basis))
    ck = torch.load(base, map_location="cpu", weights_only=False)
    mc = ModelConfig.from_dict(ck["model_cfg"])
    tok_path = find_tokenizer(base)
    tok = Tokenizer(tok_path)
    out = Path(args.ausgabe) if args.ausgabe else base.parent.parent / ("chat_" + base.parent.name.replace("modell_", ""))
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "sft_daten.pt"
    if not data_path.exists() or args.neu_tokenisieren:
        sources = args.datensatz or [STANDARD_DATENSATZ]
        print("Bereite Chat-Daten vor...")
        prepare(sources, tok, mc.max_seq_len, data_path, identity_repeat=args.identitaet,
                max_examples=args.max_beispiele, name=args.name, ersteller=args.ersteller)
    data = ChatData(data_path, pad_id=tok.eos_id)
    tokens_per_step = args.micro_batch * args.akkumulation * mc.max_seq_len
    steps = args.schritte or max(200, int(args.epochen * data.train_tokens / (tokens_per_step * 0.5)))
    tc = TrainConfig(micro_batch=args.micro_batch, grad_accum=args.akkumulation, lr=args.lr,
                     warmup_steps=min(100, steps // 10), max_steps=steps, eval_every=max(50, steps // 20),
                     save_every=max(100, steps // 10), sample_every=10**9, weight_decay=0.0, dropout=0.0)
    print(f"Chat-Fine-Tuning: {len(data.train):,} Beispiele, {steps:,} Schritte")
    train(mc, tc, data.batch, out, tok_path, pick_device(args.geraet), resume=args.weiter,
          init_weights=None if args.weiter else ck["model"], run_name="sft")
    print("\nJetzt chatten:")
    print(f"  python -m llm chat --modell {out}")


def cmd_chat(args) -> None:
    from .generate import encode_chat, generate, load_model, pick_device
    from .tokenizer import StreamDecoder

    model, tok, _ = load_model(Path(args.modell), pick_device(args.geraet))
    print(f"Chat mit {args.name} ({model.num_params() / 1e6:.0f} Mio. Parameter). "
          "Befehle: /neu = Verlauf loeschen, /exit = beenden\n")
    history: list[dict] = []
    if args.system:
        history.append({"role": "system", "content": args.system})
    while True:
        try:
            user = input("Du: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/bye", "/quit"):
            break
        if user == "/neu":
            history = history[:1] if args.system else []
            print("(Verlauf geloescht)\n")
            continue
        history.append({"role": "user", "content": user})
        print(f"{args.name}: ", end="", flush=True)
        dec = StreamDecoder(tok, lstrip=True)
        ids = generate(model, encode_chat(tok, history), max_new_tokens=args.laenge, temperature=args.temperatur,
                       top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.wiederholung,
                       stop_ids={tok.eos_id, tok.user_id},
                       on_token=lambda t: print(dec.push(t), end="", flush=True))
        print("\n")
        history.append({"role": "assistant", "content": tok.decode(ids).strip()})


def cmd_web(args) -> None:
    from .web import serve

    serve(Path(args.modell), port=args.port, system=args.system, name=args.name)


def cmd_export(args) -> None:
    from .export import export_gguf, export_hf, ollama_create, write_modelfile
    from .generate import find_checkpoint

    import torch

    model_path = Path(args.modell)
    ck = torch.load(find_checkpoint(model_path), map_location="cpu", weights_only=False)
    out = Path(args.ausgabe) if args.ausgabe else Path("export") / args.name
    hf_dir = out / "hf"
    print(f"[1/3] Hugging-Face-Format -> {hf_dir}")
    export_hf(model_path, hf_dir, args.name)
    gguf = out / f"{args.name}-{args.gguf_typ}.gguf"
    print(f"[2/3] GGUF ({args.gguf_typ}) -> {gguf}")
    export_gguf(hf_dir, gguf, Path(args.llama_cpp) if args.llama_cpp else None, args.gguf_typ)
    mf = write_modelfile(gguf, out, args.system, ck["model_cfg"]["max_seq_len"])
    print(f"[3/3] Ollama-Modelfile -> {mf}")
    if args.ollama and ollama_create(args.name, mf):
        print(f"\nFertig! Starten mit:  ollama run {args.name}")
    else:
        print("\nFertig! In Ollama einbinden:")
        print(f"  cd {out}")
        print(f"  ollama create {args.name} -f Modelfile")
        print(f"  ollama run {args.name}")
        print(f"\nOder in LM Studio: die Datei {gguf.name} in den Modellordner von LM Studio kopieren")
        print("  (LM Studio -> My Models -> Ordner oeffnen, dort Unterordner z.B. eigen/eigen-lm/ anlegen).")


def build_parser() -> argparse.ArgumentParser:
    from .config import PRESETS
    from .data import QUELLEN

    p = argparse.ArgumentParser(prog="python -m llm", description="Eigenes deutsches Sprachmodell bauen.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--arbeit", default="arbeit", help="Arbeitsordner fuer Daten und Modelle (Standard: arbeit)")
    sub = p.add_subparsers(dest="befehl", required=True)

    def gen_opts(sp):
        sp.add_argument("--temperatur", type=float, default=0.7)
        sp.add_argument("--top-k", type=int, default=50)
        sp.add_argument("--top-p", type=float, default=0.9)
        sp.add_argument("--wiederholung", type=float, default=1.1, help="Wiederholungsstrafe")
        sp.add_argument("--laenge", type=int, default=300, help="max. neue Tokens")
        sp.add_argument("--geraet", default=None, help="cuda / cpu (Standard: automatisch)")

    sp = sub.add_parser("info", help="Systemcheck")
    sp.set_defaults(fn=cmd_info)

    for name, fn, hilfe in [("vorbereiten", cmd_vorbereiten, "Daten + Tokenizer + Tokenisieren"),
                            ("daten", cmd_daten, "Nur Text herunterladen"),
                            ("tokenizer", cmd_tokenizer, "Nur Tokenizer trainieren"),
                            ("tokenisieren", cmd_tokenisieren, "Nur tokenisieren")]:
        sp = sub.add_parser(name, help=hilfe)
        sp.add_argument("--quelle", default="fineweb2",
                        help=f"{' / '.join(QUELLEN)} oder Pfad zu einem Ordner mit .txt-Dateien")
        sp.add_argument("--preset", default="klein", choices=list(PRESETS))
        sp.add_argument("--tokens", type=int, default=None, help="Wie viele Tokens laden (Standard: je nach Preset)")
        sp.add_argument("--vocab", type=int, default=16000)
        sp.add_argument("--prozesse", type=int, default=None)
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("trainieren", help="Vortraining")
    sp.add_argument("--preset", default="klein", choices=list(PRESETS))
    sp.add_argument("--stunden", type=float, default=None, help="Zeitbudget, z.B. 6")
    sp.add_argument("--weiter", action="store_true", help="Vom letzten Checkpoint fortsetzen")
    sp.add_argument("--ausgabe", default=None)
    sp.add_argument("--micro-batch", dest="micro_batch", type=int)
    sp.add_argument("--akkumulation", type=int)
    sp.add_argument("--lr", type=float)
    sp.add_argument("--schritte", type=int)
    sp.add_argument("--kontext", type=int, help="Kontextlaenge ueberschreiben")
    sp.add_argument("--compile", action="store_true", help="torch.compile (unter Windows meist nicht verfuegbar)")
    sp.add_argument("--geraet", default=None)
    sp.set_defaults(fn=cmd_trainieren)

    sp = sub.add_parser("generieren", help="Text fortsetzen")
    sp.add_argument("--modell", required=True)
    sp.add_argument("--text", default="Heidelberg ist")
    sp.add_argument("--anzahl", type=int, default=3)
    gen_opts(sp)
    sp.set_defaults(fn=cmd_generieren, laenge=150, temperatur=0.8)

    sp = sub.add_parser("sft", help="Chat-Fine-Tuning")
    sp.add_argument("--basis", required=True, help="Ordner des vortrainierten Modells")
    sp.add_argument("--datensatz", action="append", help="HF-Datensatz oder JSONL-Datei (mehrfach moeglich)")
    sp.add_argument("--ausgabe", default=None)
    sp.add_argument("--name", default="Eigen-LM", help="So nennt sich das Modell selbst")
    sp.add_argument("--ersteller", default="Leo", help="Wer das Modell gebaut hat")
    sp.add_argument("--identitaet", type=int, default=20, help="Wie oft die Identitaets-Beispiele vorkommen")
    sp.add_argument("--max-beispiele", dest="max_beispiele", type=int, default=None)
    sp.add_argument("--epochen", type=float, default=2.0)
    sp.add_argument("--schritte", type=int, default=None)
    sp.add_argument("--lr", type=float, default=2e-4)
    sp.add_argument("--micro-batch", dest="micro_batch", type=int, default=16)
    sp.add_argument("--akkumulation", type=int, default=2)
    sp.add_argument("--weiter", action="store_true")
    sp.add_argument("--neu-tokenisieren", dest="neu_tokenisieren", action="store_true")
    sp.add_argument("--geraet", default=None)
    sp.set_defaults(fn=cmd_sft)

    sp = sub.add_parser("chat", help="Chat im Terminal")
    sp.add_argument("--modell", required=True)
    sp.add_argument("--name", default="Eigen-LM")
    sp.add_argument("--system", default=None, help="Systemanweisung")
    gen_opts(sp)
    sp.set_defaults(fn=cmd_chat)

    sp = sub.add_parser("web", help="Chat im Browser")
    sp.add_argument("--modell", required=True)
    sp.add_argument("--port", type=int, default=8800)
    sp.add_argument("--name", default="Eigen-LM")
    sp.add_argument("--system", default=None)
    sp.set_defaults(fn=cmd_web)

    sp = sub.add_parser("export", help="Export nach GGUF / Ollama")
    sp.add_argument("--modell", required=True)
    sp.add_argument("--name", default="eigen-lm", help="Name in Ollama (klein, ohne Leerzeichen)")
    sp.add_argument("--ausgabe", default=None)
    sp.add_argument("--gguf-typ", dest="gguf_typ", default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    sp.add_argument("--llama-cpp", dest="llama_cpp", default=None, help="Pfad zu einem llama.cpp-Ordner")
    sp.add_argument("--system", default=None, help="Standard-Systemanweisung in der Modelfile")
    sp.add_argument("--ollama", action="store_true", help="Direkt 'ollama create' ausfuehren")
    sp.set_defaults(fn=cmd_export)
    return p


def main(argv=None) -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
