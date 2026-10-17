"""Kommandozeile:  python -m insights <befehl>

    demo       Beispieldaten erzeugen und den kompletten Bericht bauen
    bericht    Bericht aus eigenen Daten (CSV/Excel oder mapping.json)
    pruefen    Nur Datenqualitaet pruefen
    prognose   Prognose + Szenarien fuer eine Marke im Terminal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _szenarien(args):
    from .prognose import Szenario

    erg = []
    for text in args.szenario or []:
        # Format: "Name:preis=5,distribution=-2,werbung=20"
        name, _, werte = text.partition(":")
        kw = {"preis": 0.0, "distribution": 0.0, "werbung": 0.0}
        for teil in filter(None, werte.split(",")):
            k, _, v = teil.partition("=")
            kw[k.strip()] = float(v.replace(",", "."))
        erg.append(Szenario(name.strip(), kw["preis"], kw["distribution"], kw["werbung"]))
    return erg or None


def _ausgabe(erg) -> None:
    print("\nWichtigste Erkenntnisse:")
    for e in erg.erkenntnisse[:8]:
        print(f"  - [{e.kategorie}] {e.text}")
    print(f"\nExcel:      {erg.excel}")
    print(f"PowerPoint: {erg.folien}")


def cmd_demo(args) -> None:
    from .beispieldaten import schreibe_beispiel
    from .bericht import erstelle

    ordner = Path(args.ordner)
    print("Erzeuge Beispieldaten (erfundene Marken) ...")
    pfade = schreibe_beispiel(ordner / "daten")
    print(f"  {pfade['panel'].name}, {pfade['zusatz'].name}, {pfade['mapping'].name}")
    print("Erstelle Bericht ...")
    erg = erstelle(pfade["mapping"], ordner / "berichte", ende=args.bis, ollama=args.ollama,
                   hinweis="Beispieldaten – alle Hersteller, Marken und Zahlen sind frei erfunden",
                   szenarien=_szenarien(args))
    _ausgabe(erg)


def cmd_bericht(args) -> None:
    from .bericht import erstelle

    erg = erstelle(Path(args.daten), Path(args.ausgabe), ende=args.bis, fenster=args.fenster,
                   szenarien=_szenarien(args), titel=args.titel, ollama=args.ollama)
    _ausgabe(erg)


def cmd_pruefen(args) -> None:
    from .laden import lade, pruefe

    df, meta = lade(Path(args.daten))
    print(f"{len(df):,} Zeilen aus {', '.join(q['datei'] for q in meta['quellen'])}")
    for p in pruefe(df):
        print(f"  [{p.stufe}] {p.thema}: {p.text}")


def cmd_prognose(args) -> None:
    from .kommentar import pp, zahl
    from .laden import lade
    from .prognose import prognostiziere

    df, _ = lade(Path(args.daten))
    p = prognostiziere(df, args.kategorie, args.marke, horizont=args.monate, szenarien=_szenarien(args),
                       ebene="hersteller" if args.hersteller else "marke")
    print(f"Prognose {p.wer} ({p.kategorie}), heute {zahl(p.historie['anteil'].iloc[-1])} %")
    for _, z in p.tabelle.iterrows():
        print(f"  {z['monat']}: {zahl(z['prognose'], 2)} %  [{zahl(z['unten'], 2)} – {zahl(z['oben'], 2)}]")
    if len(p.rueckblick):
        print("Rueckblicktest (mittlerer Fehler in pp):",
              ", ".join(f"{int(r.horizont)} M: {zahl(r.mae_kombi, 2)}" for r in p.rueckblick.itertuples()))
    for name, t in p.szenarien.items():
        print(f"Szenario {name}: {zahl(t['prognose'].iloc[-1], 2)} % "
              f"({pp(t['prognose'].iloc[-1] - p.tabelle['prognose'].iloc[-1])})")
    for name, w in p.sensitivitaet.items():
        print(f"Hebel {name}: {pp(w)}")


def main(argv=None) -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    p = argparse.ArgumentParser(prog="python -m insights", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="befehl", required=True)

    def gemeinsam(sp):
        sp.add_argument("--bis", default=None, help="Letzter Monat des Berichts, z.B. 2026-06 (Standard: neuester)")
        sp.add_argument("--szenario", action="append",
                        help='z.B. "Preiserhoehung:preis=5" oder "Listung:distribution=3,werbung=20"')
        sp.add_argument("--ollama", action="store_true", help="Texte mit lokalem Ollama-Modell glaetten")

    sp = sub.add_parser("demo", help="Beispieldaten + kompletter Bericht")
    sp.add_argument("--ordner", default="demo")
    gemeinsam(sp)
    sp.set_defaults(fn=cmd_demo)

    sp = sub.add_parser("bericht", help="Bericht aus eigenen Daten")
    sp.add_argument("--daten", required=True, help="CSV/Excel im internen Format oder mapping.json")
    sp.add_argument("--ausgabe", default="berichte")
    sp.add_argument("--fenster", type=int, default=3, help="Monate pro Zeitraum (1 = Monat, 3 = rollierend)")
    sp.add_argument("--titel", default="Marktbericht Nikotinkategorien")
    gemeinsam(sp)
    sp.set_defaults(fn=cmd_bericht)

    sp = sub.add_parser("pruefen", help="Datenqualitaet pruefen")
    sp.add_argument("--daten", required=True)
    sp.set_defaults(fn=cmd_pruefen)

    sp = sub.add_parser("prognose", help="Prognose und Szenarien im Terminal")
    sp.add_argument("--daten", required=True)
    sp.add_argument("--kategorie", required=True)
    sp.add_argument("--marke", required=True, help="Marke (oder Hersteller mit --hersteller)")
    sp.add_argument("--hersteller", action="store_true")
    sp.add_argument("--monate", type=int, default=12)
    sp.add_argument("--szenario", action="append")
    sp.set_defaults(fn=cmd_prognose)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
