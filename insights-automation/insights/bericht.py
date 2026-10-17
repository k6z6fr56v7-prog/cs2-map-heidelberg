"""Der komplette Monatsbericht in einem Aufruf: laden -> pruefen -> analysieren ->
prognostizieren -> kommentieren -> Excel + PowerPoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config, excel, folien
from .kennzahlen import anteile_zeitreihe, beitragsanalyse, eigene_marken, markt_ueberblick, ranking, segment_matrix, zeitraum
from .kommentar import erkenntnisse, glaetten_mit_ollama
from .laden import lade, pruefe
from .prognose import Szenario, prognostiziere


@dataclass
class Ergebnis:
    excel: Path
    folien: Path
    erkenntnisse: list
    pruefungen: list


def erstelle(daten: Path, ausgabe: Path, ende: str | None = None, fenster: int = 3,
             szenarien: list[Szenario] | None = None, titel: str = "Marktbericht Nikotinkategorien",
             ollama: bool = False, hinweis: str = "") -> Ergebnis:
    df, meta = lade(daten)
    eigener = meta.get("eigener_hersteller") or config.EIGENER_HERSTELLER
    config.EIGENER_HERSTELLER = eigener
    pruefungen = pruefe(df)
    if any(p.stufe == "Fehler" for p in pruefungen):
        fehler = "\n".join(f"  - {p.text}" for p in pruefungen if p.stufe == "Fehler")
        raise SystemExit(f"Datenfehler - Bericht wird nicht erstellt:\n{fehler}")
    if ende:
        df = df[df["monat"] <= ende]
    z = zeitraum(df["monat"].unique(), ende, fenster)

    kategorien = sorted(df["kategorie"].unique(), key=lambda k: -df[df["kategorie"] == k]["umsatz"].sum())
    ueberblick = markt_ueberblick(df, z, eigener)
    ueberblick = ueberblick.set_index("kategorie").loc[[k for k in kategorien if k in set(ueberblick["kategorie"])]].reset_index()
    rankings = {k: ranking(df, k, z) for k in kategorien}
    eigene = {k: eigene_marken(df, eigener, k) for k in kategorien}
    zeitreihen = {}
    for k in kategorien:
        zr = anteile_zeitreihe(df[df["kategorie"] == k])
        piv = zr.pivot(index="monat", columns="marke", values="anteil")
        reihenfolge = eigene[k] + [m for m in rankings[k]["marke"] if m not in eigene[k]]
        zeitreihen[k] = piv[[m for m in reihenfolge if m in piv.columns]]

    # Prognose fuer die groesste eigene Marke jeder Kategorie
    szenarien = szenarien if szenarien is not None else [Szenario("Preis +5 %", preis_prozent=5),
                                                         Szenario("Distribution +3 pp", distribution_pp=3)]
    prognosen = {}
    for k in kategorien:
        if eigene[k]:
            prognosen[(k, eigene[k][0])] = prognostiziere(df, k, eigene[k][0], szenarien=szenarien)

    erk = erkenntnisse(df, z, eigener, prognosen)
    if ollama:
        texte = glaetten_mit_ollama([e.text for e in erk])
        for e, t in zip(erk, texte):
            e.text = t

    # Segmentfolie: die eigene Marke mit dem groessten Verlust (oder Gewinn)
    segment = None
    kandidaten = []
    for k in kategorien:
        r = rankings[k]
        for m in eigene[k]:
            d = float(r.loc[r["marke"] == m, "delta_menge_vj"].iloc[0]) if (r["marke"] == m).any() else 0
            kandidaten.append((abs(d) * (2 if d < 0 else 1), k, m))  # Verluste zuerst erklaeren
    segmente_excel = {}
    for k in kategorien:
        if eigene[k]:
            segmente_excel[k] = (eigene[k][0], segment_matrix(df, k, z, eigene[k][0]))
    if kandidaten:
        _, k, m = max(kandidaten)
        segment = (k, m, segment_matrix(df, k, z, m), beitragsanalyse(df, k, z, m))

    ausgabe.mkdir(parents=True, exist_ok=True)
    stamm = f"Marktbericht_{z.monate[-1]}"
    xlsx = excel.schreibe(ausgabe / f"{stamm}.xlsx", titel, z.name, ueberblick, rankings, segmente_excel,
                          zeitreihen, prognosen, erk, pruefungen)
    pptx = folien.schreibe(ausgabe / f"{stamm}.pptx", titel, z.name, ueberblick, rankings, zeitreihen, eigene,
                           erk, segment, prognosen, pruefungen, meta["quellen"], hinweis)
    return Ergebnis(xlsx, pptx, erk, pruefungen)
