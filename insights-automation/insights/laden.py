"""Daten einlesen, auf das interne Format bringen und pruefen.

Zwei Wege:
1. Eine Datei, die schon das interne Format hat (CSV oder Excel, Spalten wie in
   config.PFLICHT_SPALTEN).
2. Mehrere Quellen mit eigenen Spaltennamen + mapping.json, die beschreibt, wie
   Spalten heissen und wie die Quellen verknuepft werden (siehe beispieldaten.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OPTIONALE_SPALTEN, PFLICHT_SPALTEN


@dataclass
class Pruefung:
    stufe: str      # "OK", "Hinweis", "Warnung", "Fehler"
    thema: str
    text: str


def _lies_datei(pfad: Path, blatt: str | None = None, trennzeichen: str | None = None,
                dezimal: str | None = None) -> pd.DataFrame:
    if pfad.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(pfad, sheet_name=blatt or 0)
    if trennzeichen is None:  # automatisch erkennen (; ist in deutschen Exporten ueblich)
        kopf = pfad.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
        trennzeichen = ";" if kopf.count(";") > kopf.count(",") else ","
    if dezimal is None:
        dezimal = "," if trennzeichen == ";" else "."
    return pd.read_csv(pfad, sep=trennzeichen, decimal=dezimal, encoding="utf-8-sig")


def _normalisiere(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "monat" in df.columns:
        df["monat"] = pd.to_datetime(df["monat"].astype(str), errors="coerce").dt.to_period("M").astype(str)
    for spalte in ("region", "kanal", "kategorie", "hersteller", "marke"):
        if spalte in df.columns:
            df[spalte] = df[spalte].astype(str).str.strip()
    for spalte in ("absatz", "umsatz", "distribution_gew", "werbedruck"):
        if spalte in df.columns:
            df[spalte] = pd.to_numeric(df[spalte], errors="coerce")
    return df


def lade(pfad: Path) -> tuple[pd.DataFrame, dict]:
    """Laedt eine Datei im internen Format oder eine mapping.json mit mehreren Quellen."""
    pfad = Path(pfad)
    meta: dict = {"quellen": []}
    if pfad.suffix.lower() == ".json":
        mapping = json.loads(pfad.read_text(encoding="utf-8"))
        meta["eigener_hersteller"] = mapping.get("eigener_hersteller")
        df = None
        for quelle in mapping["quellen"]:
            datei = (pfad.parent / quelle["datei"]).resolve()
            teil = _lies_datei(datei, quelle.get("blatt"), quelle.get("trennzeichen"), quelle.get("dezimal"))
            fehlend = [s for s in quelle["spalten"] if s not in teil.columns]
            if fehlend:
                raise SystemExit(f"In {datei.name} fehlen Spalten: {', '.join(fehlend)}")
            teil = _normalisiere(teil[list(quelle["spalten"])].rename(columns=quelle["spalten"]))
            meta["quellen"].append({"datei": datei.name, "zeilen": len(teil)})
            if df is None:
                df = teil
            else:
                schluessel = quelle.get("verknuepfen_ueber") or [c for c in teil.columns if c in df.columns]
                df = df.merge(teil, on=schluessel, how="left", validate="many_to_one")
    else:
        df = _normalisiere(_lies_datei(pfad))
        meta["quellen"].append({"datei": pfad.name, "zeilen": len(df)})
    fehlend = [s for s in PFLICHT_SPALTEN if s not in df.columns]
    if fehlend:
        raise SystemExit(f"Pflichtspalten fehlen: {', '.join(fehlend)}. Mapping pruefen (siehe README).")
    for spalte in OPTIONALE_SPALTEN:
        if spalte not in df.columns:
            df[spalte] = np.nan
    return df[PFLICHT_SPALTEN + OPTIONALE_SPALTEN], meta


def pruefe(df: pd.DataFrame) -> list[Pruefung]:
    """Automatische Datenqualitaets-Checks, bevor irgendetwas berichtet wird."""
    erg: list[Pruefung] = []
    schluessel = ["monat", "region", "kanal", "kategorie", "marke"]

    if df["monat"].isna().any():
        erg.append(Pruefung("Fehler", "Perioden", f"{df['monat'].isna().sum()} Zeilen ohne lesbaren Monat."))
    dup = df.duplicated(schluessel).sum()
    if dup:
        erg.append(Pruefung("Fehler", "Dubletten", f"{dup} doppelte Zeilen (gleicher Monat/Region/Kanal/Marke)."))
    neg = (df[["absatz", "umsatz"]] < 0).any(axis=1).sum()
    if neg:
        erg.append(Pruefung("Fehler", "Negative Werte", f"{neg} Zeilen mit negativem Absatz oder Umsatz."))
    leer = df[["absatz", "umsatz"]].isna().any(axis=1).sum()
    if leer:
        erg.append(Pruefung("Warnung", "Fehlende Werte", f"{leer} Zeilen ohne Absatz oder Umsatz."))

    alle = sorted(df["monat"].dropna().unique())
    if alle:
        erwartet = [str(p) for p in pd.period_range(alle[0], alle[-1], freq="M")]
        luecken = sorted(set(erwartet) - set(alle))
        if luecken:
            erg.append(Pruefung("Fehler", "Perioden", f"Monate fehlen komplett: {', '.join(luecken[:6])}"))
        erg.append(Pruefung("OK", "Zeitraum", f"{alle[0]} bis {alle[-1]} ({len(alle)} Monate)."))

    # Marken, die mittendrin verschwinden (Neueinfuehrungen sind normal, Luecken nicht)
    n_monate = df.groupby(["marke"])["monat"].nunique()
    for marke, n in n_monate.items():
        erster = df.loc[df["marke"] == marke, "monat"].min()
        soll = len([m for m in alle if m >= erster])
        if n < soll:
            erg.append(Pruefung("Warnung", "Luecken", f"{marke}: {soll - n} Monate ohne Daten seit {erster}."))
        elif erster > alle[0]:
            erg.append(Pruefung("Hinweis", "Neue Marke", f"{marke} erscheint erstmals {erster}."))

    # Preisplausibilitaet: Durchschnittspreis je Marke gegen ihren eigenen Verlauf
    d = df[df["absatz"] > 0].assign(preis=lambda x: x["umsatz"] / x["absatz"])
    for (kat, marke), g in d.groupby(["kategorie", "marke"]):
        p = g.groupby("monat")["preis"].median()
        rel = (p / p.rolling(6, min_periods=3).median().shift(1) - 1).dropna()
        sprung = rel[rel.abs() > 0.15]
        for monat, wert in sprung.items():
            erg.append(Pruefung("Warnung", "Preissprung", f"{marke}: Durchschnittspreis {wert * 100:+.0f} % in {monat} "
                                                           f"- Datenfehler oder echte Preisaenderung?"))

    if df["distribution_gew"].notna().any():
        aus = ((df["distribution_gew"] < 0) | (df["distribution_gew"] > 100)).sum()
        if aus:
            erg.append(Pruefung("Fehler", "Distribution", f"{aus} Werte ausserhalb 0-100 %."))
    else:
        erg.append(Pruefung("Hinweis", "Distribution", "Keine Distributionsdaten - Treiberanalyse eingeschraenkt."))

    if not any(p.stufe in {"Fehler", "Warnung"} for p in erg):
        erg.append(Pruefung("OK", "Gesamt", "Keine Auffaelligkeiten gefunden."))
    return erg
