"""Automatische Kommentare: findet die wichtigsten Bewegungen und erklaert sie.

Regelbasiert und nachvollziehbar - jede Aussage beruht auf einer Zahl aus den
Kennzahlen. Optional kann ein lokales Sprachmodell (Ollama) die Stichpunkte
anschliessend sprachlich glaetten; die Fakten bleiben dieselben.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .kennzahlen import Zeitraum, beitragsanalyse, eigene_marken, markt_ueberblick, ranking


@dataclass
class Erkenntnis:
    kategorie: str
    typ: str          # Ueberblick, Marke, Treiber, Wettbewerb, Neu, Auffaellig, Prognose
    gewicht: float    # fuer die Reihenfolge (groesser = wichtiger)
    text: str
    eigen: bool = False


# ---------------------------------------------------------------- Formatierung


def zahl(x: float, nachkomma: int = 1) -> str:
    return f"{x:,.{nachkomma}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pp(x: float) -> str:
    return f"{'+' if x >= 0 else '−'}{zahl(abs(x))} pp"


def prozent(x: float, vorzeichen: bool = True) -> str:
    s = zahl(abs(x)) + " %"
    return (("+" if x >= 0 else "−") + s) if vorzeichen else s


def menge(x: float) -> str:
    if abs(x) >= 1e6:
        return zahl(x / 1e6) + " Mio."
    if abs(x) >= 1e3:
        return zahl(x / 1e3) + " Tsd."
    return zahl(x, 0)


def segment_name(region: str, kanal: str) -> str:
    return f"{kanal} {region.replace('Sued', 'Süd')}"


# ---------------------------------------------------------------- Segment-Treiber


def _segment_treiber(df: pd.DataFrame, kategorie: str, z: Zeitraum, marke: str, region: str, kanal: str) -> list[str]:
    seg = df[(df["kategorie"] == kategorie) & (df["region"] == region) & (df["kanal"] == kanal)]
    r = ranking(seg, kategorie, z).set_index("marke")
    if marke not in r.index:
        return []
    e = r.loc[marke]
    gruende = []
    dd = e["delta_distribution_vj"]
    if pd.notna(dd) and abs(dd) >= config.SCHWELLE_DISTRIBUTION_PP:
        gruende.append(f"gewichtete Distribution {pp(dd).replace(' pp', ' Pkt.')} auf {zahl(e['distribution'], 0)} %")
    dp = e["delta_preisindex_vj"]
    if pd.notna(dp) and abs(dp) >= config.SCHWELLE_PREIS_INDEX:
        gruende.append(f"Preisindex {'gestiegen' if dp > 0 else 'gesunken'} ({zahl(e['preisindex'], 0)}, {dp:+.0f} Pkt.)")
    # Wettbewerber mit groesstem Gewinn im Segment
    andere = r.drop(index=marke)
    if len(andere):
        top = andere["delta_menge_vj"].idxmax()
        if andere.loc[top, "delta_menge_vj"] >= config.SCHWELLE_ANTEIL_PP:
            t = andere.loc[top]
            zusatz = ""
            if pd.notna(t["delta_preisindex_vj"]) and t["delta_preisindex_vj"] <= -config.SCHWELLE_PREIS_INDEX:
                zusatz = f" nach Preissenkung ({t['delta_preisindex_vj']:+.0f} Pkt. Preisindex)"
            gruende.append(f"{top} gewinnt dort {pp(t['delta_menge_vj'])}{zusatz}")
    return gruende


def _nationale_treiber(df: pd.DataFrame, kategorie: str, z: Zeitraum, marke: str, r: pd.DataFrame,
                       richtung: float) -> list[str]:
    """Erklaerung, wenn sich die Bewegung nicht auf ein Segment konzentriert.
    Genannt werden nur Treiber, deren Richtung zur Bewegung passt (z.B. mehr Werbung
    bei Zugewinn, hoeherer Preis bei Verlust)."""
    gruende = []
    plus = richtung > 0
    e = r.set_index("marke").loc[marke]
    d = df[(df["kategorie"] == kategorie) & (df["marke"] == marke)]
    w_akt = d[d["monat"].isin(z.monate)].groupby("monat")["werbedruck"].mean().mean()
    w_vj = d[d["monat"].isin(z.vj)].groupby("monat")["werbedruck"].mean().mean()
    if pd.notna(w_akt) and pd.notna(w_vj) and w_vj > 0 and abs(w_akt / w_vj - 1) >= 0.25 \
            and (w_akt > w_vj) == plus:
        gruende.append(f"Werbedruck {prozent((w_akt / w_vj - 1) * 100)} vs. VJ")
    if pd.notna(e["delta_distribution_vj"]) and abs(e["delta_distribution_vj"]) >= config.SCHWELLE_DISTRIBUTION_PP / 2 \
            and (e["delta_distribution_vj"] > 0) == plus:
        gruende.append(f"Distribution {e['delta_distribution_vj']:+.1f} Pkt.".replace(".", ",", 1))
    if pd.notna(e["delta_preisindex_vj"]) and abs(e["delta_preisindex_vj"]) >= config.SCHWELLE_PREIS_INDEX \
            and (e["delta_preisindex_vj"] < 0) == plus:
        gruende.append(f"Preisindex {e['delta_preisindex_vj']:+.0f} Pkt.")
    if plus:
        return gruende
    neu = r[(r["anteil_menge_vj"] == 0) & (r["anteil_menge"] >= 2) & (r["marke"] != marke)]
    for _, n in neu.iterrows():
        gruende.append(f"Neueinführung {n['marke']} holt {zahl(n['anteil_menge'])} %")
    return gruende


# ---------------------------------------------------------------- Hauptfunktion


def erkenntnisse(df: pd.DataFrame, z: Zeitraum, eigener: str = config.EIGENER_HERSTELLER,
                 prognosen: dict | None = None) -> list[Erkenntnis]:
    erg: list[Erkenntnis] = []
    ueb = markt_ueberblick(df, z, eigener).set_index("kategorie")
    schwelle = config.SCHWELLE_ANTEIL_PP

    for kat in sorted(df["kategorie"].unique()):
        if kat in ueb.index:
            u = ueb.loc[kat]
            erg.append(Erkenntnis(kat, "Ueberblick", 100 + abs(u["delta_menge_vj"]),
                                  f"{kat}: Markt {prozent(u['markt_veraenderung_vj'])} vs. VJ. {eigener} hält "
                                  f"{zahl(u['anteil_menge'])} % Mengenanteil ({pp(u['delta_menge_vj'])} vs. VJ, "
                                  f"{pp(u['delta_menge_vp'])} vs. Vorperiode), Rang {int(u['rang'])}.", eigen=True))
        r = ranking(df, kat, z)

        # Eigene Marken mit nennenswerter Bewegung + Erklaerung ueber Segmente
        for marke in eigene_marken(df, eigener, kat):
            zeile = r[r["marke"] == marke]
            if zeile.empty:
                continue
            d = float(zeile["delta_menge_vj"].iloc[0])
            if abs(d) < schwelle:
                continue
            b = beitragsanalyse(df, kat, z, marke)
            haupt = b.iloc[0] if d < 0 else b.iloc[-1]
            anteil_am_effekt = haupt["beitrag"] / d * 100 if d else 0
            text = (f"{marke} {'verliert' if d < 0 else 'gewinnt'} {pp(abs(d)).lstrip('+')} vs. VJ "
                    f"(jetzt {zahl(float(zeile['anteil_menge'].iloc[0]))} %).")
            if anteil_am_effekt >= 40:
                text += (f" Rund {zahl(anteil_am_effekt, 0)} % davon kommen aus {segment_name(haupt['region'], haupt['kanal'])} "
                         f"({pp(haupt['delta_vj'])} Anteil im Segment).")
                gruende = _segment_treiber(df, kat, z, marke, haupt["region"], haupt["kanal"])
                if gruende:
                    text += " Treiber dort: " + "; ".join(gruende) + "."
            else:
                text += " Die Bewegung ist breit über die Segmente verteilt."
                if anteil_am_effekt >= 25:
                    text += (f" Größter Einzelbeitrag: {segment_name(haupt['region'], haupt['kanal'])} "
                             f"({pp(haupt['beitrag'])} national, {pp(haupt['delta_vj'])} im Segment")
                    seg = _segment_treiber(df, kat, z, marke, haupt["region"], haupt["kanal"])
                    text += (f"; {seg[0]}" if seg else "") + ")."
                gruende = _nationale_treiber(df, kat, z, marke, r, d)
                if gruende:
                    text += " Mögliche Treiber: " + "; ".join(gruende) + "."
            erg.append(Erkenntnis(kat, "Marke", 50 + abs(d) * 20, text, eigen=True))

        # Wettbewerb: groesste Bewegungen fremder Marken
        fremd = r[r["hersteller"] != eigener]
        for _, w in fremd.iterrows():
            d = w["delta_menge_vj"]
            if w["anteil_menge_vj"] == 0 and w["anteil_menge"] > 0:
                erg.append(Erkenntnis(kat, "Neu", 45 + w["anteil_menge"],
                                      f"Neue Marke {w['marke']} ({w['hersteller']}) erreicht bereits "
                                      f"{zahl(w['anteil_menge'])} % Mengenanteil."))
            elif abs(d) >= 2 * schwelle:
                zusatz = ""
                if pd.notna(w["delta_preisindex_vj"]) and w["delta_preisindex_vj"] <= -config.SCHWELLE_PREIS_INDEX:
                    zusatz = f" bei deutlich niedrigerem Preisindex ({w['delta_preisindex_vj']:+.0f} Pkt.)"
                erg.append(Erkenntnis(kat, "Wettbewerb", 30 + abs(d) * 10,
                                      f"Wettbewerb: {w['marke']} ({w['hersteller']}) {pp(d)} vs. VJ{zusatz}."))

        # Auffaellige Monatsbewegungen der eigenen Marken
        for marke in eigene_marken(df, eigener, kat):
            s = df[(df["kategorie"] == kat)]
            reihe = (s[s["marke"] == marke].groupby("monat")["absatz"].sum() / s.groupby("monat")["absatz"].sum() * 100)
            reihe = reihe.dropna()
            if len(reihe) < 13 or reihe.index[-1] != z.monate[-1]:
                continue
            diffs = reihe.diff().dropna()
            hist, letzte = diffs.iloc[:-1], diffs.iloc[-1]
            zwert = (letzte - hist.mean()) / hist.std() if hist.std() > 0 else 0
            if abs(zwert) >= config.SCHWELLE_AUSREISSER_Z:
                erg.append(Erkenntnis(kat, "Auffaellig", 40 + abs(zwert),
                                      f"Auffällig: {marke} bewegt sich im letzten Monat um {pp(letzte)} - "
                                      f"ungewöhnlich stark (z = {zahl(zwert)}). Datenlieferung prüfen oder Ursache klären.",
                                      eigen=True))

    for (kat, wer), p in (prognosen or {}).items():
        t = p.tabelle
        ende = t.iloc[-1]
        ist = p.historie["anteil"].iloc[-1]
        text = (f"Prognose {wer}: {zahl(ende['prognose'])} % im {ende['monat']} (heute {zahl(ist)} %, "
                f"80-%-Band {zahl(ende['unten'])}–{zahl(ende['oben'])} %).")
        if p.sensitivitaet:
            s = p.sensitivitaet
            text += (f" Hebel: +1 % Preis ≈ {pp(s['Preis +1 %'])}, +1 Pkt. Distribution ≈ "
                     f"{pp(s['Distribution +1 pp'])}.")
        erg.append(Erkenntnis(kat, "Prognose", 20, text, eigen=True))

    erg.sort(key=lambda e: (-e.gewicht))
    return erg


def pro_kategorie(liste: list[Erkenntnis], kategorie: str, typen: tuple[str, ...] | None = None,
                  max_anzahl: int = 6) -> list[Erkenntnis]:
    out = [e for e in liste if e.kategorie == kategorie and (typen is None or e.typ in typen)]
    return out[:max_anzahl]


def glaetten_mit_ollama(texte: list[str], modell: str | None = None, url: str = "http://localhost:11434") -> list[str]:
    """Optional: Stichpunkte von einem lokalen Modell sprachlich glaetten lassen.
    Faellt bei jedem Fehler still auf die Originaltexte zurueck."""
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=3) as r:
            modelle = [m["name"] for m in json.loads(r.read())["models"]]
        modell = modell or next((m for m in modelle if m.startswith(("qwen", "llama", "gemma", "mistral"))), None)
        if not modell:
            return texte
        prompt = ("Formuliere die folgenden Stichpunkte für eine Management-Präsentation knapp und professionell "
                  "auf Deutsch um. Ändere KEINE Zahlen, Namen oder Fakten. Gib genau eine Zeile pro Stichpunkt "
                  "zurück, ohne Nummerierung.\n\n" + "\n".join(texte))
        body = json.dumps({"model": modell, "prompt": prompt, "stream": False, "think": False,
                           "options": {"temperature": 0.2}}).encode()
        req = urllib.request.Request(url + "/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            antwort = json.loads(r.read())["response"].strip().splitlines()
        antwort = [a.strip("-• ").strip() for a in antwort if a.strip()]
        # Sicherheitscheck: alle Zahlen muessen erhalten bleiben
        import re
        for alt, neu in zip(texte, antwort):
            if sorted(re.findall(r"\d+(?:,\d+)?", alt)) != sorted(re.findall(r"\d+(?:,\d+)?", neu)):
                return texte
        return antwort if len(antwort) == len(texte) else texte
    except Exception:
        return texte
