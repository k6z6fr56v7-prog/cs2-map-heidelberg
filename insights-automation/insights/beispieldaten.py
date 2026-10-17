"""Synthetischer Beispieldatensatz (erfundene Hersteller und Marken!).

Simuliert einen deutschen Nikotinmarkt mit drei Kategorien, vier Regionen und
drei Kanaelen ueber 48 Monate. Die Marktanteile entstehen aus einem
Attraktivitaetsmodell (Preis, Distribution, Werbedruck, Trend, Zufall). Damit
die Auswertung etwas zu finden hat, sind einige Ereignisse eingebaut - die
Automatik soll sie selbst entdecken:

- ab 2025-03: Listungsverlust der eigenen Marke "Atlas" im Kiosk/Fachhandel Sued
- 2025-11 bis 2026-02: Preisaktion des Wettbewerbers "Falke" in West
- ab 2025-06: Neueinfuehrung "Spark" bei E-Zigaretten
- ab 2026-04: Kampagne fuer die eigene Marke "Ember"
- ab 2026-01: Tabaksteuererhoehung, alle Zigarettenpreise +4 %

Alle Zahlen sind frei erfunden und nur zum Ausprobieren gedacht.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REGIONEN = {"Nord": 0.20, "West": 0.31, "Sued": 0.29, "Ost": 0.20}
KANAELE = {"LEH": 0.45, "Tankstelle": 0.25, "Kiosk & Fachhandel": 0.30}

# Kategorie: (Marktvolumen pro Monat in Packungen, Trend pro Jahr, Grundpreis EUR, Saisonamplitude)
KATEGORIEN = {
    "Zigaretten": (250e6, -0.04, 8.40, 0.04),
    "Tabakerhitzer": (22e6, 0.22, 6.90, 0.03),
    "E-Zigaretten": (9e6, 0.12, 7.50, 0.08),
}

# (Kategorie, Hersteller, Marke, Grundattraktivitaet, Preis-Faktor, Distribution-Basis, Trend/Jahr)
MARKEN = [
    ("Zigaretten", "Eigen AG", "Atlas", 1.30, 1.00, 95, -0.02),
    ("Zigaretten", "Eigen AG", "Meridian", 0.40, 1.06, 82, 0.00),
    ("Zigaretten", "Nordhaven Tobacco", "Falke", 1.15, 0.98, 94, 0.01),
    ("Zigaretten", "Nordhaven Tobacco", "Kompass", 0.30, 0.93, 80, 0.03),
    ("Zigaretten", "Westway Brands", "Basalt", 0.75, 1.02, 90, -0.01),
    ("Zigaretten", "Sudetta", "Lago", 0.20, 0.90, 75, 0.04),
    ("Tabakerhitzer", "Eigen AG", "Ember", 0.80, 1.00, 85, 0.05),
    ("Tabakerhitzer", "Nordhaven Tobacco", "Heatline", 1.40, 1.04, 92, 0.00),
    ("Tabakerhitzer", "Westway Brands", "Thermo", 0.30, 0.95, 70, 0.06),
    ("E-Zigaretten", "Eigen AG", "Cirrus", 1.00, 1.00, 80, 0.02),
    ("E-Zigaretten", "Westway Brands", "Nimbus", 0.90, 0.97, 78, 0.00),
    ("E-Zigaretten", "Sudetta", "Puffin", 0.60, 0.92, 70, 0.03),
    ("E-Zigaretten", "Nordhaven Tobacco", "Drift", 0.50, 1.05, 72, -0.03),
    ("E-Zigaretten", "Sudetta", "Spark", 0.00, 0.95, 0, 0.0),  # Neueinfuehrung 2025-06
]

# Regionale/kanalspezifische Staerken (Zufall, aber fest pro Marke)
PREIS_ELASTIZITAET = 2.5
DIST_EFFEKT = 1.6
WERBE_EFFEKT = 0.12
ADSTOCK = 0.6

RAW_SPALTEN = {
    "monat": "Period",
    "region": "Area",
    "kanal": "Channel",
    "kategorie": "Category",
    "hersteller": "Manufacturer",
    "marke": "Brand",
    "absatz": "Units",
    "umsatz": "Value EUR",
    "distribution_gew": "WD %",
    "werbedruck": "Media Index",
}


def monate(start: str = "2022-10", anzahl: int = 48) -> list[str]:
    return [str(p) for p in pd.period_range(start, periods=anzahl, freq="M")]


def erzeuge(seed: int = 7, start: str = "2022-10", anzahl: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    perioden = monate(start, anzahl)
    idx = {m: i for i, m in enumerate(perioden)}
    jahre = np.arange(anzahl) / 12.0

    def ab(monat: str) -> np.ndarray:
        """1 ab diesem Monat, sonst 0."""
        return (np.arange(anzahl) >= idx.get(monat, anzahl)).astype(float)

    def zwischen(von: str, bis: str) -> np.ndarray:
        return ab(von) * (1 - ab(bis))

    steuer = 1 + 0.04 * ab("2026-01")
    zeilen = []
    for kat, (vol, trend, grundpreis, saison) in KATEGORIEN.items():
        marken = [m for m in MARKEN if m[0] == kat]
        monat_nr = np.array([int(m[-2:]) for m in perioden])
        markt = vol * (1 + trend) ** jahre * (1 + saison * np.sin((monat_nr - 4) / 12 * 2 * np.pi))
        markt *= 1 + rng.normal(0, 0.015, anzahl)
        # Werbedruck pro Marke (Kampagnenwellen)
        werbung = {}
        for _, _, marke, *_ in marken:
            w = np.clip(rng.normal(40, 12, anzahl), 0, None)
            for start_idx in rng.choice(anzahl - 3, size=3, replace=False):
                w[start_idx:start_idx + 3] += rng.uniform(25, 45)
            if marke == "Ember":
                w += 260 * zwischen("2026-04", "2026-08")
            if marke == "Spark":
                w = w * ab("2025-06") + 90 * zwischen("2025-06", "2025-10")
            werbung[marke] = w
        adstock = {}
        for marke, w in werbung.items():
            a = np.zeros(anzahl)
            for t in range(anzahl):
                a[t] = w[t] + (ADSTOCK * a[t - 1] if t else 0)
            adstock[marke] = a / (1 - ADSTOCK)

        for region, r_gew in REGIONEN.items():
            for kanal, k_gew in KANAELE.items():
                seg_gew = r_gew * k_gew * rng.uniform(0.9, 1.1)
                if kat != "Zigaretten" and kanal == "Tankstelle":
                    seg_gew *= 0.7
                nutzen, preise, dists = {}, {}, {}
                for _, hersteller, marke, basis, preisfaktor, dist_basis, mtrend in marken:
                    affin = rng.normal(0, 0.25)
                    preis = grundpreis * preisfaktor * (1 + 0.025) ** jahre * (steuer if kat == "Zigaretten" else 1)
                    preis = preis * (1 + rng.normal(0, 0.004, anzahl))
                    # gelegentliche Preisaktionen (-3 bis -6 % fuer 1-2 Monate)
                    for a in rng.choice(anzahl - 2, size=3, replace=False):
                        preis[a:a + rng.integers(1, 3)] *= 1 - rng.uniform(0.03, 0.06)
                    dist = np.clip(dist_basis + rng.normal(0, 1.5, anzahl), 5, 99.5)
                    if marke == "Atlas" and region == "Sued" and kanal == "Kiosk & Fachhandel":
                        dist = dist - 17 * ab("2025-03")
                    if marke == "Falke" and region == "West":
                        preis = preis * (1 - 0.07 * zwischen("2025-11", "2026-03"))
                    if marke == "Spark":
                        anlauf = np.clip((np.arange(anzahl) - idx["2025-06"] + 1) / 6, 0, 1)
                        dist = np.where(anlauf > 0, 5 + 65 * anlauf + rng.normal(0, 1.5, anzahl), 0.1)
                        basis = 0.55
                    rausch = np.zeros(anzahl)
                    for t in range(anzahl):
                        rausch[t] = (0.8 * rausch[t - 1] if t else 0) + rng.normal(0, 0.03)
                    nutzen[marke] = (basis + affin + mtrend * jahre + DIST_EFFEKT * np.log(np.clip(dist, 0.1, 100) / 100)
                                     + WERBE_EFFEKT * np.log1p(adstock[marke] / 100) + rausch)
                    preise[marke] = preis
                    dists[marke] = np.clip(dist, 0, 100)
                mittelpreis = np.mean(list(preise.values()), axis=0)
                for marke in nutzen:
                    nutzen[marke] = nutzen[marke] - PREIS_ELASTIZITAET * np.log(preise[marke] / mittelpreis)
                    if marke == "Spark":
                        nutzen[marke] = np.where(ab("2025-06") > 0, nutzen[marke], -50)
                matrix = np.vstack(list(nutzen.values()))
                anteile = np.exp(matrix - matrix.max(axis=0))
                anteile /= anteile.sum(axis=0)
                seg_markt = markt * seg_gew
                for j, (_, hersteller, marke, *_rest) in enumerate(marken):
                    absatz = np.round(seg_markt * anteile[j])
                    for t, monat in enumerate(perioden):
                        if absatz[t] <= 0:
                            continue
                        zeilen.append((monat, region, kanal, kat, hersteller, marke, absatz[t],
                                       round(absatz[t] * preise[marke][t], 2), round(dists[marke][t], 1),
                                       round(werbung[marke][t], 1)))
    df = pd.DataFrame(zeilen, columns=["monat", "region", "kanal", "kategorie", "hersteller", "marke", "absatz",
                                       "umsatz", "distribution_gew", "werbedruck"])
    return df


def schreibe_beispiel(ordner: Path, seed: int = 7) -> dict[str, Path]:
    """Schreibt den Datensatz bewusst in einem 'fremden' Rohformat (englische Spalten,
    wie aus einem Panel-Export) plus Mapping - so wird auch das Einlesen vorgefuehrt."""
    ordner.mkdir(parents=True, exist_ok=True)
    df = erzeuge(seed)
    roh = df.rename(columns=RAW_SPALTEN)
    roh["Region Code"] = roh["Area"].map({"Nord": "N", "West": "W", "Sued": "S", "Ost": "O"})
    markt = roh.drop(columns=["WD %", "Media Index"])
    markt_pfad = ordner / "panel_export.csv"
    markt.to_csv(markt_pfad, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    # Distribution und Media kommen typischerweise aus eigenen Quellen
    zusatz = roh[["Period", "Area", "Channel", "Brand", "WD %", "Media Index"]]
    zusatz_pfad = ordner / "distribution_media.xlsx"
    zusatz.to_excel(zusatz_pfad, index=False, sheet_name="Daten")
    mapping = {
        "quellen": [
            {"datei": markt_pfad.name, "trennzeichen": ";", "dezimal": ",",
             "spalten": {v: k for k, v in RAW_SPALTEN.items() if v in markt.columns}},
            {"datei": zusatz_pfad.name, "blatt": "Daten",
             "spalten": {"Period": "monat", "Area": "region", "Channel": "kanal", "Brand": "marke",
                         "WD %": "distribution_gew", "Media Index": "werbedruck"},
             "verknuepfen_ueber": ["monat", "region", "kanal", "marke"]},
        ],
        "eigener_hersteller": "Eigen AG",
    }
    mapping_pfad = ordner / "mapping.json"
    mapping_pfad.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"panel": markt_pfad, "zusatz": zusatz_pfad, "mapping": mapping_pfad}
