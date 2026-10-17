"""Standard-Kennzahlen: Marktanteile, Veraenderungen, Preisindex, Distribution,
Segment- und Beitragsanalyse.

Konventionen:
- Anteile in Prozent, Veraenderungen von Anteilen in Prozentpunkten (pp).
- "fenster" = Anzahl Monate, ueber die summiert wird (1 = Monat, 3 = rollierend 3 Monate).
- VJ = Vorjahreszeitraum, VP = Vorperiode (das Fenster direkt davor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Zeitraum:
    monate: list[str]
    vj: list[str]
    vp: list[str]

    @property
    def name(self) -> str:
        return self.monate[-1] if len(self.monate) == 1 else f"{self.monate[0]} bis {self.monate[-1]}"


def zeitraum(alle_monate: list[str], ende: str | None = None, fenster: int = 3) -> Zeitraum:
    alle = sorted(alle_monate)
    ende = ende or alle[-1]
    e = pd.Period(ende, freq="M")
    monate = [str(e - i) for i in range(fenster - 1, -1, -1)]
    vj = [str(pd.Period(m, freq="M") - 12) for m in monate]
    vp = [str(pd.Period(m, freq="M") - fenster) for m in monate]
    return Zeitraum(monate, vj, vp)


def _summe(df: pd.DataFrame, monate: list[str], gruppen: list[str]) -> pd.DataFrame:
    teil = df[df["monat"].isin(monate)]
    agg = teil.groupby(gruppen, dropna=False).agg(absatz=("absatz", "sum"), umsatz=("umsatz", "sum")).reset_index()
    # Distribution: gewichtet nach Absatz gemittelt
    if teil["distribution_gew"].notna().any():
        w = teil.assign(_dw=teil["distribution_gew"] * teil["absatz"])
        d = w.groupby(gruppen, dropna=False).agg(_dw=("_dw", "sum"), _a=("absatz", "sum")).reset_index()
        agg = agg.merge(d.assign(distribution=d["_dw"] / d["_a"])[gruppen + ["distribution"]], on=gruppen, how="left")
    else:
        agg["distribution"] = np.nan
    return agg


def anteile_zeitreihe(df: pd.DataFrame, ebene: str = "marke", basis: str = "absatz",
                      filter_: dict | None = None) -> pd.DataFrame:
    """Monatliche Anteile je Kategorie und Marke (oder Hersteller), national oder gefiltert."""
    d = df
    for k, v in (filter_ or {}).items():
        d = d[d[k] == v]
    g = d.groupby(["kategorie", "monat", ebene])[basis].sum().reset_index()
    g["anteil"] = g[basis] / g.groupby(["kategorie", "monat"])[basis].transform("sum") * 100
    return g


def ranking(df: pd.DataFrame, kategorie: str, z: Zeitraum, ebene: str = "marke") -> pd.DataFrame:
    """Marken (oder Hersteller) einer Kategorie mit Anteil, Veraenderung, Preisindex, Distribution."""
    d = df[df["kategorie"] == kategorie]
    gruppen = [ebene] if ebene == "hersteller" else ["hersteller", "marke"]
    teile = {}
    for name, monate in (("akt", z.monate), ("vj", z.vj), ("vp", z.vp)):
        s = _summe(d, monate, gruppen)
        s["anteil_menge"] = s["absatz"] / s["absatz"].sum() * 100 if s["absatz"].sum() else np.nan
        s["anteil_wert"] = s["umsatz"] / s["umsatz"].sum() * 100 if s["umsatz"].sum() else np.nan
        markt_preis = s["umsatz"].sum() / s["absatz"].sum() if s["absatz"].sum() else np.nan
        s["preisindex"] = (s["umsatz"] / s["absatz"]) / markt_preis * 100
        teile[name] = s.set_index(gruppen)
    akt = teile["akt"]
    out = akt[["absatz", "umsatz", "anteil_menge", "anteil_wert", "preisindex", "distribution"]].copy()
    for name in ("vj", "vp"):
        alt = teile[name].reindex(out.index)
        out[f"anteil_menge_{name}"] = alt["anteil_menge"].fillna(0)
        out[f"delta_menge_{name}"] = out["anteil_menge"] - out[f"anteil_menge_{name}"]
        out[f"delta_wert_{name}"] = out["anteil_wert"] - alt["anteil_wert"].fillna(0)
        out[f"delta_preisindex_{name}"] = out["preisindex"] - alt["preisindex"]
        out[f"delta_distribution_{name}"] = out["distribution"] - alt["distribution"]
    out["absatz_vj"] = teile["vj"].reindex(out.index)["absatz"]
    out["absatz_veraenderung_vj"] = (out["absatz"] / out["absatz_vj"] - 1) * 100
    return out.reset_index().sort_values("anteil_menge", ascending=False).reset_index(drop=True)


def markt_ueberblick(df: pd.DataFrame, z: Zeitraum, eigener: str) -> pd.DataFrame:
    """Eine Zeile je Kategorie: Marktvolumen, eigener Anteil, Veraenderungen, Preisindex."""
    zeilen = []
    for kat in sorted(df["kategorie"].unique()):
        d = df[df["kategorie"] == kat]
        vol = d[d["monat"].isin(z.monate)]["absatz"].sum()
        vol_vj = d[d["monat"].isin(z.vj)]["absatz"].sum()
        h = ranking(df, kat, z, ebene="hersteller").set_index("hersteller")
        if eigener in h.index:
            e = h.loc[eigener]
            zeilen.append({
                "kategorie": kat, "markt_absatz": vol,
                "markt_veraenderung_vj": (vol / vol_vj - 1) * 100 if vol_vj else np.nan,
                "anteil_menge": e["anteil_menge"], "delta_menge_vj": e["delta_menge_vj"],
                "delta_menge_vp": e["delta_menge_vp"], "anteil_wert": e["anteil_wert"],
                "delta_wert_vj": e["delta_wert_vj"], "preisindex": e["preisindex"],
                "distribution": e["distribution"], "rang": int((h["anteil_menge"] > e["anteil_menge"]).sum() + 1),
            })
    return pd.DataFrame(zeilen)


def segment_matrix(df: pd.DataFrame, kategorie: str, z: Zeitraum, wer: str, ebene: str = "marke") -> pd.DataFrame:
    """Anteil und Veraenderung vs. VJ je Region x Kanal fuer eine Marke/einen Hersteller."""
    d = df[df["kategorie"] == kategorie]
    erg = []
    for (region, kanal), g in d.groupby(["region", "kanal"]):
        akt = g[g["monat"].isin(z.monate)]
        alt = g[g["monat"].isin(z.vj)]
        a = akt[akt[ebene] == wer]["absatz"].sum() / akt["absatz"].sum() * 100 if akt["absatz"].sum() else np.nan
        b = alt[alt[ebene] == wer]["absatz"].sum() / alt["absatz"].sum() * 100 if alt["absatz"].sum() else np.nan
        erg.append({"region": region, "kanal": kanal, "anteil": a, "delta_vj": a - b,
                    "segment_absatz": akt["absatz"].sum()})
    return pd.DataFrame(erg)


def beitragsanalyse(df: pd.DataFrame, kategorie: str, z: Zeitraum, wer: str, ebene: str = "marke") -> pd.DataFrame:
    """Zerlegt die nationale Anteilsveraenderung (pp vs. VJ) in Beitraege der Segmente.

    Nationaler Anteil = Summe(Segmentgewicht x Segmentanteil). Die Veraenderung
    wird je Segment aufgeteilt in
      - Performance: Anteilsveraenderung im Segment x Gewicht (wie gut wir dort laufen)
      - Mix: Gewichtsveraenderung des Segments x (Segmentanteil - nationaler Anteil)
    Die Summe beider Beitraege ueber alle Segmente ergibt genau die nationale Veraenderung.
    """
    m = segment_matrix(df, kategorie, z, wer, ebene)
    d = df[df["kategorie"] == kategorie]
    alt = d[d["monat"].isin(z.vj)].groupby(["region", "kanal"])["absatz"].sum()
    akt = d[d["monat"].isin(z.monate)].groupby(["region", "kanal"])["absatz"].sum()
    m = m.set_index(["region", "kanal"])
    w1 = akt / akt.sum()
    w0 = alt / alt.sum()
    s1 = m["anteil"]
    s0 = m["anteil"] - m["delta_vj"]
    national0 = (w0 * s0).sum()
    m["gewicht"] = w1 * 100
    m["performance"] = (s1 - s0) * (w0 + w1) / 2
    m["mix"] = (w1 - w0) * ((s0 + s1) / 2 - national0)
    m["beitrag"] = m["performance"] + m["mix"]
    return m.reset_index().sort_values("beitrag")


def eigene_marken(df: pd.DataFrame, eigener: str, kategorie: str | None = None) -> list[str]:
    d = df[df["hersteller"] == eigener]
    if kategorie:
        d = d[d["kategorie"] == kategorie]
    return list(d.groupby("marke")["absatz"].sum().sort_values(ascending=False).index)
