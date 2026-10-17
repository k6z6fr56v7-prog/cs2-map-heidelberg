"""Marktanteilsprognose fuer den S&OP-Prozess, mit Rueckblicktest und Szenarien.

Zwei Modelle, die kombiniert werden:

A) Treibermodell: logit(Anteil) = Trend + Preisindex + Distribution + Werbedruck
   (Ridge-Regression). Erklaert, WARUM sich der Anteil bewegt, und erlaubt
   Szenarien ("Was passiert bei +5 % Preis?").
B) Basismodell: gedaempfte Trendfortschreibung (Holt) auf logit(Anteil).
   Robust, braucht keine Annahmen zu Treibern.

Die Gewichte der Kombination und die Unsicherheitsbaender kommen aus einem
Rueckblicktest: Das Modell wird mehrfach "in der Vergangenheit" gestartet und
gegen die tatsaechlichen Werte gemessen (rollierender Ursprung).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ADSTOCK = 0.6
LAMBDA = 4.0  # Ridge-Staerke (zieht unsichere Treiberwirkungen Richtung 0)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float) / 100, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 100 / (1 + np.exp(-np.asarray(x, dtype=float)))


def adstock(x: np.ndarray, rate: float = ADSTOCK) -> np.ndarray:
    out = np.zeros(len(x))
    for t, v in enumerate(np.nan_to_num(x)):
        out[t] = v + (rate * out[t - 1] if t else 0)
    return out * (1 - rate)


# ---------------------------------------------------------------- Daten


def treiber_reihe(df: pd.DataFrame, kategorie: str, wer: str, ebene: str = "marke") -> pd.DataFrame:
    """Monatliche Reihe: Anteil, relativer Preis, Distribution, Werbedruck."""
    d = df[df["kategorie"] == kategorie]
    markt = d.groupby("monat").agg(absatz=("absatz", "sum"), umsatz=("umsatz", "sum"))
    eig = d[d[ebene] == wer]
    e = eig.groupby("monat").agg(absatz=("absatz", "sum"), umsatz=("umsatz", "sum"))
    dist = (eig.assign(_w=eig["distribution_gew"] * eig["absatz"]).groupby("monat")["_w"].sum() / e["absatz"])
    werbung = eig.groupby(["monat", "marke"])["werbedruck"].mean().groupby("monat").sum()
    r = pd.DataFrame(index=markt.index)
    r["anteil"] = e["absatz"] / markt["absatz"] * 100
    r["preisindex"] = (e["umsatz"] / e["absatz"]) / (markt["umsatz"] / markt["absatz"]) * 100
    r["distribution"] = dist
    r["werbedruck"] = werbung
    r = r.dropna(subset=["anteil"])
    r.index.name = "monat"
    return r


def _design(r: pd.DataFrame, t0: int = 0) -> np.ndarray:
    n = len(r)
    t = (np.arange(n) + t0) / 12.0
    spalten = [np.ones(n), t, np.log(r["preisindex"].to_numpy() / 100)]
    if r["distribution"].notna().all():
        spalten.append(np.log(np.clip(r["distribution"].to_numpy(), 1, 100) / 100))
    if r["werbedruck"].notna().all():
        spalten.append(np.log1p(r["werbe_adstock"].to_numpy() / 100))
    return np.column_stack(spalten)


TREIBER_NAMEN = ["Konstante", "Trend pro Jahr", "Preisindex (log)", "Distribution (log)", "Werbedruck (Adstock)"]


# ---------------------------------------------------------------- Modelle


@dataclass
class Treibermodell:
    koef: np.ndarray
    mittel: np.ndarray
    std: np.ndarray
    n_hist: int

    @classmethod
    def fit(cls, r: pd.DataFrame) -> "Treibermodell":
        X = _design(r)
        y = logit(r["anteil"])
        mittel = X[:, 1:].mean(axis=0)
        std = X[:, 1:].std(axis=0)
        std[std == 0] = 1
        Xs = np.column_stack([X[:, 0], (X[:, 1:] - mittel) / std])
        P = LAMBDA * np.eye(Xs.shape[1])
        P[0, 0] = 0  # Konstante nicht bestrafen
        koef = np.linalg.solve(Xs.T @ Xs + P, Xs.T @ y)
        return cls(koef, mittel, std, len(r))

    def vorhersage(self, r_zukunft: pd.DataFrame) -> np.ndarray:
        X = _design(r_zukunft, t0=self.n_hist)
        Xs = np.column_stack([X[:, 0], (X[:, 1:] - self.mittel) / self.std])
        return inv_logit(Xs @ self.koef)

    def einfluss(self) -> dict[str, float]:
        """Koeffizienten in Originaleinheiten (logit-Skala)."""
        raw = self.koef[1:] / self.std
        return dict(zip(TREIBER_NAMEN[1:1 + len(raw)], raw))


@dataclass
class Holt:
    niveau: float
    trend: float
    phi: float

    @classmethod
    def fit(cls, y_prozent: np.ndarray, phi: float = 0.9) -> "Holt":
        y = logit(y_prozent)
        best = None
        for a in np.linspace(0.1, 0.9, 9):
            for b in (0.01, 0.05, 0.1, 0.2, 0.3):
                lvl, tr, sse = y[0], (y[min(3, len(y) - 1)] - y[0]) / max(min(3, len(y) - 1), 1), 0.0
                for v in y[1:]:
                    f = lvl + phi * tr
                    sse += (v - f) ** 2
                    neu = a * v + (1 - a) * f
                    tr = b * (neu - lvl) + (1 - b) * phi * tr
                    lvl = neu
                if best is None or sse < best[0]:
                    best = (sse, lvl, tr)
        return cls(best[1], best[2], phi)

    def vorhersage(self, h: int) -> np.ndarray:
        schritte = np.cumsum(self.phi ** np.arange(1, h + 1))
        return inv_logit(self.niveau + schritte * self.trend)


# ---------------------------------------------------------------- Szenarien


@dataclass
class Szenario:
    name: str = "Basis"
    preis_prozent: float = 0.0          # Preisaenderung der eigenen Marke in %
    distribution_pp: float = 0.0        # Veraenderung gewichtete Distribution in pp
    werbedruck_prozent: float = 0.0     # Veraenderung Werbedruck in %
    ab_monat: str | None = None         # ab wann (Standard: ab erstem Prognosemonat)


def zukunft_treiber(r: pd.DataFrame, monate: list[str], s: Szenario) -> pd.DataFrame:
    """Planannahmen: Treiber bleiben auf dem Niveau der letzten 3 Monate, plus Szenario."""
    basis = r.tail(3).mean(numeric_only=True)
    z = pd.DataFrame(index=pd.Index(monate, name="monat"))
    aktiv = np.array([m >= (s.ab_monat or monate[0]) for m in monate], dtype=float)
    z["preisindex"] = basis["preisindex"] * (1 + s.preis_prozent / 100 * aktiv)
    z["distribution"] = np.clip(basis["distribution"] + s.distribution_pp * aktiv, 1, 100) \
        if pd.notna(basis.get("distribution")) else np.nan
    z["werbedruck"] = basis["werbedruck"] * (1 + s.werbedruck_prozent / 100 * aktiv) \
        if pd.notna(basis.get("werbedruck")) else np.nan
    return z


def _mit_adstock(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["werbe_adstock"] = adstock(r["werbedruck"].to_numpy()) if r["werbedruck"].notna().all() else np.nan
    return r


# ---------------------------------------------------------------- Gesamtprognose


@dataclass
class Prognose:
    wer: str
    kategorie: str
    historie: pd.DataFrame
    tabelle: pd.DataFrame                    # monat, prognose, unten, oben, treiber, basis
    rueckblick: pd.DataFrame                 # horizont, mae_treiber, mae_basis, mae_kombi
    gewicht_treiber: float
    einfluss: dict[str, float]
    szenarien: dict[str, pd.DataFrame] = field(default_factory=dict)
    sensitivitaet: dict[str, float] = field(default_factory=dict)
    hinweis: str = ""


def _ein_lauf(r: pd.DataFrame, monate: list[str], s: Szenario, nutze_treiber: bool):
    h = len(monate)
    basis = Holt.fit(r["anteil"].to_numpy()).vorhersage(h)
    if not nutze_treiber:
        return basis, basis
    voll = _mit_adstock(pd.concat([r, zukunft_treiber(r, monate, s)]))
    hist, zuk = voll.iloc[:len(r)], voll.iloc[len(r):]
    modell = Treibermodell.fit(hist)
    treiber = modell.vorhersage(zuk)
    # Restfehler des Modells als AR(1)-Prozess fortschreiben: Die aktuelle Abweichung
    # vom Modell klingt mit der geschaetzten Rate ab (statt sie ewig mitzuschleppen).
    rest = logit(hist["anteil"]) - logit(modell.vorhersage(hist))
    rho = float(np.clip(np.corrcoef(rest[:-1], rest[1:])[0, 1], 0, 0.95)) if len(rest) > 3 else 0.0
    korrektur = rest[-1] * rho ** np.arange(1, h + 1)
    treiber = inv_logit(logit(treiber) + korrektur)
    return treiber, basis


def prognostiziere(df: pd.DataFrame, kategorie: str, wer: str, horizont: int = 12, ebene: str = "marke",
                   szenarien: list[Szenario] | None = None, rueckblick_monate: int = 12) -> Prognose:
    r = treiber_reihe(df, kategorie, wer, ebene)
    nutze_treiber = len(r) >= 18 and r["preisindex"].notna().all()
    hinweis = "" if nutze_treiber else "Zu kurze Historie fuer das Treibermodell - nur Trendfortschreibung."
    letzte = pd.Period(r.index[-1], freq="M")
    monate = [str(letzte + i) for i in range(1, horizont + 1)]

    # Rueckblicktest mit rollierendem Ursprung
    fehler = []
    max_h = min(6, horizont)
    starts = range(max(18, len(r) - rueckblick_monate - max_h), len(r) - 1) if len(r) > 20 else []
    for s0 in starts:
        train = r.iloc[:s0]
        ist = r.iloc[s0:s0 + max_h]["anteil"].to_numpy()
        tr, ba = _ein_lauf(train, list(r.index[s0:s0 + len(ist)]), Szenario(), nutze_treiber)
        for k in range(len(ist)):
            fehler.append((k + 1, ist[k] - tr[k], ist[k] - ba[k]))
    fe = pd.DataFrame(fehler, columns=["horizont", "f_treiber", "f_basis"])
    if len(fe):
        mae_t = fe["f_treiber"].abs().mean()
        mae_b = fe["f_basis"].abs().mean()
        w = (1 / mae_t) / (1 / mae_t + 1 / mae_b) if nutze_treiber and mae_t > 0 and mae_b > 0 else 0.0
        fe["f_kombi"] = w * fe["f_treiber"] + (1 - w) * fe["f_basis"]
        rb = fe.groupby("horizont").agg(mae_treiber=("f_treiber", lambda x: x.abs().mean()),
                                        mae_basis=("f_basis", lambda x: x.abs().mean()),
                                        mae_kombi=("f_kombi", lambda x: x.abs().mean()),
                                        q80=("f_kombi", lambda x: x.abs().quantile(0.8))).reset_index()
    else:
        w = 0.5 if nutze_treiber else 0.0
        rb = pd.DataFrame(columns=["horizont", "mae_treiber", "mae_basis", "mae_kombi", "q80"])

    def kombiniere(s: Szenario) -> pd.DataFrame:
        tr, ba = _ein_lauf(r, monate, s, nutze_treiber)
        p = w * tr + (1 - w) * ba
        if len(rb):
            q = rb.set_index("horizont")["q80"]
            breite = np.array([q.get(h, q.iloc[-1] * np.sqrt(h / q.index[-1])) for h in range(1, horizont + 1)])
        else:  # ohne Rueckblicktest: Streuung der Monatsveraenderungen als grobe Bandbreite
            sd = np.nanstd(np.diff(r["anteil"].to_numpy())) if len(r) > 3 else np.nan
            breite = 1.28 * sd * np.sqrt(np.arange(1, horizont + 1))
        return pd.DataFrame({"monat": monate, "prognose": p, "unten": p - breite, "oben": p + breite,
                             "treiber": tr, "basis": ba})

    tabelle = kombiniere(Szenario())
    erg_szen = {s.name: kombiniere(s) for s in (szenarien or [])}

    einfluss, sens = {}, {}
    if nutze_treiber:
        hist = _mit_adstock(r)
        einfluss = Treibermodell.fit(hist).einfluss()
        basis_ende = tabelle["prognose"].iloc[-1]
        for name, s in (("Preis +1 %", Szenario(preis_prozent=1)),
                        ("Distribution +1 pp", Szenario(distribution_pp=1)),
                        ("Werbedruck +10 %", Szenario(werbedruck_prozent=10))):
            sens[name] = kombiniere(s)["prognose"].iloc[-1] - basis_ende
    return Prognose(wer, kategorie, r, tabelle, rb, w, einfluss, erg_szen, sens, hinweis)
