"""Zentrale Einstellungen: Spaltennamen, Farben, Schwellenwerte.

Echte Datenquellen (Handelspanel, Abverkauf, Distribution, Mediadaten) haben
eigene Spaltennamen. Diese werden beim Laden per SPALTEN-Mapping auf das
interne Format gebracht - siehe laden.py und beispiel_mapping.json.
"""

from __future__ import annotations

# Internes Datenformat (eine Zeile = Monat x Region x Kanal x Marke)
PFLICHT_SPALTEN = [
    "monat",            # Periode, z.B. 2026-09
    "region",           # Verkaufsgebiet
    "kanal",            # Absatzkanal
    "kategorie",        # Produktkategorie
    "hersteller",
    "marke",
    "absatz",           # Menge (Packungen / Einheiten)
    "umsatz",           # Umsatz in EUR
]
OPTIONALE_SPALTEN = [
    "distribution_gew",  # gewichtete Distribution in %
    "werbedruck",        # Mediadruck / Marketingindex
]

# Der eigene Hersteller (alle Auswertungen werden aus dieser Sicht formuliert)
EIGENER_HERSTELLER = "Eigen AG"

# Schwellenwerte fuer automatische Kommentare
SCHWELLE_ANTEIL_PP = 0.3        # ab dieser Veraenderung (Prozentpunkte) wird kommentiert
SCHWELLE_PREIS_INDEX = 3.0      # Punkte Preisindex-Veraenderung
SCHWELLE_DISTRIBUTION_PP = 5.0  # Prozentpunkte gewichtete Distribution
SCHWELLE_AUSREISSER_Z = 3.0     # z-Wert fuer Ausreisser

# Farben (validierte Palette, farbfehlsichtig-tauglich): eigene Marke hervorgehoben,
# Wettbewerb in den naechsten Kategorialfarben, Rest grau.
FARBE_EIGEN = "2A78D6"
FARBEN_WETTBEWERB = ["EB6834", "1BAF7A", "EDA100"]
FARBE_UEBRIGE = "A8A6A0"
FARBE_PROGNOSE_BAND = "B7D3F6"
DIVERGIEREND_MITTE = "F0EFEC"
DIVERGIEREND_PLUS = "2A78D6"   # Zugewinn
DIVERGIEREND_MINUS = "E34948"  # Verlust

TINTE = "0B0B0B"
TINTE_2 = "52514E"
TINTE_GEDAEMPFT = "898781"
GITTER = "E1E0D9"
DUNKEL = "0D366B"   # Hintergrund Titel-Folien
HELL_AUF_DUNKEL = "CDE2FB"


def mische(hex_a: str, hex_b: str, t: float) -> str:
    """Lineare Mischung zweier Hex-Farben (t=0 -> a, t=1 -> b)."""
    a = [int(hex_a[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(hex_b[i:i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def divergierende_farbe(wert: float, grenze: float) -> str:
    """Farbe fuer eine Veraenderung: blau = Zugewinn, rot = Verlust, grau = neutral."""
    t = max(-1.0, min(1.0, wert / grenze if grenze else 0.0))
    ziel = DIVERGIEREND_PLUS if t >= 0 else DIVERGIEREND_MINUS
    # in 3 Stufen pro Seite quantisieren, damit gleiche Werte gleich aussehen
    stufe = round(abs(t) * 3) / 3
    return mische(DIVERGIEREND_MITTE, ziel, stufe)
