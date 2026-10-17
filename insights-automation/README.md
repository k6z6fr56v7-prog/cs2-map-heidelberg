# Insights-Automation

Automatisiert die Routinearbeit eines **Business Insights Executive** im Konsumgütermarkt
(hier am Beispiel Nikotinkategorien). Aus den monatlichen Performance-Daten entstehen in
wenigen Sekunden:

- ein **Excel-Report** mit Übersicht, Markenrankings, Segmentanalyse, Zeitreihen,
  Prognose und Datenqualität, inklusive echter Excel-Diagramme
- ein **PowerPoint-Deck** für das Management (11 Folien): Kernaussagen, je Kategorie eine
  Folie mit Aussage-Titel und Kommentar, Gewinner/Verlierer, Segment-Heatmap, Prognose mit
  Szenarien, Methodik. Alle Diagramme sind **native PowerPoint-Diagramme** und lassen sich
  nachträglich bearbeiten. Sprechernotizen sind enthalten.

> **Beispieldaten:** Alle Hersteller, Marken und Zahlen im Demo-Datensatz sind **frei
> erfunden**. Sie zeigen nur, was das Werkzeug kann. Mit echten Daten funktioniert es
> genauso (siehe [Eigene Daten](#eigene-daten)).

## Was automatisiert wird

| Aufgabe aus der Stellenbeschreibung | Was das Werkzeug macht |
|---|---|
| Analysen auf Basis vieler Datenquellen | Liest mehrere Quellen (CSV/Excel, z. B. Handelspanel + Distribution + Media), bringt sie per `mapping.json` auf ein Format und **prüft die Datenqualität** (Lücken, Dubletten, Preissprünge, unplausible Werte), bevor irgendetwas berichtet wird |
| Standard-Reportings | Marktanteile Menge/Wert, Veränderung in pp vs. Vorjahr und Vorperiode, Preisindex, gewichtete Distribution, Ranking je Kategorie, rollierend 3 Monate oder monatlich |
| Wettbewerbsanalyse | Gewinner/Verlierer, Neueinführungen, Preisaktionen des Wettbewerbs |
| Ursachen finden | **Beitragsanalyse:** zerlegt die nationale Anteilsveränderung exakt in Beiträge je Region × Kanal (Performance + Mix) und prüft dort die Treiber (Distribution, Preis, Wettbewerber) |
| Marktanteilsprognose für S&OP | Treibermodell (Preis, Distribution, Werbedruck) kombiniert mit gedämpfter Trendfortschreibung. Gewichte und **80-%-Band aus Rückblicktest** (wie gut hätte die Prognose in der Vergangenheit getroffen?) |
| Marketingpläne bewerten | **Szenarien:** „Was passiert bei +5 % Preis / +3 Pkt. Distribution / +20 % Werbedruck?“ und Hebel pro Einheit |
| Ergebnisse und Empfehlungen präsentieren | Automatische Kommentare auf Deutsch („Atlas verliert 1,5 pp … größter Einzelbeitrag Kiosk & Fachhandel Süd, gewichtete Distribution −16 Pkt.“) und das fertige Deck |

## Schnellstart (Windows)

1. `1_installieren.bat`: richtet Python-Umgebung und Pakete ein (einmalig, ca. 2 Minuten)
2. `2_demo_bericht.bat`: erzeugt Beispieldaten und den kompletten Bericht. Der Ordner
   mit Excel und PowerPoint öffnet sich danach automatisch.

Voraussetzung: Python 3.11 oder neuer ([python.org](https://www.python.org/downloads/)).

## Befehle

```powershell
python -m insights demo                                   # Beispieldaten + Bericht
python -m insights demo --bis 2025-06                     # Bericht für einen früheren Monat
python -m insights bericht --daten daten\mapping.json     # eigene Daten
python -m insights bericht --daten daten.csv --fenster 1  # Monatsbetrachtung statt 3 Monate
python -m insights pruefen --daten daten\mapping.json     # nur Datenqualität
python -m insights prognose --daten demo\daten\mapping.json --kategorie Zigaretten --marke Atlas ^
       --szenario "Preiserhoehung:preis=5" --szenario "Listung:distribution=3,werbung=20"
python -m insights demo --ollama                          # Texte zusätzlich von lokalem Ollama glätten lassen
```

`--ollama` nutzt ein lokales Sprachmodell (z. B. Qwen in Ollama), um die Kommentare
sprachlich zu glätten. Ein Sicherheitscheck verwirft die Umformulierung, wenn sich
dabei auch nur eine Zahl verändert.

## Eigene Daten

**Internes Format** (eine Zeile pro Monat × Region × Kanal × Marke):

| Spalte | Beispiel | Pflicht |
|---|---|---|
| `monat` | 2026-09 | ja |
| `region`, `kanal`, `kategorie`, `hersteller`, `marke` | Nord, LEH, Zigaretten, … | ja |
| `absatz` | Packungen/Einheiten | ja |
| `umsatz` | EUR | ja |
| `distribution_gew` | gewichtete Distribution in % | nein, aber für Treiberanalyse nötig |
| `werbedruck` | Media-Index / GRP | nein |

**Andere Spaltennamen oder mehrere Quellen:** eine `mapping.json` daneben legen. Die
Demo erzeugt eine Vorlage unter `demo\daten\mapping.json`:

```json
{
  "quellen": [
    {"datei": "panel_export.csv", "trennzeichen": ";", "dezimal": ",",
     "spalten": {"Period": "monat", "Area": "region", "Channel": "kanal", "Category": "kategorie",
                 "Manufacturer": "hersteller", "Brand": "marke", "Units": "absatz", "Value EUR": "umsatz"}},
    {"datei": "distribution_media.xlsx", "blatt": "Daten",
     "spalten": {"Period": "monat", "Area": "region", "Channel": "kanal", "Brand": "marke",
                 "WD %": "distribution_gew", "Media Index": "werbedruck"},
     "verknuepfen_ueber": ["monat", "region", "kanal", "marke"]}
  ],
  "eigener_hersteller": "Eigen AG"
}
```

Schwellenwerte für Kommentare (ab wie viel pp etwas erwähnt wird usw.) und Farben stehen
in `insights/config.py`.

## Wie gut ist die Prognose?

Bei jedem Bericht wird die Prognose im **Rückblicktest** geprüft: Das Modell startet
mehrfach in der Vergangenheit und wird gegen die tatsächlichen Werte gemessen. Auf dem
Deck steht die Treffsicherheit („± 0,5 pp drei Monate voraus“), im Excel die Fehler je
Horizont für beide Teilmodelle. Die Bandbreite (80 %) kommt aus genau diesen Fehlern und
ist keine Annahme.

**Grenzen:** Treiberwirkungen sind statistische Schätzungen aus der Historie, keine
Kausalbeweise. Ereignisse ohne Vorbild in den Daten, etwa eine große Neueinführung,
sieht kein Modell voraus. Dafür gibt es die Szenarien.

## Projektstruktur

```
insights/
  config.py        Spaltennamen, Eigener Hersteller, Schwellenwerte, Farben
  beispieldaten.py Synthetischer Markt mit eingebauten Ereignissen (fiktiv)
  laden.py         Einlesen (CSV/Excel/Mapping) + Datenqualitätsprüfung
  kennzahlen.py    Anteile, Ranking, Preisindex, Distribution, Segment- und Beitragsanalyse
  prognose.py      Treibermodell + Trendmodell, Rückblicktest, Szenarien
  kommentar.py     Automatische Erkenntnisse auf Deutsch (+ optional Ollama)
  excel.py         Excel-Report
  folien.py        PowerPoint-Deck
  bericht.py       Alles in einem Aufruf
tests/             19 Tests (python -m pytest tests -q)
```
