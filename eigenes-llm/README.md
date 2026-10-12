# Eigenes Sprachmodell (Eigen-LM)

Ein deutsches Sprachmodell, **von Grund auf selbst trainiert**, auf deinem eigenen PC.
Kein Fine-Tuning eines fremden Modells: Tokenizer, Architektur, Training, Chat-Training
und Export sind hier komplett enthalten.

Am Ende läuft dein Modell:
- im Terminal (`5_chatten.bat`)
- im Browser (`6_chat_im_browser.bat`)
- in **Ollama** und **LM Studio** (`7_export_ollama.bat`)

> **Ehrliche Erwartung:** Ein Modell mit 30 Millionen Parametern, das eine Nacht auf einer
> RTX 3050 trainiert wurde, schreibt flüssiges, meist grammatisch richtiges Deutsch und
> lernt einiges an Wissen. Mit ChatGPT, Claude oder Qwen kann es nicht mithalten: Die haben
> tausendmal mehr Parameter und wurden mit tausendmal mehr Rechenleistung trainiert. Es
> erfindet Fakten und verliert bei langen Antworten den Faden. Dafür ist es zu 100 % deins,
> und du verstehst, wie so etwas funktioniert.

---

## Schnellstart (Windows)

Einfach nacheinander doppelklicken:

| Datei | Was passiert | Dauer (RTX 3050) |
|---|---|---|
| `1_installieren.bat` | Python-Umgebung, PyTorch mit CUDA, Pakete | 5–15 min |
| `2_daten_vorbereiten.bat` | ~2,6 GB deutschen Text laden (70 % Web, 30 % Wikipedia), Tokenizer trainieren | 30–90 min |
| `3_trainieren.bat` | Vortraining des Modells (Preset „klein“, 8 Stunden) | 8 h (über Nacht) |
| `4a_lehrer_ueber_nacht.bat` | *optional, sehr empfohlen:* dein Qwen in Ollama schreibt Chat-Beispiele | 8 h (zweite Nacht) |
| `4_chat_training.bat` | Chat-Training: Modell lernt, auf Fragen zu antworten | 1–2 h |
| `5_chatten.bat` | Chatten im Terminal | – |
| `6_chat_im_browser.bat` | Chatten im Browser unter http://localhost:8800 | – |
| `7_export_ollama.bat` | Export als GGUF, Einrichtung in Ollama | 2 min |
| `8_bewerten.bat` | Deutsch-Test: Grammatik, Wissen, Logik in Prozent | 1 min |
| `3c_training_verlaengern.bat` | Fertiges Training um eine weitere Nacht verlängern | 8 h |

**Voraussetzungen:** Python 3.12–3.14 ([python.org](https://www.python.org/downloads/), beim
Installieren „Add python.exe to PATH“ anhaken), aktueller NVIDIA-Treiber, ca. 10 GB freier
Speicher. Für den Export zusätzlich [Git](https://git-scm.com/download/win).

**Tipp:** Leg den Ordner auf ein Laufwerk mit Platz (z. B. `D:\eigenes-llm`). Während des
Trainings sollten CS2, Ollama und LM Studio geschlossen sein, weil sie Grafikspeicher brauchen.

---

## Turbo: was das Modell besser macht

### 1. Muon-Optimizer (Standard)
Für die großen Gewichtsmatrizen nutzt das Training [Muon](https://kellerjordan.github.io/posts/muon/)
statt AdamW, skaliert nach [Moonlight](https://arxiv.org/abs/2502.16982). Muon „verteilt“
jeden Lernschritt gleichmäßig auf alle Richtungen einer Matrix. Bei gleicher Rechenzeit lernt
das Modell dadurch mehr.
Zurück zum Klassiker: `trainieren --optimizer adamw`.

### 2. WSD-Lernratenplan: Training verlängern ohne Neustart
Die Lernrate bleibt lange konstant und fällt erst in den letzten 20 % ab („Abkühlen“). Kurz
davor wird `ckpt_stabil.pt` gespeichert. Willst du nach einer Nacht weitertrainieren, setzt
`3c_training_verlaengern.bat` genau dort an. Das Ergebnis ist so gut, als hättest du von
Anfang an länger trainiert. Mit einem klassischen Plan (Cosine) ginge das nicht sauber.

### 3. Lehrer-Modell (Destillation)
Kleine Modelle lernen aus sauberen, kurzen, gut formulierten Beispielen viel mehr als aus
rohem Webtext. `4a_lehrer_ueber_nacht.bat` lässt ein großes Modell in deinem Ollama (bevorzugt
Qwen) über Nacht deutsche Fragen und Antworten zu 120 Themen schreiben, in 12 Aufgabentypen
(Erklärungen, Anleitungen, Tipps, Vergleiche, kurze Texte, Rechenbeispiele …). Schlechte
Antworten werden aussortiert: englisch, chinesisch, zu lang, Verweigerungen, doppelte Fragen.
Beim Chat-Training werden diese Beispiele automatisch 3-fach gewichtet mitverwendet.

- Mit Qwen 3.5 9B auf der RTX 3050 entstehen grob 300–600 Beispiele pro Stunde. Schneller
  geht es mit einem kleineren Lehrer, z. B. `lehrer --modell qwen3.5:4b`.
- Eigene Themen: `python -m llm lehrer --thema "Counter-Strike Maps" --thema "Heidelberg"`
- Lehrer und Training nicht gleichzeitig laufen lassen, beide brauchen die Grafikkarte.

### 4. Deutsch-Test
116 Aufgaben zu Grammatik, Wissen und Logik, zum Beispiel: *„Heidelberg liegt am … Neckar / Main /
Rhein / Elbe“*. Das Modell wählt die Fortsetzung, die es für am wahrscheinlichsten hält. Das
geht auch schon beim reinen Textmodell. Der Test läuft nach jedem Training automatisch, und
`8_bewerten.bat` vergleicht mehrere Modelle nebeneinander. So siehst du in Zahlen, ob eine
Änderung etwas gebracht hat. Daneben steht immer, wie viel man durch Raten treffen würde.

### 5. Datenmix
Standard: 70 % FineWeb-2 (Webtext, natürliche Sprache) und 30 % Wikipedia (Fakten). Eigene
Mischungen gehen auch, z. B. `--quelle fineweb2:5+wikipedia:3+D:\meine_texte:2`.

---

## Wie es funktioniert

```
 deutscher Text ──► Tokenizer ──► Vortraining ──► Chat-Training ──► Export
 (FineWeb-2)        (16.000        ("nächstes      (Frage/Antwort-    (GGUF für
                     Wortteile)     Wort raten")    Beispiele)         Ollama/LM Studio)
```

1. **Daten:** Deutscher Webtext aus [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
   (gefiltert, frei nutzbar). Alternativ die deutsche Wikipedia oder eigene Textdateien.
2. **Tokenizer:** Zerlegt Text in Wortteile („Heidelberg“, „▁liegt“, „▁am“ …). Wird auf
   deinen Daten trainiert, deshalb kennt er deutsche Wörter und Umlaute gut.
3. **Vortraining:** Das Modell liest Hunderte Millionen Wortteile und lernt jeweils, den
   nächsten vorherzusagen. Dabei lernt es Grammatik, Stil und Wissen.
4. **Chat-Training (SFT):** Mit ca. 50.000 deutschen Frage-Antwort-Beispielen
   ([alpaca-gpt4-deutsch](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-deutsch))
   lernt es, wie ein Assistent zu antworten. Dazu kommen Beispiele, in denen es lernt, wer
   es ist („Ich bin Eigen-LM, Leo hat mich trainiert …“).
5. **Export:** Das Modell nutzt exakt die **Llama-Architektur**. Deshalb kann es ins
   Hugging-Face-Format und dann mit dem offiziellen llama.cpp-Konverter nach GGUF
   umgewandelt werden. Das ist das Format von Ollama und LM Studio.

### Die Architektur

Ein Decoder-only-Transformer, baugleich mit Llama 2/3:

- **RMSNorm** statt LayerNorm
- **RoPE** (rotierende Positionskodierung)
- **SwiGLU**-Feedforward
- **Grouped-Query-Attention** (weniger Speicher beim Generieren)
- geteilte Ein- und Ausgabe-Embeddings
- Flash-Attention über `scaled_dot_product_attention`, KV-Cache beim Generieren

Code: `llm/model.py`, gut 200 Zeilen.

---

## Presets

| Preset | Parameter | Kontext | Daten | Training auf RTX 3050* | Ergebnis |
|---|---|---|---|---|---|
| `test` | 2,8 Mio. | 128 | 5 Mio. Tokens | Minuten | nur zum Ausprobieren |
| `mini` | 9 Mio. | 256 | 150 Mio. | ~1 h | Rechtschreibung, einfache Sätze |
| `klein` | 32 Mio. | 512 | 600 Mio. | ~8 h | **empfohlen**: flüssige Absätze |
| `mittel` | 88 Mio. | 1024 | 2 Mrd. | mehrere Nächte | GPT-2-Klasse |

\*Schätzwerte. Die tatsächliche Geschwindigkeit zeigt das Training laufend an
(`Tok/s` und Restzeit). Mit `--stunden` legst du ein Zeitbudget fest. Die Lernrate wird dann
so geplant, dass das Training nach dieser Zeit sauber endet.

Anderes Preset: In den `.bat`-Dateien `klein` durch `mini` oder `mittel` ersetzen (und bei
`5_`–`7_` den Ordner `chat_klein` entsprechend anpassen).

---

## Alle Befehle

Mit aktivierter Umgebung (`.venv\Scripts\activate`) oder über `.venv\Scripts\python.exe`:

```powershell
python -m llm info                                   # GPU-Check
python -m llm vorbereiten --preset klein                           # 70 % Web + 30 % Wikipedia
python -m llm vorbereiten --quelle wikipedia --preset klein
python -m llm vorbereiten --quelle D:\meine_texte --preset mini   # eigene .txt-Dateien
python -m llm trainieren --preset klein --stunden 8
python -m llm trainieren --preset klein --weiter                # nach Abbruch fortsetzen
python -m llm generieren --modell arbeit\modell_klein --text "Heidelberg ist"
python -m llm trainieren --preset klein --verlaengern --stunden 16      # fertiges Training verlaengern
python -m llm bewerten --modell arbeit\modell_klein --modell arbeit\chat_klein
python -m llm lehrer --stunden 8                                  # Ollama schreibt Chat-Beispiele
python -m llm sft --basis arbeit\modell_klein --name "Eigen-LM" --ersteller "Leo"
python -m llm chat --modell arbeit\chat_klein
python -m llm web --modell arbeit\chat_klein
python -m llm export --modell arbeit\chat_klein --name eigen-lm --ollama
```

Jeder Befehl hat `--help`. Nützliche Optionen:

- `trainieren --micro-batch 8`: weniger Grafikspeicher (bei „out of memory“ passiert das
  aber auch automatisch)
- `sft --datensatz meine_daten.jsonl`: eigene Chat-Beispiele (mehrfach angeben möglich)
- `chat --system "Antworte immer kurz."`: Systemanweisung
- `export --gguf-typ q8_0`: halb so große Datei, kaum Qualitätsverlust

### Eigene Chat-Beispiele

Eine `.jsonl`-Datei, ein Beispiel pro Zeile:

```json
{"messages": [{"role": "user", "content": "Was ist Heidelberg?"}, {"role": "assistant", "content": "Eine Stadt am Neckar."}]}
```

Auch das Alpaca-Format (`instruction`/`input`/`output`) und ShareGPT (`conversations`) werden erkannt.

---

## Ergebnis in Ollama und LM Studio

Nach `7_export_ollama.bat` liegt alles in `export\eigen-lm\`:

- `eigen-lm-f16.gguf`: das Modell
- `Modelfile`: Ollama-Konfiguration inkl. Chat-Format
- `hf\`: Hugging-Face-Format

Ist Ollama installiert, wird das Modell automatisch eingerichtet:

```powershell
ollama run eigen-lm
```

**LM Studio:** Im LM-Studio-Modellordner (*My Models → Ordner öffnen*) einen Unterordner
`eigen\eigen-lm\` anlegen und die `.gguf`-Datei hineinkopieren. Das Chat-Format ist im GGUF
gespeichert und wird automatisch erkannt.

---

## Was getestet ist

`python -m pytest tests -q`: 28 Tests, unter anderem:

- **Rechnet exakt wie Llama:** Die Ausgaben stimmen mit `LlamaForCausalLM` von Hugging
  Face überein.
- **Export funktioniert durchgängig:** Das GGUF wird mit llama.cpp geladen. llama.cpp
  zerlegt den Chat-Text in genau dieselben Tokens und erzeugt denselben Text wie PyTorch.
- **Chat-Vorlagen stimmen:** Die Ollama-Vorlage (Go) und die LM-Studio-Vorlage (Jinja)
  erzeugen genau das Format, mit dem das Modell trainiert wurde.
- **Lehrer-Skript:** wird gegen einen nachgebauten Ollama-Server getestet, inklusive Filter
  und Fortsetzen nach Abbruch.
- **Technische Details:** Der KV-Cache liefert dieselben Ergebnisse wie die volle Rechnung.
  Training, Fortsetzen, Chat-Training, Umlaute und Emoji funktionieren.

Ungetestet, weil die Cloud-Umgebung keine GPU und kein Windows hat: das Training auf der
RTX 3050 selbst, das Installationsskript unter Windows und `ollama create`. Die
Geschwindigkeitsangaben oben sind deshalb Schätzungen.

---

## Probleme?

| Problem | Lösung |
|---|---|
| `info` zeigt „CUDA NEIN“ | NVIDIA-Treiber aktualisieren, `1_installieren.bat` erneut starten |
| PyTorch-Installation schlägt fehl | Python 3.13 installieren (3.14 ist sehr neu), `.venv` löschen, neu installieren |
| „Grafikspeicher voll“ | Andere Programme schließen; das Training halbiert den Batch automatisch |
| Download bricht ab | `2_daten_vorbereiten.bat` erneut starten (beginnt neu) oder `--quelle wikipedia` |
| Antworten sind Unsinn | Normal bei kurzem Training. Länger trainieren (`3b_weitertrainieren.bat`), mehr Daten, größeres Preset |
| Modell wiederholt sich | Im Chat `--wiederholung 1.2` oder niedrigere `--temperatur` |

## Projektstruktur

```
llm/
  config.py     Presets und Hyperparameter
  model.py      Transformer (Llama-Architektur)
  tokenizer.py  SentencePiece-Tokenizer + Chat-Marker
  data.py       Download, Tokenisierung, Datenlader
  train.py      Trainingsschleife (bf16, Zeitbudget, WSD, Checkpoints, OOM-Schutz)
  optim.py      Muon-Optimizer
  lehrer.py     Destillation: Ollama-Modell schreibt Chat-Beispiele
  bewertung.py  Deutsch-Test und Perplexitaet
  sft.py        Chat-Fine-Tuning
  generate.py   Sampling mit KV-Cache, Chat-Format
  web.py        Browser-Chat
  export.py     Export nach Hugging Face / GGUF / Ollama
daten/sft/identitaet.jsonl   Wer bin ich?-Beispiele (Name und Ersteller werden eingesetzt)
daten/test/deutsch_test.jsonl  116 Testaufgaben
tests/                       Tests
```
