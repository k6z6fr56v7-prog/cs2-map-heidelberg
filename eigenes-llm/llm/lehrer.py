"""Lehrer-Modell (Destillation): ein grosses lokales Modell in Ollama schreibt
Trainingsbeispiele fuer dein kleines Modell.

Kleine Modelle lernen aus sauberen, kurzen, gut formulierten Beispielen viel
mehr als aus rohem Webtext. Deshalb laesst man ein grosses Modell (z.B. Qwen)
Fragen und Antworten auf Deutsch schreiben und trainiert damit das Chat-Fine-Tuning.

Laeuft ueber die HTTP-Schnittstelle von Ollama (http://localhost:11434).
Kann jederzeit mit Strg+C beendet und spaeter fortgesetzt werden.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .system import wach_bleiben

OLLAMA_URL = "http://localhost:11434"

THEMEN = [
    # Wissen
    "Photosynthese", "Schwarze Löcher", "das Immunsystem", "Vulkane", "Klimawandel", "Elektrizität",
    "das Sonnensystem", "Magnetismus", "Evolution", "Wetter und Wolken", "Erdbeben", "das menschliche Herz",
    "Bienen", "Wale", "Regenwald", "Wasserkreislauf", "Atome", "Licht und Farben", "Schall", "Impfungen",
    "Schlaf", "Ernährung", "Muskeln und Training", "das Gehirn", "Antibiotika", "Ebbe und Flut",
    # Geschichte und Gesellschaft
    "die Römer in Germanien", "das Mittelalter", "Martin Luther", "die Industrialisierung", "die Weimarer Republik",
    "die deutsche Wiedervereinigung", "die Europäische Union", "das Grundgesetz", "Wahlen in Deutschland",
    "die Französische Revolution", "die Mondlandung", "die Erfindung des Buchdrucks", "das alte Ägypten",
    "die Hanse", "Burgen und Ritter", "die Geschichte des Internets",
    # Geografie
    "Heidelberg", "Hamburg", "Berlin", "München", "der Rhein", "die Alpen", "die Nordsee", "der Schwarzwald",
    "Österreich", "die Schweiz", "Italien", "Japan", "Kanada", "die Sahara", "der Amazonas", "Island",
    # Alltag
    "Kochen von Nudeln", "Brot backen", "Geld sparen", "einen Lebenslauf schreiben", "Zeitmanagement",
    "ein Vorstellungsgespräch", "Umzug planen", "Fahrrad reparieren", "Pflanzen pflegen", "Haustiere",
    "Stress abbauen", "gesund frühstücken", "Mülltrennung", "Energie sparen im Haushalt", "Reisen mit der Bahn",
    "Freundschaft", "Konflikte lösen", "Lernen für Prüfungen", "Studium an der Uni", "Mietvertrag",
    # Technik und Computer
    "Computer und Prozessoren", "Grafikkarten", "Arbeitsspeicher", "Programmieren lernen", "Python",
    "künstliche Intelligenz", "Passwörter und Sicherheit", "WLAN", "Smartphones", "Backups",
    "Betriebssysteme", "Videospiele", "Counter-Strike", "Elektroautos", "Solaranlagen", "3D-Druck",
    # Sprache und Kultur
    "Goethe", "Schiller", "die Brüder Grimm", "deutsche Grammatik", "Kommasetzung", "Redewendungen",
    "Weihnachten in Deutschland", "Oktoberfest", "Fußball", "klassische Musik", "Malerei", "Kino",
    # Mathe und Logik
    "Prozentrechnung", "Brüche", "Dreisatz", "Wahrscheinlichkeit", "Primzahlen", "geometrische Formen",
]

AUFGABEN = [
    "eine einfache Wissensfrage", "eine Frage, die mit 'Warum' beginnt", "eine Frage, die mit 'Wie' beginnt",
    "die Bitte um eine kurze Erklärung für ein Kind", "die Bitte um drei praktische Tipps",
    "die Bitte, zwei Dinge zu vergleichen", "die Bitte um eine kurze Zusammenfassung",
    "die Bitte, einen kurzen Text zu schreiben (z.B. eine Nachricht, eine Einladung oder ein Gedicht)",
    "eine Frage nach einer Definition", "die Bitte um eine Schritt-für-Schritt-Anleitung",
    "eine Alltagsfrage, wie man sie einem Freund stellen würde", "eine Frage mit einem kleinen Rechenbeispiel",
]

SYSTEM_LEHRER = (
    "Du erzeugst Trainingsdaten für einen kleinen deutschen KI-Assistenten. "
    "Schreibe ausschließlich korrektes, natürliches Deutsch. Antworten sind sachlich richtig, "
    "freundlich, klar gegliedert und kurz (höchstens 120 Wörter). Keine Emojis, kein Markdown außer "
    "einfachen Aufzählungen mit '-'."
)

SCHEMA = {
    "type": "object",
    "properties": {"frage": {"type": "string"}, "antwort": {"type": "string"}},
    "required": ["frage", "antwort"],
}


def ollama_request(path: str, payload: Optional[dict] = None, url: str = OLLAMA_URL, timeout: float = 600):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def list_models(url: str = OLLAMA_URL) -> list[str]:
    return [m["name"] for m in ollama_request("/api/tags", url=url, timeout=10).get("models", [])]


def pick_model(wanted: Optional[str], url: str = OLLAMA_URL) -> str:
    try:
        models = list_models(url)
    except (urllib.error.URLError, OSError):
        raise SystemExit("Ollama ist nicht erreichbar. Laeuft Ollama? (Startmenue -> Ollama)")
    if not models:
        raise SystemExit("In Ollama ist noch kein Modell installiert. Z.B.: ollama pull qwen3.5:9b")
    if wanted:
        if wanted in models or any(m.startswith(wanted) for m in models):
            return wanted
        raise SystemExit(f"Modell '{wanted}' nicht gefunden. Installiert: {', '.join(models)}")
    # bevorzugt ein Qwen, sonst das erste Modell, aber nicht unser eigenes
    for pref in ("qwen3.5", "qwen3", "qwen2.5", "llama3", "gemma", "mistral"):
        for m in models:
            if m.startswith(pref):
                return m
    others = [m for m in models if not m.startswith("eigen-lm")]
    return (others or models)[0]


CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
DEUTSCH = re.compile(r"\b(der|die|das|und|ist|nicht|ein|eine|mit|für|auf|sich|auch|es|zu|den|von)\b", re.I)
ENGLISCH = re.compile(r"\b(the|and|is|are|with|for|this|that|you|of)\b", re.I)


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    return text.strip()


def is_good(frage: str, antwort: str) -> bool:
    if not (8 <= len(frage) <= 400 and 20 <= len(antwort) <= 1500):
        return False
    both = frage + " " + antwort
    if CJK.search(both):
        return False
    de, en = len(DEUTSCH.findall(both)), len(ENGLISCH.findall(both))
    if de < 3 or en > de / 3:
        return False
    lower = antwort.lower()
    if any(s in lower for s in ("als ki-sprachmodell", "als sprachmodell kann ich", "i'm sorry", "ich kann dir dabei nicht")):
        return False
    return True


def make_prompt(thema: str, aufgabe: str, rng: random.Random) -> str:
    stil = rng.choice(["locker", "sachlich", "neugierig", "kurz angebunden", "höflich"])
    return (f"Erfinde {aufgabe} zum Thema '{thema}', so wie sie ein Nutzer ({stil}) einem KI-Assistenten "
            f"stellen würde. Schreibe dann die ideale Antwort des Assistenten. "
            f"Gib JSON zurück: {{\"frage\": \"...\", \"antwort\": \"...\"}}")


def generate_one(model: str, prompt: str, url: str, temperature: float, seed: int) -> Optional[dict]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_LEHRER}, {"role": "user", "content": prompt}],
        "format": SCHEMA,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "seed": seed, "num_predict": 700},
    }
    try:
        res = ollama_request("/api/chat", payload, url=url)
    except urllib.error.HTTPError as e:
        if e.code == 400 and "think" in payload:  # aeltere Ollama-Versionen kennen 'think' nicht
            payload.pop("think")
            res = ollama_request("/api/chat", payload, url=url)
        else:
            raise
    content = clean(res.get("message", {}).get("content", ""))
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    frage, antwort = clean(str(obj.get("frage", ""))), clean(str(obj.get("antwort", "")))
    if not is_good(frage, antwort):
        return None
    return {"messages": [{"role": "user", "content": frage}, {"role": "assistant", "content": antwort}],
            "quelle": model}


def run(out_path: Path, model: Optional[str] = None, anzahl: Optional[int] = None,
        hours: Optional[float] = None, url: str = OLLAMA_URL, temperature: float = 0.9,
        extra_themen: Optional[list[str]] = None) -> int:
    model = pick_model(model, url)
    themen = THEMEN + (extra_themen or [])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    fragen: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done_ids.add(row.get("id", ""))
            fragen.add(row["messages"][0]["content"].lower())
    jobs = [(t, a, k) for t in themen for a in AUFGABEN for k in range(3)]
    random.Random(42).shuffle(jobs)
    jobs = [j for j in jobs if f"{j[0]}|{j[1]}|{j[2]}" not in done_ids]

    print(f"  Lehrer-Modell: {model}")
    print(f"  Ausgabe:       {out_path} ({len(done_ids):,} vorhanden, {len(jobs):,} Aufgaben offen)")
    print("  Beenden mit Strg+C - beim naechsten Start geht es dort weiter.\n")
    t0 = time.time()
    neu = verworfen = 0
    try:
        with wach_bleiben(), out_path.open("a", encoding="utf-8") as f:
            for i, (thema, aufgabe, k) in enumerate(jobs):
                if anzahl and neu >= anzahl:
                    break
                if hours and time.time() - t0 > hours * 3600:
                    break
                job_id = f"{thema}|{aufgabe}|{k}"
                rng = random.Random(job_id)
                try:
                    ex = generate_one(model, make_prompt(thema, aufgabe, rng), url, temperature, seed=rng.randint(0, 2**31))
                except (urllib.error.URLError, OSError, TimeoutError) as e:
                    print(f"\n  Verbindungsproblem ({e}) - warte 10 s ...")
                    time.sleep(10)
                    continue
                if ex is None or ex["messages"][0]["content"].lower() in fragen:
                    verworfen += 1
                else:
                    ex["id"] = job_id
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    f.flush()
                    fragen.add(ex["messages"][0]["content"].lower())
                    neu += 1
                rate = neu / max(time.time() - t0, 1e-6) * 3600
                print(f"\r  {neu:,} neue Beispiele ({verworfen} verworfen), ~{rate:,.0f} pro Stunde   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n  Abgebrochen.")
    print(f"\n  Fertig: {neu:,} neue Beispiele in {out_path}")
    return neu
