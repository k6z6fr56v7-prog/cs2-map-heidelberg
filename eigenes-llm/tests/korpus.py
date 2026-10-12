"""Erzeugt einen kleinen deutschen Testkorpus (fuer Tests ohne Internet).

Die Saetze sind schablonenhaft, aber abwechslungsreich genug, dass man sehen
kann, ob das Modell lernt. Fuer echtes Training kommen die Daten aus FineWeb-2
oder Wikipedia.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

STAEDTE = [("Heidelberg", "Neckar", "Baden-Württemberg"), ("Hamburg", "Elbe", "Hamburg"),
           ("Köln", "Rhein", "Nordrhein-Westfalen"), ("Dresden", "Elbe", "Sachsen"),
           ("Frankfurt", "Main", "Hessen"), ("Würzburg", "Main", "Bayern"),
           ("Passau", "Donau", "Bayern"), ("Bremen", "Weser", "Bremen"),
           ("Mainz", "Rhein", "Rheinland-Pfalz"), ("Tübingen", "Neckar", "Baden-Württemberg")]
TIERE = ["Der Fuchs", "Die Katze", "Der Hund", "Die Eule", "Das Pferd", "Der Igel", "Die Amsel"]
ORTE = ["im Wald", "auf der Wiese", "am Fluss", "im Garten", "hinter dem Haus", "auf dem Berg"]
TATEN = ["schläft", "sucht Futter", "läuft schnell", "wartet geduldig", "spielt", "ruht sich aus"]
ZEITEN = ["Am Morgen", "Am Abend", "In der Nacht", "Im Sommer", "Im Winter", "Nach dem Regen"]
FRAGEN = [("Was ist die Hauptstadt von Deutschland?", "Die Hauptstadt von Deutschland ist Berlin."),
          ("Wie viele Tage hat eine Woche?", "Eine Woche hat sieben Tage."),
          ("Welche Farbe hat der Himmel?", "Bei klarem Wetter ist der Himmel blau."),
          ("Was ist zwei plus zwei?", "Zwei plus zwei ist vier."),
          ("Wie heißt der längste Fluss Deutschlands?", "Der längste Fluss, der durch Deutschland fließt, ist der Rhein.")]


def dokument(rng: random.Random) -> str:
    art = rng.random()
    if art < 0.5:
        stadt, fluss, land = rng.choice(STAEDTE)
        saetze = [f"{stadt} ist eine Stadt in {land}.", f"Die Stadt liegt am {fluss}.",
                  f"Viele Menschen besuchen {stadt} wegen der schönen Altstadt.",
                  f"Im Jahr {rng.randint(1100, 1900)} wurde in {stadt} eine Brücke über den {fluss} gebaut.",
                  f"Heute leben in {stadt} etwa {rng.randint(50, 900)} tausend Menschen."]
        rng.shuffle(saetze[2:])
        return " ".join(saetze) + "\n\n" + f"Der {fluss} ist ein wichtiger Fluss für {stadt}."
    if art < 0.85:
        teile = []
        for _ in range(rng.randint(3, 7)):
            teile.append(f"{rng.choice(ZEITEN)} {rng.choice(TATEN)} {rng.choice(TIERE).lower()} {rng.choice(ORTE)}."
                         .replace(" der ", " der ").capitalize())
        return " ".join(teile)
    frage, antwort = rng.choice(FRAGEN)
    return f"Frage: {frage}\nAntwort: {antwort}"


def schreibe_korpus(ordner: Path, anzahl: int = 3000, seed: int = 0) -> Path:
    rng = random.Random(seed)
    ordner.mkdir(parents=True, exist_ok=True)
    p = ordner / "korpus.txt"
    p.write_text("\n\n".join(dokument(rng) for _ in range(anzahl)), encoding="utf-8")
    return p


def schreibe_sft(pfad: Path) -> Path:
    with pfad.open("w", encoding="utf-8") as f:
        for frage, antwort in FRAGEN:
            f.write(json.dumps({"messages": [{"role": "user", "content": frage},
                                             {"role": "assistant", "content": antwort}]}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"instruction": "Nenne eine Stadt am Neckar.", "input": "",
                            "output": "Heidelberg liegt am Neckar."}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"conversations": [{"from": "human", "value": "Wo liegt Hamburg?"},
                                              {"from": "gpt", "value": "Hamburg liegt an der Elbe."}]},
                           ensure_ascii=False) + "\n")
    return pfad
