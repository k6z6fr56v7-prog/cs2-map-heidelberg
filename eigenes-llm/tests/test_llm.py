"""Tests fuer die komplette Pipeline. Laufen auf der CPU in ca. 1-2 Minuten.

    python -m pytest tests -q
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.config import PRESETS, ModelConfig, TrainConfig  # noqa: E402
from llm.data import TokenData, download, tokenize  # noqa: E402
from llm.generate import encode_chat, generate, load_model, render_chat  # noqa: E402
from llm.model import GPT  # noqa: E402
from llm.sft import ChatData, normalize, prepare, tokenize_chat  # noqa: E402
from llm.tokenizer import StreamDecoder, Tokenizer, train_tokenizer  # noqa: E402
from llm.train import train  # noqa: E402
from tests.korpus import schreibe_korpus, schreibe_sft  # noqa: E402

TINY = ModelConfig(vocab_size=1000, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=176, max_seq_len=64)


@pytest.fixture(scope="session")
def work(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("arbeit")
    src = schreibe_korpus(root / "quelle", anzahl=4000)
    download(str(src.parent), root, max_chars=10**9)
    train_tokenizer([root / "tokenizer_stichprobe.txt"], root / "tokenizer", vocab_size=1000, threads=2)
    tokenize(root, root / "tokenizer.model", workers=2, val_every=20)
    return root


@pytest.fixture(scope="session")
def tok(work) -> Tokenizer:
    return Tokenizer(work / "tokenizer.model")


# ------------------------------------------------------------------ Modell


def test_param_count_and_shapes():
    m = GPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (2, 16))
    y = torch.randint(0, TINY.vocab_size, (2, 16))
    logits, loss, _ = m(x, y)
    assert logits.shape == (2, 16, TINY.vocab_size)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(1000.0)).item(), rel=0.1)
    # Embeddings sind geteilt -> lm_head zaehlt nicht doppelt
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()


@pytest.mark.parametrize("name", ["mini", "klein", "mittel"])
def test_presets_have_expected_size(name):
    cfg = replace(PRESETS[name].model, vocab_size=16000)
    with torch.device("meta"):
        n = GPT(cfg).num_params()
    expected = {"mini": 9e6, "klein": 32e6, "mittel": 88e6}[name]
    assert abs(n - expected) / expected < 0.15, n


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = GPT(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (1, 20))
    full, _, _ = m(x)
    # Prefill 12 Tokens, dann 3 auf einmal (Maskenpfad), dann einzeln
    logits, _, past = m(x[:, :12], use_cache=True)
    torch.testing.assert_close(logits[:, -1], full[:, 11], atol=1e-5, rtol=1e-4)
    logits, _, past = m(x[:, 12:15], past=past, use_cache=True)
    torch.testing.assert_close(logits[:, -1], full[:, 14], atol=1e-5, rtol=1e-4)
    for t in range(15, 20):
        logits, _, past = m(x[:, t:t + 1], past=past, use_cache=True)
        torch.testing.assert_close(logits[:, -1], full[:, t], atol=1e-5, rtol=1e-4)


def test_generate_respects_context_window():
    m = GPT(TINY).eval()
    out = generate(m, [1, 5, 6], max_new_tokens=150, temperature=1.0, seed=1)
    assert len(out) == 150  # laeuft ueber max_seq_len=64 hinaus dank gleitendem Fenster


def test_matches_huggingface_llama(tmp_path, tok):
    """Der wichtigste Test: unser Modell rechnet exakt wie LlamaForCausalLM."""
    transformers = pytest.importorskip("transformers")
    from llm.export import export_hf

    torch.manual_seed(0)
    cfg = replace(TINY, vocab_size=tok.vocab_size, tie_embeddings=True)
    m = GPT(cfg).eval()
    with torch.no_grad():  # RMSNorm-Gewichte nicht alle 1 lassen, damit der Test etwas prueft
        for name, p in m.named_parameters():
            if "norm" in name:
                p.add_(torch.randn_like(p) * 0.1)
    ck = tmp_path / "m"
    ck.mkdir()
    torch.save({"model": m.state_dict(), "model_cfg": cfg.to_dict()}, ck / "modell.pt")
    shutil.copy(tok.path, ck / "tokenizer.model")
    hf_dir = export_hf(ck, tmp_path / "hf")

    hf = transformers.AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 40))
    with torch.no_grad():
        ours, _, _ = m(x)
        theirs = hf(x).logits
    torch.testing.assert_close(ours, theirs, atol=2e-4, rtol=1e-4)


# ------------------------------------------------------------------ Tokenizer


def test_tokenizer_roundtrip_and_umlauts(tok):
    text = "Über die Brücke in Würzburg gehen Füße. Straße!\nNeue Zeile mit 1234 und Emoji 🙂"
    assert tok.decode(tok.encode(text)) == text


def test_chat_tokens_are_atomic(tok):
    ids = tok.encode("<|user|>\nHallo\n<|assistant|>\n")
    assert ids[0] == tok.user_id
    assert tok.assistant_id in ids
    assert len({tok.system_id, tok.user_id, tok.assistant_id}) == 3


def test_segment_encoding_equals_joined_encoding(tok):
    """Ohne Dummy-Prefix ist stueckweises Tokenisieren gleich dem Ganzen (wichtig fuer Ollama)."""
    msgs = [{"role": "system", "content": "Sei nett."}, {"role": "user", "content": "Wo liegt Heidelberg?"},
            {"role": "assistant", "content": "Am Neckar."}, {"role": "user", "content": "Und Köln?"}]
    segmented = encode_chat(tok, msgs)
    rendered = render_chat(msgs)
    joined = [tok.bos_id]
    for piece in rendered.split("</s>"):
        joined += tok.encode(piece) + [tok.eos_id]
    joined = joined[:-1]
    assert segmented == joined


def test_stream_decoder_never_emits_broken_chars(tok):
    text = "Grüße aus Österreich 🙂 und Übermut"
    dec = StreamDecoder(tok)
    out = "".join(dec.push(i) for i in tok.encode(text))
    assert out == text and "�" not in out


# ------------------------------------------------------------------ Daten


def test_data_files(work, tok):
    meta = json.loads((work / "daten.json").read_text())
    assert meta["train_tokens"] > 50_000 and meta["val_tokens"] > 0
    data = TokenData(work, 32)
    x, y = data.batch("train", 4, torch.device("cpu"))
    assert x.shape == (4, 32) and torch.equal(x[:, 1:], y[:, :-1])
    arr = data.train
    assert arr[0] == tok.bos_id and arr[-1] == tok.eos_id
    assert (arr == tok.bos_id).sum() == (arr == tok.eos_id).sum() > 1


# ------------------------------------------------------------------ SFT


def test_normalize_formats():
    assert normalize({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]})
    assert normalize({"conversations": [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b"}]})
    assert normalize({"instruction": "a", "input": "x", "output": "b"})[0]["content"] == "a\n\nx"
    assert normalize({"text": "nur Text"}) is None


def test_sft_masks_only_answers(tok):
    msgs = [{"role": "user", "content": "Wie heißt du?"}, {"role": "assistant", "content": "Ich heiße Test."}]
    ids, labels = tokenize_chat(tok, msgs, 200)
    answer = tok.encode("Ich heiße Test.") + [tok.eos_id]
    trained = [l for l in labels if l != -100]
    assert trained == answer
    assert ids[-len(answer):] == answer


# ------------------------------------------------------------------ Ende-zu-Ende


@pytest.fixture(scope="session")
def trained(work, tok, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("modell")
    cfg = replace(TINY, vocab_size=tok.vocab_size, max_seq_len=64)
    tc = TrainConfig(micro_batch=16, grad_accum=1, lr=3e-3, warmup_steps=20, max_steps=300, eval_every=100,
                     eval_batches=5, save_every=100, sample_every=10**9, log_every=50)
    data = TokenData(work, cfg.max_seq_len)
    train(cfg, tc, data.batch, out, work / "tokenizer.model", torch.device("cpu"))
    return out


def test_training_learns(trained):
    ck = torch.load(trained / "modell.pt", weights_only=False)
    assert ck["val_loss"] < 3.5, ck["val_loss"]  # Start liegt bei ln(1000) = 6.9
    log = (trained / "vortraining_log.csv").read_text().splitlines()
    assert len(log) > 3


def test_resume_continues(work, tok, trained):
    ck = torch.load(trained / "ckpt.pt", weights_only=False)
    cfg = ModelConfig.from_dict(ck["model_cfg"])
    tc = TrainConfig(**{**ck["train_cfg"], "max_steps": ck["step"] + 20})
    data = TokenData(work, cfg.max_seq_len)
    train(cfg, tc, data.batch, trained, trained / "tokenizer.model", torch.device("cpu"), resume=True)
    assert torch.load(trained / "ckpt.pt", weights_only=False)["step"] == ck["step"] + 20


def test_generation_after_training(trained):
    model, tok, _ = load_model(trained, torch.device("cpu"))
    ids = generate(model, tok.encode("Heidelberg ist eine Stadt in", bos=True), max_new_tokens=20,
                   temperature=0, stop_ids={tok.eos_id})
    text = tok.decode(ids)
    assert len(text) > 3


def test_sft_end_to_end(trained, tmp_path):
    tok = Tokenizer(trained / "tokenizer.model")
    jsonl = schreibe_sft(tmp_path / "sft.jsonl")
    ck = torch.load(trained / "modell.pt", weights_only=False)
    cfg = ModelConfig.from_dict(ck["model_cfg"])
    prepare([str(jsonl)], tok, cfg.max_seq_len, tmp_path / "sft.pt", identity_repeat=0, name="Testi")
    data = ChatData(tmp_path / "sft.pt", pad_id=tok.eos_id)
    x, y = data.batch("train", 4, torch.device("cpu"))
    assert x.shape == y.shape and (y == -100).any() and (y != -100).any()
    out = tmp_path / "chat"
    tc = TrainConfig(micro_batch=8, grad_accum=1, lr=3e-3, warmup_steps=10, max_steps=300, eval_every=100,
                     eval_batches=3, save_every=10**9, sample_every=10**9, log_every=50, weight_decay=0.0)
    train(cfg, tc, data.batch, out, trained / "tokenizer.model", torch.device("cpu"), init_weights=ck["model"],
          run_name="sft")
    model, tok, _ = load_model(out, torch.device("cpu"))
    ids = generate(model, encode_chat(tok, [{"role": "user", "content": "Wie viele Tage hat eine Woche?"}]),
                   max_new_tokens=30, temperature=0, stop_ids={tok.eos_id})
    assert "sieben" in tok.decode(ids)


LLAMA_CPP = Path(os.environ.get("LLAMA_CPP", "/tmp/llama.cpp"))


@pytest.mark.skipif(not (LLAMA_CPP / "convert_hf_to_gguf.py").exists(), reason="llama.cpp nicht vorhanden")
def test_gguf_export_and_llama_cpp(trained, tmp_path):
    from llm.export import export_gguf, export_hf, write_modelfile

    hf_dir = export_hf(trained, tmp_path / "hf")
    gguf = export_gguf(hf_dir, tmp_path / "m.gguf", LLAMA_CPP, "f32")
    assert gguf.stat().st_size > 100_000
    write_modelfile(gguf, tmp_path, "Sei nett.", 64)
    assert "TEMPLATE" in (tmp_path / "Modelfile").read_text()

    tokenize_bin = LLAMA_CPP / "build" / "bin" / "llama-tokenize"
    if not tokenize_bin.exists():
        pytest.skip("llama.cpp nicht gebaut")
    tok = Tokenizer(trained / "tokenizer.model")
    prompt = render_chat([{"role": "user", "content": "Wo liegt Heidelberg? Über die Brücke!"}])
    res = subprocess.run([str(tokenize_bin), "-m", str(gguf), "-p", prompt, "--ids", "--log-disable"],
                         capture_output=True, text=True, check=True)
    llama_ids = json.loads(res.stdout.strip().splitlines()[-1])
    ours = [tok.bos_id] + tok.encode(prompt)
    assert llama_ids == ours, (llama_ids, ours)

    # Gleiche Rechnung: greedy-Fortsetzung in llama.cpp == unsere greedy-Fortsetzung
    simple_bin = LLAMA_CPP / "build" / "bin" / "llama-simple"
    if not simple_bin.exists():
        pytest.skip("llama-simple nicht gebaut")
    start = "Heidelberg ist eine Stadt in"
    res = subprocess.run([str(simple_bin), "-m", str(gguf), "-n", "12", start],
                         capture_output=True, text=True, timeout=120)
    model, tok, _ = load_model(trained, torch.device("cpu"))
    ids = generate(model, tok.encode(start, bos=True), max_new_tokens=12, temperature=0, repetition_penalty=1.0)
    assert start + tok.decode(ids) in res.stdout, (tok.decode(ids), res.stdout[-500:])


GO_RENDER = r'''
package main

import (
	"encoding/json"
	"os"
	"text/template"
)

type Message struct {
	Role    string
	Content string
}

func main() {
	var in struct {
		Template string
		Messages []Message
	}
	json.NewDecoder(os.Stdin).Decode(&in)
	t := template.Must(template.New("t").Parse(in.Template))
	t.Execute(os.Stdout, map[string]any{"Messages": in.Messages})
}
'''


@pytest.mark.skipif(shutil.which("go") is None, reason="Go nicht installiert")
def test_ollama_template_matches_training_format(tmp_path):
    """Ollama rendert Templates mit Go text/template - wir pruefen das Ergebnis gegen render_chat."""
    from llm.export import OLLAMA_TEMPLATE

    (tmp_path / "main.go").write_text(GO_RENDER)
    msgs = [{"role": "system", "content": "Sei nett."}, {"role": "user", "content": "Wo liegt Heidelberg?"},
            {"role": "assistant", "content": "Am Neckar."}, {"role": "user", "content": "Und Köln?"}]
    payload = json.dumps({"Template": OLLAMA_TEMPLATE,
                          "Messages": [{"Role": m["role"], "Content": m["content"]} for m in msgs]})
    res = subprocess.run(["go", "run", "main.go"], cwd=tmp_path, input=payload, capture_output=True, text=True,
                         check=True, timeout=300, env={**os.environ, "GOFLAGS": "-mod=mod", "GO111MODULE": "off"})
    assert res.stdout == render_chat(msgs)


def test_jinja_template_matches_training_format():
    jinja2 = pytest.importorskip("jinja2")
    from llm.export import CHAT_TEMPLATE

    msgs = [{"role": "system", "content": "Sei nett."}, {"role": "user", "content": "Wo liegt Heidelberg?"},
            {"role": "assistant", "content": "Am Neckar."}, {"role": "user", "content": "Und Köln?"}]
    assert jinja2.Template(CHAT_TEMPLATE).render(messages=msgs) == render_chat(msgs)
