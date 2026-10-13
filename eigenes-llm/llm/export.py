"""Export: eigenes Modell -> Hugging Face -> GGUF -> Ollama / LM Studio.

1. `export_hf` schreibt einen Ordner im Format von `LlamaForCausalLM`
   (config.json, model.safetensors, tokenizer.model, tokenizer_config.json).
2. `export_gguf` ruft das offizielle Konvertierungsskript von llama.cpp auf.
3. `write_modelfile` erzeugt die Ollama-Modelfile mit passendem Chat-Template.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch

from .config import ModelConfig
from .generate import find_checkpoint, find_tokenizer
from .tokenizer import ASSISTANT, SYSTEM, USER, Tokenizer

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"

# Jinja-Template (landet im GGUF, LM Studio nutzt es automatisch).
# BOS fuegt der Tokenizer selbst hinzu (add_bos_token).
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}" + SYSTEM + "\n{{ message['content'] | trim }}\n"
    "{% elif message['role'] == 'user' %}" + USER + "\n{{ message['content'] | trim }}\n" + ASSISTANT + "\n"
    "{% elif message['role'] == 'assistant' %}{{ message['content'] | trim }}</s>"
    "{% endif %}{% endfor %}"
)

# Go-Template fuer Ollama (gleiches Format)
OLLAMA_TEMPLATE = (
    "{{- range .Messages }}"
    "{{ if eq .Role \"system\" }}" + SYSTEM + "\n{{ .Content }}\n"
    "{{ else if eq .Role \"user\" }}" + USER + "\n{{ .Content }}\n" + ASSISTANT + "\n"
    "{{ else if eq .Role \"assistant\" }}{{ .Content }}</s>"
    "{{ end }}"
    "{{ end }}"
)


def hf_config(cfg: ModelConfig) -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.dim,
        "intermediate_size": cfg.ffn_dim,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "rope_scaling": None,
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": cfg.tie_embeddings,
        "bos_token_id": cfg.bos_id,
        "eos_token_id": cfg.eos_id,
        "torch_dtype": "float32",
        "initializer_range": 0.02,
        "use_cache": True,
    }


def export_hf(model_path: Path, out_dir: Path, name: str = "eigen-lm") -> Path:
    from safetensors.torch import save_file

    ckpt_path = find_checkpoint(model_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_cfg"])
    state = ckpt["model"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    for k, v in state.items():
        if cfg.tie_embeddings and k == "lm_head.weight":
            continue  # geteilt mit embed_tokens, HF erwartet es dann nicht separat
        tensors[k] = v.detach().to(torch.float32).contiguous()
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    (out_dir / "config.json").write_text(json.dumps(hf_config(cfg), indent=2), encoding="utf-8")
    (out_dir / "generation_config.json").write_text(json.dumps({
        "bos_token_id": cfg.bos_id, "eos_token_id": cfg.eos_id, "do_sample": True,
        "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1,
    }, indent=2), encoding="utf-8")

    tok_path = find_tokenizer(ckpt_path)
    shutil.copy(tok_path, out_dir / "tokenizer.model")
    # Die Chat-Marker als Sonder-Tokens ausweisen, sonst zerlegt llama.cpp sie in Buchstaben
    tok = Tokenizer(tok_path)
    added = {
        str(tok.sp.piece_to_id(piece)): {"content": piece, "special": True, "lstrip": False, "rstrip": False,
                                         "normalized": False, "single_word": False}
        for piece in ("<unk>", "<s>", "</s>", SYSTEM, USER, ASSISTANT)
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": added,
        "tokenizer_class": "LlamaTokenizer",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "add_bos_token": True,
        "add_eos_token": False,
        "add_prefix_space": False,
        "legacy": False,
        "model_max_length": cfg.max_seq_len,
        "chat_template": CHAT_TEMPLATE,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
        "additional_special_tokens": [SYSTEM, USER, ASSISTANT],
    }, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        f"# {name}\n\nSelbst trainiertes deutsches Sprachmodell (Llama-Architektur, "
        f"{sum(t.numel() for t in tensors.values()) / 1e6:.1f} Mio. Parameter).\n", encoding="utf-8")
    return out_dir


def ensure_llama_cpp(path: Optional[Path]) -> Path:
    if path and (path / "convert_hf_to_gguf.py").exists():
        return path
    target = path or Path("werkzeuge") / "llama.cpp"
    if (target / "convert_hf_to_gguf.py").exists():
        return target
    if not shutil.which("git"):
        raise SystemExit("Fuer den Export wird Git gebraucht: https://git-scm.com/download/win installieren, "
                         "danach das Fenster neu oeffnen und den Export erneut starten.")
    print(f"  Lade llama.cpp nach {target} (nur das Konvertierungsskript wird gebraucht)...")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(target)], check=True)
    return target


def export_gguf(hf_dir: Path, out_file: Path, llama_cpp: Optional[Path] = None, outtype: str = "f16") -> Path:
    llama_cpp = ensure_llama_cpp(llama_cpp)
    script = llama_cpp / "convert_hf_to_gguf.py"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), str(hf_dir), "--outfile", str(out_file), "--outtype", outtype]
    print("  " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        # Bewusst NICHT die requirements-Datei von llama.cpp empfehlen: die wuerde
        # PyTorch durch eine CPU-Version ersetzen.
        raise SystemExit("GGUF-Konvertierung fehlgeschlagen. Haeufigste Ursache: fehlende Pakete -> "
                         "pip install transformers protobuf gguf")
    return out_file


def write_modelfile(gguf_file: Path, out_dir: Path, system: Optional[str] = None,
                    max_seq_len: int = 512) -> Path:
    lines = [
        f"FROM ./{gguf_file.name}",
        "",
        f'TEMPLATE """{OLLAMA_TEMPLATE}"""',
        "",
        f"PARAMETER num_ctx {max_seq_len}",
        'PARAMETER stop "</s>"',
        f'PARAMETER stop "{USER}"',
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.9",
        "PARAMETER repeat_penalty 1.1",
    ]
    if system:
        lines += ["", f'SYSTEM """{system}"""']
    path = out_dir / "Modelfile"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def ollama_create(name: str, modelfile: Path) -> bool:
    exe = shutil.which("ollama")
    if not exe:
        return False
    print(f"  ollama create {name} ...")
    res = subprocess.run([exe, "create", name, "-f", modelfile.name], cwd=modelfile.parent)
    return res.returncode == 0
