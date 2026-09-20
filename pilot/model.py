"""One-token HF inference and decoder-output patches on CPU, CUDA or MPS.

Geometry never backpropagates through the language model. Every path checkpoint
is a fresh forward pass with one last-prompt-token replacement; no KV cache is
carried from one checkpoint to another.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import time

import numpy as np

from .data import LABELS, prompt
from .storage import digest, versions


class Runtime:
    def __init__(self, cfg, pinned_revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.cfg = cfg
        m = cfg["model"]
        torch.set_num_threads(m["threads"])
        device = m["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable; choose cpu explicitly")
        self.device = torch.device(device)
        dtype_name = m["dtype"]
        if dtype_name == "auto":
            dtype_name = "float32" if device == "cpu" else "float16" if device == "mps" else "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        if device == "cpu" and dtype_name == "float16":
            raise ValueError("Use float32 or bfloat16 on CPU, not float16")
        if device.startswith("cuda") and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16; select float16 or float32 before starting the run")
        kwargs = dict(revision=pinned_revision or m["revision"], local_files_only=m["local_files_only"],
                      trust_remote_code=False, cache_dir=m["cache_dir"])
        self.tokenizer = AutoTokenizer.from_pretrained(m["name"], **kwargs)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            m["name"], torch_dtype=getattr(torch, dtype_name),
            use_safetensors=True, attn_implementation="eager", **kwargs,
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        core = getattr(self.model, "model", None)
        self.layers = getattr(core, "layers", None)
        if self.layers is None:
            raise ValueError("This adapter supports HF Llama/Qwen-style model.model.layers decoder models")
        if max(cfg["fit"]["layers"]) >= len(self.layers):
            raise ValueError(f"Requested layer exceeds the model's {len(self.layers)} decoder layers")
        self.token_ids = self._answer_tokens()
        actual_revision = getattr(self.model.config, "_commit_hash", None)
        local_config = Path(m["name"]) / "config.json"
        self.identity = {
            "name": m["name"], "revision": actual_revision or kwargs["revision"],
            "local_config_hash": digest(local_config.read_text()) if local_config.exists() else None,
            "dtype": dtype_name, "chat_template": m["chat_template"],
            "token_ids": self.token_ids, "layers": len(self.layers),
        }
        self.metadata = {**self.identity, "device": str(self.device), "versions": versions()}
        self.last_logits_only = "logits_to_keep" in inspect.signature(self.model.forward).parameters

    def _answer_tokens(self):
        seen = set()
        result = []
        for label in LABELS[:-1]:
            spellings = [label, "grey"] if label == "gray" else [label]
            ids = set()
            for spelling in spellings:
                for variant in {spelling, spelling.title(), " " + spelling, " " + spelling.title()}:
                    encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                    if len(encoded) == 1:
                        ids.add(encoded[0])
            if not ids or ids & seen:
                raise ValueError(f"Missing or overlapping single-token readout for {label}")
            seen.update(ids)
            result.append(sorted(ids))
        return result

    def _texts(self, rows, extended):
        texts = [prompt(row, extended) for row in rows]
        if self.cfg["model"]["chat_template"]:
            if not self.tokenizer.chat_template:
                raise ValueError("chat_template requested for a tokenizer without one")
            texts = [self.tokenizer.apply_chat_template([{"role": "user", "content": t}],
                       tokenize=False, add_generation_prompt=True) for t in texts]
        return texts

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self.torch.mps.synchronize()

    def forward(self, rows, capture=(), patch_layer=None, replacements=None, extended=None, capture_full=False):
        if not rows:
            raise ValueError("Empty forward batch")
        torch = self.torch
        extended = self.cfg["readout"]["extended"] if extended is None else extended
        texts = self._texts(rows, extended)
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=False, add_special_tokens=not self.cfg["model"]["chat_template"])
        if tokens["input_ids"].shape[1] > self.cfg["model"]["max_length"]:
            raise ValueError("Prompt exceeds max_length; refusing to truncate the intervention site")
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        positions = tokens["attention_mask"].long().cumsum(-1) - 1
        positions.masked_fill_(tokens["attention_mask"] == 0, 0)
        tokens["position_ids"] = positions
        captures = {}
        handles = []

        def make_hook(index):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if index in capture:
                    captures[index] = hidden[:, -1].detach().float().cpu().numpy().copy()
                if index != patch_layer:
                    return None
                replacement = torch.as_tensor(replacements, device=hidden.device, dtype=hidden.dtype)
                if replacement.shape != hidden[:, -1].shape:
                    raise ValueError("Replacement shape differs from last-token hidden state")
                updated = hidden.clone()
                updated[:, -1] = replacement
                return (updated,) + output[1:] if isinstance(output, tuple) else updated
            return hook

        try:
            for layer in set(capture) | ({patch_layer} if patch_layer is not None else set()):
                handles.append(self.layers[layer].register_forward_hook(make_hook(layer)))
            self.sync()
            begin = time.perf_counter()
            with torch.no_grad():
                extra = {"logits_to_keep": 1} if self.last_logits_only else {}
                output = self.model(**tokens, use_cache=False, return_dict=True, **extra)
                logits = output.logits[:, -1].float()
                probabilities = torch.softmax(logits, dim=-1)
                named = torch.stack([probabilities[:, ids].sum(-1) for ids in self.token_ids], dim=-1)
                other = (1 - named.sum(-1, keepdim=True)).clamp_min(0)
                result = torch.cat([named, other], dim=-1).cpu().numpy()
                self.last_full_probabilities = probabilities.cpu().numpy() if capture_full else None
                top = logits.argmax(-1).cpu().numpy()
            self.sync()
            elapsed = time.perf_counter() - begin
        finally:
            for handle in handles:
                handle.remove()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite model outputs; try float32 or smaller batches")
        return result, captures, top, elapsed

    def batched(self, rows, capture=(), patch_layer=None, replacements=None, extended=None, capture_full=False):
        size = self.cfg["model"]["batch_size"]
        probs, tops, elapsed = [], [], 0.0
        full = []
        activations = {layer: [] for layer in capture}
        for start in range(0, len(rows), size):
            stop = start + size
            patch = None if replacements is None else replacements[start:stop]
            p, a, top, seconds = self.forward(rows[start:stop], capture, patch_layer, patch, extended, capture_full)
            if capture_full:
                full.append(self.last_full_probabilities)
            probs.append(p)
            tops.append(top)
            elapsed += seconds
            for layer in capture:
                activations[layer].append(a[layer])
        self.last_full_probabilities = np.concatenate(full) if capture_full else None
        return np.concatenate(probs), {l: np.concatenate(a) for l, a in activations.items()}, np.concatenate(tops), elapsed
