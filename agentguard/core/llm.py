"""Pluggable text generator.

The demo must run even with no model weights present, so `TemplateLLM` is a
dependency-free fallback that produces the explanation text from the guard
verdict itself. `LlamaCppLLM` uses a local GGUF model when one is available,
which keeps generation on the mainframe: no data leaves the platform.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Protocol


class LLM(Protocol):
    name: str

    def complete(self, prompt: str, max_tokens: int = 220) -> str: ...


class TemplateLLM:
    """Deterministic fallback. No weights, no network, always available."""

    name = "template"

    def complete(self, prompt: str, max_tokens: int = 220) -> str:
        return ""  # Callers fall back to their own templated text.


class LlamaCppLLM:
    """Local GGUF model via the llama.cpp CLI binary."""

    name = "llama.cpp"

    def __init__(self, binary: str, model_path: str, threads: int = 3):
        self.binary = binary
        self.model_path = model_path
        self.threads = threads

    @classmethod
    def discover(cls) -> "LlamaCppLLM | None":
        binary = os.environ.get("LLAMA_BIN") or shutil.which("llama-cli")
        model = os.environ.get("LLAMA_MODEL")
        if binary and model and os.path.exists(binary) and os.path.exists(model):
            return cls(binary, model)
        return None

    def complete(self, prompt: str, max_tokens: int = 220) -> str:
        cmd = [self.binary, "-m", self.model_path, "-p", prompt,
               "-n", str(max_tokens), "-t", str(self.threads),
               "--temp", "0.3", "--no-display-prompt", "-no-cnv"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except (subprocess.TimeoutExpired, OSError):
            return ""
        return out.stdout.strip()


def get_llm() -> LLM:
    """Prefer a local model; fall back to templates so the demo never breaks."""
    return LlamaCppLLM.discover() or TemplateLLM()
