#!/usr/bin/env python3
"""Helpers for calling a local Ollama-hosted Gemma model."""

import json
import os
from urllib import error, request


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "300"))


def chat_completion(system_prompt, user_prompt, *, temperature=0.1, model=None, timeout=None):
    """Call the Ollama chat API and return the assistant text."""
    payload = {
        "model": model or OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
        },
    }

    req = request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout or OLLAMA_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Is `ollama serve` running?"
        ) from exc

    message = data.get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected Ollama response shape: {data}")
    return content.strip()


def resolve_project_file(filename):
    """Find shared project files from inside the local subfolder."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, filename),
        os.path.join(os.path.dirname(here), filename),
        os.path.join(os.path.dirname(os.path.dirname(here)), filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))), filename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]
