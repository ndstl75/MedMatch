"""Local Ollama backend."""

import json
import os
import time
from urllib import error, request

from medmatch.core.scorer import coerce_output_object, parse_json_response
from medmatch.llm.base import LLMBackend


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "300"))
OLLAMA_THINK = os.environ.get("OLLAMA_THINK")


def chat_completion(system_prompt, user_prompt, *, temperature=0.1, model=None, timeout=None):
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
    if OLLAMA_THINK is not None:
        payload["think"] = OLLAMA_THINK.strip().lower() not in {"0", "false", "no", "off"}
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


class LocalOllamaBackend(LLMBackend):
    def __init__(self, model=None, temperature=None, retries=None, retry_delay=None):
        self.model = model or OLLAMA_MODEL
        self.temperature = float(
            os.environ.get("MEDMATCH_TEMPERATURE", "0.1") if temperature is None else temperature
        )
        self.retries = int(os.environ.get("MEDMATCH_MAX_RETRIES", "3") if retries is None else retries)
        self.retry_delay = float(
            os.environ.get("MEDMATCH_RETRY_DELAY", "5") if retry_delay is None else retry_delay
        )

    def generate_text(self, system_prompt, user_prompt, *, temperature=None):
        temp = self.temperature if temperature is None else temperature
        for attempt in range(self.retries):
            try:
                return chat_completion(system_prompt, user_prompt, temperature=temp, model=self.model)
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(self.retry_delay)

    def generate_json(self, system_prompt, user_prompt, expected_keys, *, temperature=None):
        text = self.generate_text(system_prompt, user_prompt, temperature=temperature)
        parsed = parse_json_response(text)
        return coerce_output_object(parsed, expected_keys), text
