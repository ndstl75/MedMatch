"""Remote Google GenAI backend."""

import os
import time

from google import genai
from google.genai import errors, types

from medmatch.core.scorer import coerce_output_object, parse_json_response
from medmatch.llm.base import LLMBackend


def load_google_api_key():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key and os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("GOOGLE_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    break
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your environment or .env.")
    return api_key


def build_response_schema(expected_keys):
    return {
        "type": "object",
        "properties": {
            key: {"type": ["string", "number", "integer", "null"]} for key in expected_keys
        },
        "required": expected_keys,
        "additionalProperties": False,
    }


class RemoteGemmaBackend(LLMBackend):
    def __init__(self, model=None, temperature=None, retries=None, retry_delay=None):
        self.model = model or os.environ.get("GOOGLE_MODEL_NAME", "gemma-3-27b-it")
        self.temperature = float(
            os.environ.get("MEDMATCH_TEMPERATURE", "0.1") if temperature is None else temperature
        )
        self.retries = int(os.environ.get("MEDMATCH_MAX_RETRIES", "2") if retries is None else retries)
        self.retry_delay = float(
            os.environ.get("MEDMATCH_RETRY_DELAY", "5") if retry_delay is None else retry_delay
        )
        self.client = genai.Client(api_key=load_google_api_key())

    def generate_text(self, system_prompt, user_prompt, *, temperature=None):
        temp = self.temperature if temperature is None else temperature
        for attempt in range(self.retries):
            try:
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temp,
                        ),
                    )
                except errors.ClientError as exc:
                    if "Developer instruction is not enabled" not in str(exc):
                        raise
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=f"{system_prompt}\n\n{user_prompt}",
                        config=types.GenerateContentConfig(temperature=temp),
                    )
                return (response.text or "").strip()
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(self.retry_delay)

    def generate_json(self, system_prompt, user_prompt, expected_keys, *, temperature=None):
        temp = self.temperature if temperature is None else temperature
        response_schema = build_response_schema(expected_keys)

        for attempt in range(self.retries):
            raw_text = ""
            try:
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temp,
                            response_mime_type="application/json",
                            response_json_schema=response_schema,
                        ),
                    )
                except errors.ClientError as exc:
                    message = str(exc)
                    if (
                        "JSON mode is not enabled" not in message
                        and "Developer instruction is not enabled" not in message
                    ):
                        raise
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=f"{system_prompt}\n\n{user_prompt}",
                        config=types.GenerateContentConfig(temperature=temp),
                    )
                raw_text = (response.text or "").strip()
                parsed = parse_json_response(raw_text)
                return coerce_output_object(parsed, expected_keys), raw_text
            except Exception:
                if attempt == self.retries - 1:
                    return {key: "" for key in expected_keys}, raw_text
                time.sleep(self.retry_delay)

