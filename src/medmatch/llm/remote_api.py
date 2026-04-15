"""OpenAI- and Azure-backed remote LLM adapters."""

import os
import time

from medmatch.core.scorer import coerce_output_object, parse_json_response
from medmatch.llm.base import LLMBackend

try:
    from openai import AzureOpenAI, OpenAI

    OPENAI_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    AzureOpenAI = None
    OpenAI = None
    OPENAI_SDK_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    AZURE_IDENTITY_AVAILABLE = True
except ImportError:  # pragma: no cover
    DefaultAzureCredential = None
    get_bearer_token_provider = None
    AZURE_IDENTITY_AVAILABLE = False


def _read_dotenv_value(name):
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(f"{name}="):
                    return line.strip().split("=", 1)[1]
    return ""


def _require_env(name):
    value = os.environ.get(name, "") or _read_dotenv_value(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your environment or .env.")
    return value


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, model=None, temperature=None, retries=None, retry_delay=None):
        if not OPENAI_SDK_AVAILABLE:
            raise ImportError("openai package is required for the OpenAI backend.")
        self.model = model or os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
        self.temperature = float(
            os.environ.get("MEDMATCH_TEMPERATURE", "0.1") if temperature is None else temperature
        )
        self.retries = int(os.environ.get("MEDMATCH_MAX_RETRIES", "2") if retries is None else retries)
        self.retry_delay = float(
            os.environ.get("MEDMATCH_RETRY_DELAY", "5") if retry_delay is None else retry_delay
        )
        self.client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))

    def generate_text(self, system_prompt, user_prompt, *, temperature=None):
        temp = self.temperature if temperature is None else temperature
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temp,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(self.retry_delay)

    def generate_json(self, system_prompt, user_prompt, expected_keys, *, temperature=None):
        text = self.generate_text(system_prompt, user_prompt, temperature=temperature)
        parsed = parse_json_response(text)
        return coerce_output_object(parsed, expected_keys), text


class AzureOpenAIBackend(LLMBackend):
    def __init__(self, model=None, temperature=None, retries=None, retry_delay=None):
        if not OPENAI_SDK_AVAILABLE:
            raise ImportError("openai package is required for the Azure backend.")
        if not AZURE_IDENTITY_AVAILABLE:
            raise ImportError("azure-identity package is required for the Azure backend.")

        deployment = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get(
            "AZURE_MODEL_NAME", "gpt-4o-mini"
        )
        self.model = deployment.replace("azure-", "")
        self.temperature = float(
            os.environ.get("MEDMATCH_TEMPERATURE", "0.1") if temperature is None else temperature
        )
        self.retries = int(os.environ.get("MEDMATCH_MAX_RETRIES", "2") if retries is None else retries)
        self.retry_delay = float(
            os.environ.get("MEDMATCH_RETRY_DELAY", "5") if retry_delay is None else retry_delay
        )
        endpoint = _require_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
        credential = DefaultAzureCredential(exclude_managed_identity_credential=True)
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )

    def generate_text(self, system_prompt, user_prompt, *, temperature=None):
        temp = self.temperature if temperature is None else temperature
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temp,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(self.retry_delay)

    def generate_json(self, system_prompt, user_prompt, expected_keys, *, temperature=None):
        text = self.generate_text(system_prompt, user_prompt, temperature=temperature)
        parsed = parse_json_response(text)
        return coerce_output_object(parsed, expected_keys), text
