from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from medmatch.llm import remote_api


class FakeOpenAIClient:
    last_init = None
    last_request = None

    def __init__(self, **kwargs):
        FakeOpenAIClient.last_init = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        FakeOpenAIClient.last_request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"drug_name":"adenosine"}'))]
        )


class OpenAIBackendTests(unittest.TestCase):
    def setUp(self):
        FakeOpenAIClient.last_init = None
        FakeOpenAIClient.last_request = None

    def test_openai_backend_uses_base_url_and_extra_body_env(self):
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "http://127.0.0.1:8011/v1/",
            "OPENAI_MODEL": "Qwen/Qwen3.6-35B-A3B",
            "OPENAI_EXTRA_BODY_JSON": '{"chat_template_kwargs":{"enable_thinking":false}}',
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(remote_api, "OPENAI_SDK_AVAILABLE", True):
                with patch.object(remote_api, "OpenAI", FakeOpenAIClient):
                    backend = remote_api.OpenAICompatibleBackend()
                    text = backend.generate_text("system", "user", temperature=0)

        self.assertEqual(text, '{"drug_name":"adenosine"}')
        self.assertEqual(
            FakeOpenAIClient.last_init,
            {"api_key": "test-key", "base_url": "http://127.0.0.1:8011/v1"},
        )
        self.assertEqual(
            FakeOpenAIClient.last_request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(FakeOpenAIClient.last_request["model"], "Qwen/Qwen3.6-35B-A3B")

    def test_qwen_local_backend_has_local_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(remote_api, "OPENAI_SDK_AVAILABLE", True):
                with patch.object(remote_api, "OpenAI", FakeOpenAIClient):
                    backend = remote_api.LocalQwenOpenAIBackend()
                    backend.generate_text("system", "user", temperature=0)

        self.assertEqual(backend.model, "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(
            FakeOpenAIClient.last_init,
            {"api_key": "local-qwen", "base_url": "http://127.0.0.1:8011/v1"},
        )
        self.assertEqual(
            FakeOpenAIClient.last_request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_local_openai_backend_has_gemma4_defaults_and_no_default_extra_body(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(remote_api, "OPENAI_SDK_AVAILABLE", True):
                with patch.object(remote_api, "OpenAI", FakeOpenAIClient):
                    backend = remote_api.LocalOpenAIBackend()
                    backend.generate_text("system", "user", temperature=0)

        self.assertEqual(backend.model, "google/gemma-4-26B-A4B-it")
        self.assertEqual(
            FakeOpenAIClient.last_init,
            {"api_key": "local-gemma4", "base_url": "http://127.0.0.1:8013/v1"},
        )
        self.assertNotIn("extra_body", FakeOpenAIClient.last_request)

    def test_local_openai_backend_accepts_local_extra_body_env(self):
        env = {
            "LOCAL_OPENAI_EXTRA_BODY": '{"enable_thinking":false,"chat_template_kwargs":{"enable_thinking":false}}',
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.object(remote_api, "OPENAI_SDK_AVAILABLE", True):
                with patch.object(remote_api, "OpenAI", FakeOpenAIClient):
                    backend = remote_api.LocalOpenAIBackend()
                    backend.generate_text("system", "user", temperature=0)

        self.assertEqual(
            FakeOpenAIClient.last_request["extra_body"],
            {"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_openai_backend_falls_back_to_reasoning_content_when_content_empty(self):
        class FakeReasoningClient(FakeOpenAIClient):
            def create(self, **kwargs):
                FakeOpenAIClient.last_request = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=None, reasoning_content='{"drug_name":"adenosine"}')
                        )
                    ]
                )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch.object(remote_api, "OPENAI_SDK_AVAILABLE", True):
                with patch.object(remote_api, "OpenAI", FakeReasoningClient):
                    backend = remote_api.OpenAICompatibleBackend()
                    text = backend.generate_text("system", "user", temperature=0)

        self.assertEqual(text, '{"drug_name":"adenosine"}')
