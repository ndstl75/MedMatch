from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"

for candidate in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from scripts import probing_medmatch
from scripts import run_cot
from scripts import run_normalization


class QwenLocalPipelineConfigTests(unittest.TestCase):
    def test_qwen_local_default_model_name_respects_openai_model_name_fallback(self):
        env = {"OPENAI_MODEL_NAME": "Qwen/Qwen3.6-14B"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                probing_medmatch.default_model_name_for_mode("qwen_local"),
                "Qwen/Qwen3.6-14B",
            )
            self.assertEqual(run_cot.default_model_name("qwen_local"), "Qwen/Qwen3.6-14B")
            self.assertEqual(run_normalization.default_model_name("qwen_local"), "Qwen/Qwen3.6-14B")

    def test_qwen_local_create_backend_preserves_explicit_model_name(self):
        with patch.object(probing_medmatch, "LocalQwenOpenAIBackend") as backend_cls:
            probing_medmatch.create_backend("qwen_local", "gpt-4o-mini", 0.2)

        backend_cls.assert_called_once_with(model="gpt-4o-mini", temperature=0.2)

    def test_local_openai_default_model_name_uses_gemma4_local_endpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                probing_medmatch.default_model_name_for_mode("local_openai"),
                "google/gemma-4-26B-A4B-it",
            )
            self.assertEqual(run_cot.default_model_name("local_openai"), "google/gemma-4-26B-A4B-it")
            self.assertEqual(
                run_normalization.default_model_name("local_openai"),
                "google/gemma-4-26B-A4B-it",
            )

    def test_local_openai_default_model_name_respects_local_openai_model_name(self):
        env = {"LOCAL_OPENAI_MODEL_NAME": "local/custom-gemma"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                probing_medmatch.default_model_name_for_mode("local_openai"),
                "local/custom-gemma",
            )
            self.assertEqual(run_cot.default_model_name("local_openai"), "local/custom-gemma")
            self.assertEqual(run_normalization.default_model_name("local_openai"), "local/custom-gemma")

    def test_local_openai_create_backend_preserves_explicit_model_name(self):
        with patch.object(probing_medmatch, "LocalOpenAIBackend") as backend_cls:
            probing_medmatch.create_backend("local_openai", "google/gemma-4-26B-A4B-it", 0.2)

        backend_cls.assert_called_once_with(model="google/gemma-4-26B-A4B-it", temperature=0.2)

    def test_local_openai_uses_local_prompt_style(self):
        self.assertFalse(probing_medmatch.is_remote_style_mode("local_openai"))


if __name__ == "__main__":
    unittest.main()
