"""Backend interface for MedMatch LLM calls."""

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate_text(self, system_prompt, user_prompt, *, temperature=None):
        raise NotImplementedError

    @abstractmethod
    def generate_json(self, system_prompt, user_prompt, expected_keys, *, temperature=None):
        raise NotImplementedError

