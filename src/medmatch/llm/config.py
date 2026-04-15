"""Shared backend selection helpers."""

SUPPORTED_BACKENDS = ("local", "remote", "google", "openai", "azure")
REMOTE_BACKENDS = {"remote", "google", "openai", "azure"}


def canonical_backend_name(name):
    normalized = str(name).strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {name}")
    return normalized


def is_remote_backend(name):
    return canonical_backend_name(name) in REMOTE_BACKENDS
