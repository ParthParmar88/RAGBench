"""Shared test setup.

`app.config.Settings` requires OPENAI_API_KEY, and importing almost anything
under `app.` pulls settings in transitively. Tests here are offline (pure
functions and monkeypatched LLM calls), so a dummy key is injected before any
app import. This keeps the suite runnable on a machine with no .env and no
secrets — a prerequisite for running it in CI later.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
