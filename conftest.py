"""Shared fixtures.

`sounddevice` needs PortAudio, which does not exist on a CI runner and has no
business being installed there: nothing in these tests plays audio. It is
stubbed before `evaluate_asr` is imported, so the whole module - normalisation,
metrics, manifest loading, the result schema - can be tested on any machine
with no hardware and no system packages.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _stub_sounddevice() -> None:
    if "sounddevice" in sys.modules:
        return
    sys.modules["sounddevice"] = types.SimpleNamespace(
        play=lambda *a, **k: None,
        wait=lambda: None,
        stop=lambda: None,
        default=types.SimpleNamespace(device=(0, 1)),
        query_devices=lambda index=None, kind=None: {
            "name": "stub output", "default_samplerate": 48000.0},
    )


@pytest.fixture(scope="session")
def ev():
    """The evaluation module, imported once with the audio layer stubbed."""
    _stub_sounddevice()
    sys.path.insert(0, str(ROOT))
    import evaluate_asr
    return evaluate_asr


@pytest.fixture
def manifest_csv(tmp_path: Path) -> Path:
    """A small dataset with its audio actually present on disk."""
    audio_dir = tmp_path / "combined"
    audio_dir.mkdir()
    rows = [
        ("00001", "the meeting starts at eight twenty", "combined/a.wav", "times"),
        ("00002", "yes we have crop loans in assam", "combined/b.wav", "domain_terms"),
        ("00003", "please reduce the speaker volume", "combined/c.wav", "general"),
    ]
    for _, _, rel, _ in rows:
        (tmp_path / rel).write_bytes(b"RIFF----WAVEfmt ")

    csv_path = tmp_path / "dataset_v9.csv"
    lines = ["id,transcription,file,content_type"]
    lines += [f"{i},{text},{rel},{cat}" for i, text, rel, cat in rows]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path
