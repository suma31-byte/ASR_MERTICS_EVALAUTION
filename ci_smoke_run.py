#!/usr/bin/env python3
"""
Evaluate ten samples end to end, with no phone attached.

    python ci_smoke_run.py

Builds a ten-sample dataset whose transcripts are corrupted in known ways,
runs the real harness against a stand-in for the app, and checks that the
metrics come out as the corruptions dictate. Prints the results table and the
summary, then writes the same CSV and Excel report a real run would.

This exists because the unit tests prove each piece works in isolation and say
nothing about whether the pieces are still connected. A run that plays audio
before recording starts, or scores the live text instead of the transcript
button, passes every unit test and produces a full report of nonsense. Ten
samples with known answers catch that in under a second.

Exits non-zero if any expected figure is wrong, so CI fails on it.
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# No sound card on a CI runner, and nothing here plays audio.
sys.modules.setdefault("sounddevice", types.SimpleNamespace(
    play=lambda *a, **k: None, wait=lambda: None, stop=lambda: None,
    default=types.SimpleNamespace(device=(0, 1)),
    query_devices=lambda index=None, kind=None: {
        "name": "ci stub", "default_samplerate": 48000.0}))

import evaluate_asr as ev  # noqa: E402

# reference, what the app "hears", and what that corruption should score
SAMPLES = [
    ("perfect match",
     "the meeting starts at ten thirty tomorrow",
     "the meeting starts at ten thirty tomorrow", (0, 0, 0)),
    ("one word substituted",
     "yes we have crop loans in assam",
     "yes we have crop loans in bengal", (1, 0, 0)),
    ("one word deleted",
     "please transfer five thousand rupees today",
     "please transfer five thousand rupees", (0, 1, 0)),
    ("one word inserted",
     "book a cab to the airport",
     "book a cab to the main airport", (0, 0, 1)),
    ("two words substituted",
     "the parcel will arrive on friday evening",
     "the parcel will arrive on monday morning", (2, 0, 0)),
    ("three words deleted",
     "she studied computer science at anna university in chennai",
     "she studied computer science at anna", (0, 3, 0)),
    ("substitution and deletion",
     "the train leaves from platform number seven",
     "the train leaves from platform nine", (1, 1, 0)),
    ("heard nothing at all",
     "please reduce the speaker volume", "", None),
    ("spoken number against digits",
     "the appointment is on 15 june",
     "the appointment is on fifteenth june", (0, 0, 0)),
    ("whole sentence wrong",
     "ugadi is celebrated as the new year",
     "the weather is cold outside today", None),
]


class FakePhone:
    """Behaves like the app: records, saves a session, answers the button."""

    def __init__(self, heard: dict[str, str]):
        self.heard = heard
        self.history: list[dict] = []
        self.recording = False
        self.events: list[str] = []
        self.playing: str | None = None
        self._n = 0

    def toggle_record(self):
        if self.recording:
            self.recording = False
            self._n += 1
            self.events.append("stop")
            full = self.heard.get(self.playing, "")
            # The live text is deliberately a partial version of what the
            # TRANSCRIPT button later produces. If the harness ever scores the
            # live field by mistake, the metrics below will not match and this
            # run fails - which is the whole point of checking it here.
            partial = " ".join(full.split()[:2])
            self.history.insert(0, {
                "id": f"s{self._n}", "timestamp": f"{1700000000 + self._n}",
                "text": partial,
            })
        else:
            self.recording = True
            self.events.append("start")

    def play(self, sample_id: str):
        if not self.recording:
            self.events.append("PLAYED WHILE NOT RECORDING")
        self.playing = sample_id
        self.events.append("play")

    def press_transcript(self):
        self.events.append("transcript")
        if self.history:
            self.history[0]["transcriptText"] = self.heard.get(self.playing, "")


def build_dataset(root: Path) -> Path:
    audio = root / "combined"
    audio.mkdir(parents=True, exist_ok=True)
    rows = ["id,transcription,file,content_type"]
    for i, (_, reference, _, _) in enumerate(SAMPLES, 1):
        sid = f"{i:05d}"
        (audio / f"{sid}.wav").write_bytes(b"RIFF----WAVEfmt ")
        rows.append(f"{sid},{reference},combined/{sid}.wav,general")
    manifest = root / "dataset_ci.csv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def wire_up(root: Path, phone: FakePhone) -> None:
    """Replace everything that would touch hardware."""
    ev.DATASET_DIR = root
    ev.RESULTS_CSV = root / "results_detailed.csv"
    ev.STATUS_JSON = root / "status.json"
    ev.STOP_REQUEST = root / "stop.request"
    ev.ANALYSIS_MODE = "transcript"
    ev.PRE_RECORD_SETTLE_S = 0
    ev.POST_RECORD_TAIL_S = 0
    ev.INTER_SAMPLE_DELAY_S = 0
    ev.POLL_INTERVAL_S = 0.01

    ev.ensure_single_device = lambda s: "CI-FAKE"
    ev.check_debuggable = lambda s: None
    ev.grant_record_audio = lambda s: None
    ev.launch_app = lambda s: None
    ev.keep_screen_awake = lambda s: None
    ev.is_overlay_open = lambda s: False
    ev.ensure_screen_ready = lambda s: True
    ev.close_session_overlay = lambda s: None
    ev.choose_mic = lambda s, preset: "phone"
    ev.launch_dashboard = lambda **k: None
    ev.get_mic_xy = lambda s: (540, 1454)
    ev.app_version = lambda s: "ci-stub"
    ev.device_model = lambda s: "FakePhone"
    ev.pin_output_device = lambda: {"name": "ci stub"}
    ev.output_device_changed = lambda: None
    ev.check_resources = lambda s, loud=True: None
    ev.app_message_on_screen = lambda s: None
    ev.read_transcript_from_ui = lambda s: ""
    ev.find_analysis_button_xy = lambda s, labels=None: (540, 1800)
    ev.read_history = lambda s: list(phone.history)
    ev.is_recording = lambda s: phone.recording

    def tap(x, y, serial=None):
        phone.press_transcript() if y == 1800 else phone.toggle_record()

    def play(path, rms=None, target_dbfs=None):
        phone.play(Path(path).stem)
        return 1.0, 0.0

    ev.tap = tap
    ev.play_audio_blocking = play


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="asr-ci-"))
    try:
        manifest = build_dataset(root)
        heard = {f"{i:05d}": h for i, (_, _, h, _) in enumerate(SAMPLES, 1)}
        phone = FakePhone(heard)
        wire_up(root, phone)

        print("=" * 78)
        print("  CI smoke run: 10 samples through the full pipeline, no phone")
        print("=" * 78)
        ev.cmd_run(serial=None, limit=None, start=0, manifest=manifest,
                   mic_mode="phone", resume=False, dashboard=False)

        rows = list(csv.DictReader(ev.RESULTS_CSV.open(encoding="utf-8")))
        by_id = {r["id"]: r for r in rows}

        print("\n" + "=" * 78)
        print(f"  {'what was corrupted':<28}{'WER %':>8}{'S':>4}{'D':>4}{'I':>4}"
              f"   expected   result")
        print("-" * 78)

        failures: list[str] = []
        for i, (label, _, _, expected) in enumerate(SAMPLES, 1):
            r = by_id.get(f"{i:05d}")
            if r is None:
                failures.append(f"{label}: no result row")
                continue
            s, d, ins = (int(r["substitutions"] or 0), int(r["deletions"] or 0),
                         int(r["insertions"] or 0))
            wer = float(r["WER"] or 0)
            if expected is None:
                verdict = "ok" if (wer > 0 or not r["hypothesis"]) else "WRONG"
                want = "errors"
            else:
                want = f"{expected[0]}/{expected[1]}/{expected[2]}"
                verdict = "ok" if (s, d, ins) == expected else "WRONG"
            if verdict == "WRONG":
                failures.append(f"{label}: got S={s} D={d} I={ins}, expected {want}")
            print(f"  {label:<28}{wer:>8.2f}{s:>4}{d:>4}{ins:>4}   {want:>8}   {verdict}")

        scored = [r for r in rows if r["status"] == "ok" and r["hypothesis"].strip()]
        print("-" * 78)
        print(f"  {len(rows)} samples attempted, {len(scored)} scored, "
              f"{len(rows) - len(scored)} failed")

        # the flow itself, not just the arithmetic
        if "PLAYED WHILE NOT RECORDING" in phone.events:
            failures.append("audio was played before recording started")
        if phone.events[:4] != ["start", "play", "stop", "transcript"]:
            failures.append(f"wrong order of operations: {phone.events[:4]}")
        fields = {r.get("analysis_field") for r in scored}
        if fields != {"transcriptText"}:
            failures.append(f"scored the wrong field: {fields}")

        # and that the report still builds from it
        try:
            import make_report
            report = make_report.build(ev.RESULTS_CSV, root / "ci_report.xlsx")
            print(f"  Report built: {report.name} ({report.stat().st_size // 1024} kB)")
        except Exception as e:
            failures.append(f"report could not be built: {type(e).__name__}: {e}")

        # keep the artefacts where CI can upload them
        out = HERE / "ci_output"
        out.mkdir(exist_ok=True)
        for f in (ev.RESULTS_CSV, root / "ci_report.xlsx"):
            if f.exists():
                shutil.copy(f, out / f.name)
        print(f"  Artefacts in: {out.name}/")

        print("=" * 78)
        if failures:
            print("  FAILED")
            for f in failures:
                print(f"    - {f}")
            return 1
        print("  All ten samples scored as expected; the pipeline is intact.")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
