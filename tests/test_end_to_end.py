"""A full run against a simulated phone.

Every other test checks one function. This one drives the pipeline the way a
real run does - manifest, record, play, read the transcript back, score, save,
report - and asserts the flow held together. No phone, no sound card, no adb:
the device is replaced by a small object that behaves like the app, so the
whole thing runs on a CI machine in under a second.

What this catches that unit tests do not: a stage wired to the wrong function,
a result column that silently stopped being written, a report that no longer
builds from the CSV. It is deliberately picky about the ORDER of events,
because a run that plays audio before recording starts still produces a
plausible-looking report full of empty transcripts.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


class FakePhone:
    """Stands in for the app: records, saves a session, answers the buttons."""

    def __init__(self, transcripts: dict[str, str]):
        self.transcripts = transcripts          # sample id -> what it "hears"
        self.history: list[dict] = []
        self.recording = False
        self.events: list[str] = []
        self.playing: str | None = None
        self._n = 0

    # the three things the harness does to it
    def toggle_record(self):
        if self.recording:
            self.recording = False
            self.events.append("stop")
            self._n += 1
            heard = self.transcripts.get(self.playing, "")
            self.history.insert(0, {
                "id": f"s{self._n}", "timestamp": f"{1700000000 + self._n}",
                "text": heard,
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
            # the button writes its own field, distinct from the live text
            self.history[0]["transcriptText"] = self.history[0]["text"]


@pytest.fixture
def run_environment(ev, tmp_path, monkeypatch):
    """A three-sample dataset and a phone that transcribes it imperfectly."""
    audio = tmp_path / "combined"
    audio.mkdir()
    samples = [
        ("00001", "the meeting starts at eight twenty", "the meeting starts at eight twenty"),
        ("00002", "yes we have crop loans in assam", "yes we have crop loans"),
        ("00003", "please reduce the speaker volume", ""),          # heard nothing
    ]
    for sid, _, _ in samples:
        (audio / f"{sid}.wav").write_bytes(b"RIFF----WAVEfmt ")

    manifest = tmp_path / "dataset_v9.csv"
    manifest.write_text(
        "id,transcription,file,content_type\n"
        + "".join(f"{sid},{ref},combined/{sid}.wav,general\n" for sid, ref, _ in samples),
        encoding="utf-8")

    phone = FakePhone({sid: heard for sid, _, heard in samples})
    current = {"id": None}

    # everything that would touch hardware
    monkeypatch.setattr(ev, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(ev, "RESULTS_CSV", tmp_path / "results_detailed.csv")
    monkeypatch.setattr(ev, "STATUS_JSON", tmp_path / "status.json")
    monkeypatch.setattr(ev, "STOP_REQUEST", tmp_path / "stop.request")
    monkeypatch.setattr(ev, "ensure_single_device", lambda s: "FAKE")
    monkeypatch.setattr(ev, "check_debuggable", lambda s: None)
    monkeypatch.setattr(ev, "grant_record_audio", lambda s: None)
    monkeypatch.setattr(ev, "launch_app", lambda s: None)
    monkeypatch.setattr(ev, "keep_screen_awake", lambda s: None)
    monkeypatch.setattr(ev, "is_overlay_open", lambda s: False)
    monkeypatch.setattr(ev, "ensure_screen_ready", lambda s: True)
    monkeypatch.setattr(ev, "close_session_overlay", lambda s: None)
    monkeypatch.setattr(ev, "choose_mic", lambda s, preset: "phone")
    monkeypatch.setattr(ev, "launch_dashboard", lambda **k: None)
    monkeypatch.setattr(ev, "get_mic_xy", lambda s: (540, 1454))
    monkeypatch.setattr(ev, "app_version", lambda s: "1.1-test")
    monkeypatch.setattr(ev, "device_model", lambda s: "FakePhone")
    monkeypatch.setattr(ev, "pin_output_device", lambda: {"name": "stub"})
    monkeypatch.setattr(ev, "output_device_changed", lambda: None)
    monkeypatch.setattr(ev, "check_resources", lambda s, loud=True: None)
    monkeypatch.setattr(ev, "app_message_on_screen", lambda s: None)
    monkeypatch.setattr(ev, "read_transcript_from_ui", lambda s: "")
    monkeypatch.setattr(ev, "find_analysis_button_xy", lambda s, labels=None: (540, 1800))
    monkeypatch.setattr(ev, "POST_RECORD_TAIL_S", 0)
    monkeypatch.setattr(ev, "PRE_RECORD_SETTLE_S", 0)
    monkeypatch.setattr(ev, "INTER_SAMPLE_DELAY_S", 0)
    monkeypatch.setattr(ev, "POLL_INTERVAL_S", 0.01)

    monkeypatch.setattr(ev, "read_history", lambda s: list(phone.history))
    monkeypatch.setattr(ev, "is_recording", lambda s: phone.recording)

    def fake_tap(x, y, serial=None):
        # the mic button and the analysis button are at known positions
        phone.press_transcript() if y == 1800 else phone.toggle_record()

    def fake_play(path, rms=None, target_dbfs=None):
        phone.play(Path(path).stem)
        return 1.0, 0.0

    monkeypatch.setattr(ev, "tap", fake_tap)
    monkeypatch.setattr(ev, "play_audio_blocking", fake_play)
    monkeypatch.setattr(ev, "ANALYSIS_MODE", "transcript")

    return ev, phone, manifest, tmp_path, current


def test_a_run_scores_every_sample_and_writes_a_report(run_environment):
    ev, phone, manifest, tmp_path, _ = run_environment

    ev.cmd_run(serial=None, limit=None, start=0, manifest=manifest,
               mic_mode="phone", resume=False, dashboard=False)

    # Each run mints its own timestamped results file, so read the one it used.
    results = ev.RESULTS_CSV
    assert results.exists(), "the run wrote no results file"
    rows = list(csv.DictReader(results.open(encoding="utf-8")))
    assert len(rows) == 3, f"expected 3 scored samples, got {len(rows)}"

    by_id = {r["id"]: r for r in rows}

    # heard perfectly
    assert by_id["00001"]["status"] == "ok"
    assert float(by_id["00001"]["WER"]) == 0.0

    # three words dropped at the end
    assert by_id["00002"]["status"] == "ok"
    assert float(by_id["00002"]["WER"]) > 0
    assert int(by_id["00002"]["deletions"]) == 2

    # heard nothing: recorded as a failure, never scored as a bad transcript
    assert by_id["00003"]["hypothesis"] == ""
    assert by_id["00003"]["status"] != "ok"


def test_the_order_of_events_is_record_play_stop_transcript(run_environment):
    ev, phone, manifest, tmp_path, _ = run_environment
    ev.cmd_run(serial=None, limit=1, start=0, manifest=manifest,
               mic_mode="phone", resume=False, dashboard=False)

    assert "PLAYED WHILE NOT RECORDING" not in phone.events, (
        "audio was played before recording started - every transcript would be empty")
    assert phone.events[:4] == ["start", "play", "stop", "transcript"]


def test_the_transcript_button_output_is_scored_not_the_live_text(run_environment):
    ev, phone, manifest, tmp_path, _ = run_environment
    ev.cmd_run(serial=None, limit=1, start=0, manifest=manifest,
               mic_mode="phone", resume=False, dashboard=False)

    rows = list(csv.DictReader(ev.RESULTS_CSV.open(encoding="utf-8")))
    assert rows[0]["analysis_field"] == "transcriptText", (
        "the live `text` field was scored instead of the TRANSCRIPT button output")


def test_the_report_builds_from_the_results(run_environment):
    ev, phone, manifest, tmp_path, _ = run_environment
    ev.cmd_run(serial=None, limit=None, start=0, manifest=manifest,
               mic_mode="phone", resume=False, dashboard=False)

    openpyxl = pytest.importorskip("openpyxl")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import make_report

    out = make_report.build(ev.RESULTS_CSV, tmp_path / "report.xlsx")
    assert out.exists()

    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["Summary", "Per sample"]
    assert len(wb["Summary"]._charts) == 1, "the summary chart is missing"
    assert wb["Per sample"].max_row == 4          # header plus three samples


def test_resuming_skips_what_was_already_scored(run_environment):
    ev, phone, manifest, tmp_path, _ = run_environment
    ev.cmd_run(serial=None, limit=2, start=0, manifest=manifest,
               mic_mode="phone", resume=False, dashboard=False)
    first_pass = len(phone.events)

    ev.cmd_run(serial=None, limit=2, start=0, manifest=manifest,
               mic_mode="phone", resume=True, dashboard=False)
    assert len(phone.events) == first_pass, (
        "a resumed run re-recorded samples that were already scored")
