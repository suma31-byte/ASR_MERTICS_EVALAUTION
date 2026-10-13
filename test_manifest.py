"""Loading a dataset, and refusing to load a broken one."""

import pytest


def test_a_dataset_loads_with_its_audio_resolved(ev, manifest_csv):
    rows = ev.load_manifest(manifest_csv)
    assert len(rows) == 3
    assert rows[0]["id"] == "00001"
    assert rows[0]["text"] == "the meeting starts at eight twenty"
    assert rows[0]["audio_path"].exists()
    assert rows[1]["content_type"] == "domain_terms"


def test_corrupt_references_stop_the_run(ev, tmp_path):
    # The export bug that put a whole manifest line into the transcript column
    # went unnoticed through a full balancing pass. Scored as-is, the app is
    # marked wrong for every word it did not invent.
    audio = tmp_path / "combined"
    audio.mkdir()
    (audio / "a.wav").write_bytes(b"RIFF")
    csv_path = tmp_path / "dataset_v1.csv"
    csv_path.write_text(
        "id,transcription,file\n"
        "1,audio/281474976879627_f1171_chunk_0.wav\t72262\tIs Gaia Green Tea available,combined/a.wav\n",
        encoding="utf-8")
    with pytest.raises(SystemExit, match="references need repair"):
        ev.load_manifest(csv_path)


def test_missing_audio_stops_the_run(ev, tmp_path):
    csv_path = tmp_path / "dataset_v1.csv"
    csv_path.write_text("id,transcription,file\n1,hello there,combined/missing.wav\n",
                        encoding="utf-8")
    with pytest.raises(SystemExit, match="audio could not be found"):
        ev.load_manifest(csv_path)


def test_the_result_schema_carries_the_tags(ev):
    for column in ("id", "reference", "hypothesis", "WER", "CER", "SER",
                   "substitutions", "deletions", "insertions",
                   "content_type", "status", "analysis_field"):
        assert column in ev.RESULT_FIELDS
