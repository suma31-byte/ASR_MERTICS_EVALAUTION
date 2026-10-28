#!/usr/bin/env python3
"""
Upload evaluation results to Firestore.

    python upload_results.py                        # newest results file
    python upload_results.py --results results_detailed_20261021_104824.csv
    python upload_results.py --summary-only         # just the run summary
    python upload_results.py --dry-run              # count writes, send nothing

Deliberately a separate step, run after an evaluation rather than during one.
A run that depends on the network to record a sample is a run that loses
samples when the network goes. The CSV is written first and stays the source
of truth; this copies it up afterwards.

What it writes
    runs/{run_id}__{timestamp}   one summary document per upload

A new document each time rather than an update to the run's own, so every exit
leaves its own record and the history of a long run survives. Each carries the
run id, the results file, the report it corresponds to, the elapsed time and
the figures.

Per-sample documents are off by default - a full run is thousands of writes
against a 20,000-a-day free tier, and the per-sample detail is already in the
CSV and the Excel report. --with-samples turns them on.

Re-running is safe and cheap. Every uploaded sample id is recorded locally, so
a second invocation sends only what is new - which matters because Firestore's
free tier allows 20,000 writes a day and a full 19,000-sample run sits just
under that. --max-writes stops short of a limit deliberately.

Setup
    pip install firebase-admin
    Firebase console -> Project settings -> Service accounts -> Generate key
    set FIREBASE_KEY=C:\\path\\to\\firebase-key.json

Keep that key out of the repository. It grants full access to the project.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The run writes into a "results" folder beside the scripts unless told
# otherwise. Looking only beside the scripts, as this did, finds nothing once
# the results move - and says "no results file" while they sit one folder down.
RESULTS_DIR = Path(os.environ.get("ASR_RESULTS_DIR") or (HERE / "results"))
if not RESULTS_DIR.exists():
    RESULTS_DIR = HERE


def search_dirs() -> list[Path]:
    """Where results might be: the results folder, and the older layout."""
    dirs = [RESULTS_DIR]
    if HERE not in dirs:
        dirs.append(HERE)
    return dirs

BATCH_SIZE = 400          # Firestore allows 500 writes per batch; leave headroom
DEFAULT_MAX_WRITES = 18000   # under the 20,000/day free tier, with room to spare

# Columns worth keeping in the cloud. The normalised text and the raw button
# output are large and only useful next to the audio, so they stay local.
SAMPLE_FIELDS = [
    "id", "timestamp", "run_id", "mic_mode", "reference", "hypothesis",
    "status", "app_message", "analysis_field",
    "WER", "CER", "SER", "substitutions", "deletions", "insertions",
    "duration_s", "played_s", "speaker_analysis_s", "playback_gain_db",
    "block", "source", "gender", "age_group", "native_place_state",
    "primary_language", "content_type", "length_bucket", "speaking_rate",
    "snr_band", "category", "difficulty",
]
NUMERIC = {"WER", "CER", "duration_s", "played_s", "speaker_analysis_s",
           "playback_gain_db"}
INTEGER = {"SER", "substitutions", "deletions", "insertions"}


def num(value, as_int=False):
    raw = str(value or "").strip()
    if raw in ("", "None"):
        return None
    try:
        return int(float(raw)) if as_int else float(raw)
    except ValueError:
        return None


def latest_results() -> Path | None:
    files: list[Path] = []
    for folder in search_dirs():
        files.extend(folder.glob("results_detailed_*.csv"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def run_id_for(results: Path, rows: list[dict]) -> str:
    for r in rows:
        if (r.get("run_id") or "").strip():
            return r["run_id"].strip()
    m = re.search(r"(\d{8}_\d{6})", results.name)
    return m.group(1) if m else results.stem


def shape_sample(row: dict) -> dict:
    out = {}
    for key in SAMPLE_FIELDS:
        if key not in row:
            continue
        if key in NUMERIC:
            out[key] = num(row[key])
        elif key in INTEGER:
            out[key] = num(row[key], as_int=True)
        else:
            value = (row.get(key) or "").strip()
            if value:
                out[key] = value
    return out


def elapsed_minutes_for(results: Path) -> float | None:
    """Total time spent on this results file, from the run's status file."""
    status = next((f / "status.json" for f in search_dirs()
                   if (f / "status.json").exists()), None)
    if status is None:
        return None
    try:
        data = json.loads(status.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str((data.get("run_meta") or {}).get("results_file") or "") != results.name:
        return None          # the status file describes a different run
    try:
        return round(float((data.get("timing") or {}).get("elapsed_s") or 0) / 60, 1)
    except (TypeError, ValueError):
        return None


def latest_report(results: Path) -> str | None:
    """The most recent Excel report, which is what this upload corresponds to."""
    files: list[Path] = []
    for folder in search_dirs():
        files.extend(f for f in folder.glob("report_*.xlsx")
                     if f.name != "report_in_progress.xlsx")
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime).name


def summarise(rows: list[dict], run_id: str, results: Path) -> dict:
    """The run's headline figures, and nothing else.

    Failed and empty samples are counted but never averaged: a sample that
    produced no transcript is not a sample the recogniser got wrong.
    """
    scored = [r for r in rows
              if r.get("status") == "ok" and (r.get("hypothesis") or "").strip()]
    failed = [r for r in rows if r not in scored]

    def avg(key):
        vals = [num(r.get(key)) for r in scored]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "run_id": run_id,
        "results_file": results.name,
        "report_file": latest_report(results),
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": elapsed_minutes_for(results),
        "attempted": len(rows),
        "scored": len(scored),
        "failed": len(failed),
        "WER_avg": avg("WER"),
        "CER_avg": avg("CER"),
        "SER_pct": (round(sum(1 for r in scored if (num(r.get("SER")) or 0) > 0)
                          / len(scored) * 100, 2) if scored else None),
        "substitutions": sum(num(r.get("substitutions"), True) or 0 for r in scored),
        "deletions": sum(num(r.get("deletions"), True) or 0 for r in scored),
        "insertions": sum(num(r.get("insertions"), True) or 0 for r in scored),
    }


# ----------------------------------------------------------- upload state --
def state_path(run_id: str) -> Path:
    """Beside the results it describes, so moving the results moves the state."""
    existing = next((f / f"upload_state_{run_id}.json" for f in search_dirs()
                     if (f / f"upload_state_{run_id}.json").exists()), None)
    return existing or (RESULTS_DIR / f"upload_state_{run_id}.json")


def load_uploaded(run_id: str) -> set[str]:
    path = state_path(run_id)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("ids", []))
    except (OSError, json.JSONDecodeError):
        return set()


def save_uploaded(run_id: str, ids: set[str]) -> None:
    try:
        state_path(run_id).write_text(
            json.dumps({"run_id": run_id, "ids": sorted(ids),
                        "updated": time.strftime("%Y-%m-%d %H:%M:%S")}),
            encoding="utf-8")
    except OSError as e:
        print(f"  (could not save the upload state: {e})")


# --------------------------------------------------------------- firebase --
def connect(key_path: str | None):
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        raise SystemExit(
            "firebase-admin is not installed.\n    pip install firebase-admin")

    key = key_path or os.environ.get("FIREBASE_KEY")
    if not key:
        raise SystemExit(
            "No service account key. Firebase console -> Project settings -> "
            "Service accounts -> Generate new private key, then:\n"
            "    set FIREBASE_KEY=C:\\path\\to\\serviceAccountKey.json\n"
            "or pass --key. Keep it out of the repository.")
    if not Path(key).exists():
        raise SystemExit(f"Service account key not found: {key}")

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(key))
    return firestore.client()


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload evaluation results to Firestore")
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--key", default=None, help="service account JSON (or FIREBASE_KEY)")
    ap.add_argument("--collection", default="runs")
    ap.add_argument("--live", action="store_true",
                    help="Write to a single <run_id>__live document instead of a "
                         "new one. For progress updates during a run: a record "
                         "every few minutes would otherwise leave hundreds of "
                         "documents for one run.")
    ap.add_argument("--doc-id", default=None,
                    help="Document id to write (default: run id plus the current "
                         "time, so every upload is its own record)")
    ap.add_argument("--with-samples", action="store_true",
                    help="also write one document per sample (off by default: "
                         "the summary is one write, a full run is thousands)")
    ap.add_argument("--max-writes", type=int, default=DEFAULT_MAX_WRITES,
                    help=f"stop after this many writes (default {DEFAULT_MAX_WRITES}, "
                         "under the 20,000/day free tier)")
    ap.add_argument("--all", action="store_true",
                    help="re-send samples already uploaded for this run")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be sent, contact nothing")
    cfg = ap.parse_args()

    results = cfg.results or latest_results()
    if results is None or not results.exists():
        looked = ", ".join(str(f) for f in search_dirs())
        print(f"No results_detailed_*.csv found. Looked in: {looked}")
        print("Pass --results with the file, or set ASR_RESULTS_DIR.")
        return 1

    with results.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{results.name} has no rows.")
        return 1

    run_id = run_id_for(results, rows)
    summary = summarise(rows, run_id, results)

    # A new document per upload, not an update to the run's one. Each exit
    # leaves its own record, so the history of a long run is visible rather
    # than being overwritten by whatever happened last. The id carries the run
    # and the moment, so the documents sort in order.
    if cfg.doc_id:
        doc_id = cfg.doc_id
    elif cfg.live:
        doc_id = f"{run_id}__live"
    else:
        doc_id = f"{run_id}__{time.strftime('%Y%m%d_%H%M%S')}"
    summary["doc_id"] = doc_id
    summary["state"] = "running" if cfg.live else "finished"
    uploaded = set() if cfg.all else load_uploaded(run_id)
    pending = [r for r in rows if (r.get("id") or "") not in uploaded]

    print("=" * 66)
    print(f"  {results.name}")
    print(f"  run {run_id}: {summary['attempted']} rows, "
          f"{summary['scored']} scored, {summary['failed']} failed")
    if summary["WER_avg"] is not None:
        print(f"  WER {summary['WER_avg']:.2f} %   CER {summary['CER_avg']:.2f} %   "
              f"SER {summary['SER_pct']:.2f} %")
    print("=" * 66)

    if not cfg.with_samples:
        pending = []
    capped = False
    if len(pending) > cfg.max_writes - 1:            # one write for the summary
        pending = pending[: max(0, cfg.max_writes - 1)]
        capped = True

    writes = 1 + len(pending)
    if cfg.with_samples:
        print(f"  already uploaded : {len(uploaded)}")
        print(f"  to send now      : {len(pending)} samples + 1 summary = {writes} writes")
    else:
        print("  sending the run summary (1 write). --with-samples adds a "
              "document per sample.")
    if capped:
        print(f"  capped at --max-writes {cfg.max_writes}; run again tomorrow for "
              "the rest (the state file remembers where it got to)")

    if cfg.dry_run:
        print("\n  dry run: nothing was sent.")
        print(f"  summary document would be: {cfg.collection}/{doc_id}")
        print(f"  its fields: {', '.join(summary)}")
        if pending:
            print(f"  first sample document   : {cfg.collection}/{run_id}/samples/"
                  f"{pending[0].get('id')}")
        return 0

    db = connect(cfg.key)
    run_doc = db.collection(cfg.collection).document(doc_id)
    # Replaces the document rather than merging into it. With merge, a field
    # written by an older version of this script stays forever: a summary that
    # no longer reports app_version would still show app_version: null, and
    # nothing would ever remove it.
    run_doc.set(summary)
    print("\n  summary written.")

    sent = 0
    for start in range(0, len(pending), BATCH_SIZE):
        chunk = pending[start:start + BATCH_SIZE]
        batch = db.batch()
        for row in chunk:
            sample_id = (row.get("id") or "").strip()
            if not sample_id:
                continue
            batch.set(run_doc.collection("samples").document(sample_id),
                      shape_sample(row))
        batch.commit()
        sent += len(chunk)
        uploaded |= {(r.get("id") or "").strip() for r in chunk}
        save_uploaded(run_id, uploaded)       # saved per batch, not at the end
        print(f"  {sent}/{len(pending)} samples uploaded")

    if cfg.with_samples:
        print(f"\n  Done. {cfg.collection}/{doc_id} holds {len(uploaded)} samples.")
    else:
        print(f"  Done. Wrote {cfg.collection}/{doc_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())