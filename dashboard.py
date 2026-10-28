#!/usr/bin/env python3
"""
SmartScribe ASR evaluation - control panel and live dashboard.

    python dashboard.py

That is the whole workflow. The dashboard opens in your browser, checks the ADB
connection to the phone, asks which microphone to use, and starts the run. The
evaluation itself is `evaluate_asr.py`, launched as a separate process, so the
dashboard can be closed and reopened without disturbing a run in progress.

Flow
  1. Launch the dashboard.
  2. It checks ADB: device attached and authorised, app installed, build
     debuggable.
  3. Choose phone mic or BLE mic.
       phone -> Start is available straight away.
       ble   -> connect the microphone in the app; Start unlocks once the phone
                reports the connection.
  4. Live status, per-sample and cumulative metrics, failures broken down by
     diversity tag, and the full run log.

Standard library only: no Flask, no CDN, no internet.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent

# Results, status and the stop request live in the folder the run writes to.
# Matching the run matters: a dashboard reading a status file nobody writes
# shows "idle" through an entire evaluation.
RESULTS_DIR = Path(os.environ.get("ASR_RESULTS_DIR") or (HERE / "results"))
if not RESULTS_DIR.exists() and (HERE / "status.json").exists():
    RESULTS_DIR = HERE                 # older layout, everything beside the scripts
STATUS = RESULTS_DIR / "status.json"
RUNNER = HERE / "evaluate_asr.py"
RUN_LOG = HERE / "run.log"
STOP_REQUEST = RESULTS_DIR / "stop.request"

PACKAGE = "com.example.speechtotext"
ADB = os.environ.get("ADB_PATH", "adb")

TAG_COLUMNS = ["content_type", "snr_band", "source", "gender", "age_group",
               "native_place_state", "length_bucket", "speaking_rate",
               "category", "difficulty", "mic_mode"]

_proc: subprocess.Popen | None = None
_cache: dict = {"path": None, "mtime": 0.0, "rows": []}
_adb_cache: dict = {"at": 0.0, "value": {}}
# When a browser last asked for the page or polled it. A run uses this to
# decide whether a tab is already open, instead of opening another one.
_last_view: dict = {"at": 0.0}
_ble_cache: dict = {"at": 0.0, "value": {}}


# ------------------------------------------------------------------ adb ----
def adb_exe() -> str:
    """Use the same adb binary evaluate_asr.py uses, so the two never disagree
    about which device they are talking to."""
    if ADB != "adb":
        return ADB
    if RUNNER.exists():
        m = re.search(r'^ADB\s*=\s*r?["\'](.+?)["\']',
                      RUNNER.read_text(encoding="utf-8"), re.MULTILINE)
        if m and Path(m.group(1)).exists():
            return m.group(1)
    return "adb"


def _adb(args: list[str], timeout: float = 12.0) -> str:
    try:
        out = subprocess.run([adb_exe()] + args, capture_output=True, text=True,
                             timeout=timeout)
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def preflight(force: bool = False) -> dict:
    """Everything that must be true before a run can start. Cached for a few
    seconds, since this is several adb round trips and the page polls often."""
    if not force and time.time() - _adb_cache["at"] < 6.0 and _adb_cache["value"]:
        return _adb_cache["value"]

    raw = _adb(["devices", "-l"])
    devices, unauthorised = [], []
    for line in raw.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            if parts[1] == "device":
                model = next((p.split(":", 1)[1] for p in parts if p.startswith("model:")), "")
                devices.append({"serial": parts[0], "model": model or "device"})
            elif parts[1] in ("unauthorized", "offline"):
                unauthorised.append({"serial": parts[0], "state": parts[1]})

    info = {"adb_found": bool(raw) or bool(_adb(["version"])),
            "devices": devices, "unauthorised": unauthorised,
            "app_installed": False, "debuggable": False, "app_version": "",
            "checked": time.strftime("%H:%M:%S")}

    if devices:
        info["app_installed"] = PACKAGE in _adb(["shell", f"pm list packages {PACKAGE}"])
        if info["app_installed"]:
            info["debuggable"] = "uid=" in _adb(["exec-out", f"run-as {PACKAGE} id"])
            for line in _adb(["shell", f"dumpsys package {PACKAGE} | grep versionName"]).splitlines():
                if "versionName" in line:
                    info["app_version"] = line.strip().split("=", 1)[-1].strip()
                    break

    info["ready"] = bool(devices) and info["app_installed"] and info["debuggable"]
    if not info["adb_found"]:
        info["problem"] = "adb not found - check it is on PATH"
    elif unauthorised:
        info["problem"] = "phone is unauthorised - unlock it and accept the USB debugging prompt"
    elif not devices:
        info["problem"] = "no phone detected - check the cable and that USB debugging is on"
    elif not info["app_installed"]:
        info["problem"] = f"{PACKAGE} is not installed on the phone"
    elif not info["debuggable"]:
        info["problem"] = "installed build is not debuggable - install the debug APK"
    else:
        info["problem"] = ""
    _adb_cache.update(at=time.time(), value=info)
    return info


def ble_status(force: bool = False) -> dict:
    """What the phone's Bluetooth stack currently has connected.

    The app's own scan-and-connect flow is the one that matters and cannot be
    driven from here without the debug hooks, so this reports what the phone
    sees. That is enough to stop a BLE run starting with nothing connected,
    which would otherwise produce a thousand empty transcripts.
    """
    if not force and time.time() - _ble_cache["at"] < 5.0 and _ble_cache["value"]:
        return _ble_cache["value"]

    dump = _adb(["shell", "dumpsys bluetooth_manager"], timeout=15.0)
    result = {"available": bool(dump), "enabled": None, "connected": [],
              "bonded": [], "is_connected": False,
              "checked": time.strftime("%H:%M:%S")}
    if dump:
        low = dump.lower()
        result["enabled"] = ("enabled: true" in low) or ("state: on" in low) or ("state: bleon" in low)
        section = None
        for raw_line in dump.splitlines():
            line = raw_line.strip()
            low_line = line.lower()
            if "connected devices:" in low_line:
                section = "connected"
                continue
            if "bonded devices:" in low_line:
                section = "bonded"
                continue
            m = re.match(r"([0-9A-F]{2}(?::[0-9A-F]{2}){5})\s*(.*)", line, re.I)
            if m and section:
                entry = {"address": m.group(1), "name": m.group(2).strip() or "(unnamed)"}
                target = result["connected"] if section == "connected" else result["bonded"]
                if entry["address"] not in [e["address"] for e in target]:
                    target.append(entry)
            elif not line:
                section = None
        result["is_connected"] = bool(result["connected"])
    _ble_cache.update(at=time.time(), value=result)
    return result


# --------------------------------------------------------------- results ---
def search_dirs() -> list[Path]:
    """Where results might be: the results folder, and the older layout where
    everything sat beside the scripts."""
    dirs = [RESULTS_DIR]
    if HERE not in dirs:
        dirs.append(HERE)
    return dirs


def latest_results() -> Path | None:
    files: list[Path] = []
    for folder in search_dirs():
        files.extend(folder.glob("results_detailed_*.csv"))
        legacy = folder / "results_detailed.csv"
        if legacy.exists():
            files.append(legacy)
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def read_results() -> list[dict]:
    """Cached on mtime: the file is rewritten after every sample and can run to
    thousands of rows, so re-parsing on every poll would be wasteful."""
    path = latest_results()
    if path is None:
        return []
    mtime = path.stat().st_mtime
    if _cache["path"] == path and _cache["mtime"] == mtime:
        return _cache["rows"]
    # Read a copy, never the live file. On Windows a file held open for
    # reading blocks the run's save - this dashboard polling every two seconds
    # was enough to stop a run that had been going for days. The copy is taken
    # with shutil, which opens the source only briefly, and parsed from there.
    rows = None
    tmp = path.with_suffix(".dashboard-read.tmp")
    try:
        shutil.copyfile(path, tmp)
        with tmp.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        rows = None
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    if rows is None:
        return _cache["rows"]
    _cache.update(path=path, mtime=mtime, rows=rows)
    return rows


def num(row: dict, key: str) -> float | None:
    raw = str(row.get(key, "")).strip()
    if raw in ("", "None"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def failure_analysis(rows: list[dict]) -> dict:
    """WER and failure count per diversity tag, worst first, with each group's
    gap to the overall average. This is the part that changes what you do next:
    an average over a thousand mixed samples hides the group that is failing."""
    scored = [r for r in rows if r.get("status") == "ok" and (r.get("hypothesis") or "").strip()]
    failed = [r for r in rows if r not in scored]
    overall = (sum(num(r, "WER") or 0 for r in scored) / len(scored)) if scored else 0.0

    out: dict = {"overall_wer": round(overall, 2), "scored": len(scored),
                 "failed": len(failed), "groups": {}}
    for tag in TAG_COLUMNS:
        if not rows or tag not in rows[0]:
            continue
        wers: dict[str, list[float]] = defaultdict(list)
        fails: dict[str, int] = defaultdict(int)
        for r in scored:
            v = (r.get(tag) or "").strip()
            w = num(r, "WER")
            if v and w is not None:
                wers[v].append(w)
        for r in failed:
            v = (r.get(tag) or "").strip()
            if v:
                fails[v] += 1
        groups = []
        for v in set(wers) | set(fails):
            ws = wers.get(v, [])
            wer = round(sum(ws) / len(ws), 2) if ws else None
            groups.append({"value": v, "n": len(ws) + fails.get(v, 0),
                           "scored": len(ws), "wer": wer,
                           "failed": fails.get(v, 0),
                           "delta": round(wer - overall, 2) if wer is not None else None})
        if groups:
            groups.sort(key=lambda g: (g["wer"] is None, -(g["wer"] or 0)))
            out["groups"][tag] = groups[:14]
    return out


def read_log_full() -> str:
    """The whole log. Served only when the log dropdown is opened, so a long
    run does not carry megabytes of text on every two-second poll."""
    if not RUN_LOG.exists():
        return ""
    try:
        return RUN_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_log(lines: int = 500) -> str:
    if not RUN_LOG.exists():
        return ""
    try:
        return "\n".join(RUN_LOG.read_text(encoding="utf-8", errors="replace")
                         .splitlines()[-lines:])
    except OSError:
        return ""


def list_outputs() -> list[dict]:
    files = []
    patterns = ("results_detailed_*.csv", "full_metrics_*.csv", "report_*.xlsx",
                "smartscribe_asr_eval_*.txt", "summary*.json")
    for folder, pattern in ((d, p) for d in search_dirs() for p in patterns):
        for f in folder.glob(pattern):
            st = f.stat()
            files.append({"name": f.name, "size_kb": round(st.st_size / 1024, 1),
                          "modified": time.strftime("%Y-%m-%d %H:%M",
                                                    time.localtime(st.st_mtime))})
    files.sort(key=lambda f: f["modified"], reverse=True)
    return files[:30]


def payload(full: bool = False) -> dict:
    status = {}
    if STATUS.exists():
        try:
            status = json.loads(STATUS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}

    running = _proc is not None and _proc.poll() is None
    if status.get("state") == "running" and not running and STATUS.exists():
        if time.time() - STATUS.stat().st_mtime > 180:
            status["state"] = "stalled"

    rows = read_results()
    results_file = latest_results()
    recent_rows = []
    for r in rows[-14:]:
        recent_rows.append({
            "id": r.get("id"), "status": r.get("status"),
            "WER": r.get("WER"), "CER": r.get("CER"),
            "reference": r.get("reference"), "hypothesis": r.get("hypothesis"),
            "substitutions": r.get("substitutions"), "deletions": r.get("deletions"),
            "insertions": r.get("insertions"),
            "wait_s": r.get("wait_s"), "played_s": r.get("played_s"),
            "speaker_analysis_s": r.get("speaker_analysis_s"),
            "app_message": r.get("app_message"),
            "tags": [r.get(t) for t in ("primary_language", "native_place_state",
                                        "gender", "age_group", "snr_band",
                                        "content_type", "length_bucket",
                                        "speaking_rate") if (r.get(t) or "").strip()],
            "content_type": r.get("content_type") or r.get("category") or "",
        })
    sa = [float(r["speaker_analysis_s"]) for r in rows
          if str(r.get("speaker_analysis_s") or "").strip() not in ("", "None")]
    tag_values = sorted({(r.get("content_type") or "").strip() for r in rows
                         if (r.get("content_type") or "").strip()})

    data = {
        "status": status,
        "preflight": preflight(),
        "ble": ble_status(),
        "recent_rows": recent_rows,
        "avg_speaker_analysis_s": round(sum(sa) / len(sa), 1) if sa else None,
        "tag_values": tag_values,
        "viewer_seen_s_ago": round(time.time() - _last_view["at"], 1)
                             if _last_view["at"] else None,
        "rows_in_file": len(rows),
        "results_file": results_file.name if results_file else None,
        "controlled_by_dashboard": running,
        "run_process": {
            "alive": running,
            "returncode": (_proc.poll() if _proc is not None else None),
            "foreign": foreign_run_active(),
        },
        "server_time": time.strftime("%H:%M:%S"),
    }
    if full:
        data["analysis"] = failure_analysis(rows)
        data["log"] = read_log()
        data["outputs"] = list_outputs()
    return data


# ------------------------------------------------------------- controls ----
def foreign_run_active() -> bool:
    """Is a run going that this dashboard did not start?

    A run writes its status file after every sample, so a file that says
    "running" and was touched moments ago means something is driving the phone
    right now - whether it was launched from a terminal, another dashboard, or
    a scheduled task. Starting a second one would have two processes tapping
    the same phone.
    """
    if _proc is not None and _proc.poll() is None:
        return False          # that one is ours
    if not STATUS.exists():
        return False
    try:
        state = json.loads(STATUS.read_text(encoding="utf-8")).get("state")
    except (OSError, json.JSONDecodeError):
        return False
    return state == "running" and (time.time() - STATUS.stat().st_mtime) < 90


def start_run(body: dict) -> str:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return "a run is already going"
    if foreign_run_active():
        return ("a run started outside this dashboard is already going - "
                "press Stop first, then Start or Resume")
    if not RUNNER.exists():
        return f"{RUNNER.name} not found beside the dashboard"

    pre = preflight(force=True)
    if pre["problem"]:
        return pre["problem"]

    mic = body.get("mic", "phone")
    if mic == "ble" and not ble_status(force=True)["is_connected"]:
        return "BLE microphone is not connected - connect it in the app, then start"

    cmd = [sys.executable, str(RUNNER), "run", "--mic", mic, "--no-dashboard"]
    if body.get("limit"):
        cmd += ["--limit", str(int(body["limit"]))]
    if body.get("start"):
        cmd += ["--start", str(int(body["start"]))]
    # Continuing is the runner's default, so Start has to say otherwise -
    # without this, Start and Resume did exactly the same thing.
    if body.get("resume"):
        cmd += ["--resume"]
    else:
        cmd += ["--fresh"]
    for flag, key in (("--volume", "volume"), ("--distance-cm", "distance_cm"),
                      ("--dba", "dba")):
        if body.get(key):
            cmd += [flag, str(body[key])]

    try:
        STOP_REQUEST.unlink()
    except OSError:
        pass

    log = RUN_LOG.open("w", encoding="utf-8", buffering=1)
    log.write(f"$ {' '.join(cmd[1:])}\n\n")
    kwargs: dict = {"cwd": str(HERE), "stdout": log, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    _proc = subprocess.Popen(cmd, **kwargs)

    # A run that dies on startup leaves the dashboard looking like nothing
    # happened. Give it a moment, and if it has already exited, hand back the
    # end of its log rather than a cheerful "started".
    time.sleep(1.5)
    if _proc.poll() is not None:
        log.flush()
        tail = [ln for ln in read_log_full().splitlines() if ln.strip()][-4:]
        detail = " | ".join(tail) if tail else "no output"
        return f"the run exited immediately (code {_proc.returncode}): {detail}"
    return "started: " + " ".join(cmd[2:])


def clear_stale_stop_request() -> None:
    """Remove a stop request nobody acted on, so it cannot stop the next run."""
    try:
        if STOP_REQUEST.exists():
            STOP_REQUEST.unlink()
    except OSError:
        pass


def stop_run() -> str:
    """Stop whichever run is going, including one started from a terminal.

    Two mechanisms, because the dashboard cannot signal a process it did not
    spawn: a stop-request file that any run checks before each sample, plus an
    interrupt for a run this dashboard did own. Either way the run finishes its
    current sample, flushes results and exits so it can be resumed.
    """
    try:
        STOP_REQUEST.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    except OSError as e:
        return f"could not write the stop request: {e}"

    ours = _proc is not None and _proc.poll() is None
    if ours:
        try:
            if os.name == "nt":
                _proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                _proc.send_signal(signal.SIGINT)
        except OSError:
            pass
    return "stop requested - the run saves and exits after the current sample"


# ------------------------------------------------------------------ http ---
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASR evaluation</title>
<style>
:root{
  --page:#eef1f6; --card:#fff; --head:#1f3a5f; --head-2:#2b4e7e;
  --ink:#16202e; --dim:#5f6b7d; --rule:#dde2ea;
  --green:#2e7d4f; --green-bg:#e6f4ec; --red:#b3261e; --red-bg:#fdeceb;
  --blue:#1a56a8; --blue-bg:#e8f0fb; --amber:#8a6100; --amber-bg:#fdf3dc;
  --mono:ui-monospace,Consolas,Menlo,monospace;
}
*{box-sizing:border-box} html,body{margin:0}
body{background:var(--page);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;
  padding:16px clamp(12px,2.5vw,26px) 36px;-webkit-font-smoothing:antialiased}
.num{font-variant-numeric:tabular-nums}

.bar{background:var(--head);color:#fff;border-radius:7px 7px 0 0;padding:12px 18px;
  display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.bar h1{font-size:16px;font-weight:600;margin:0;flex:1 1 auto}
.bar .meta{font-size:12.5px;color:#c3d2e6}
.chip{display:inline-flex;align-items:center;gap:7px;background:var(--head-2);
  padding:4px 11px;border-radius:999px;font-size:12.5px}
.chip i{width:8px;height:8px;border-radius:50%;background:#7fd6a4;flex:none}
.chip.off i{background:#f09a93}

.shell{background:var(--card);border:1px solid var(--rule);border-top:0;
  border-radius:0 0 7px 7px;padding:14px 18px;margin-bottom:14px}
.controls{display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap}
button{border-radius:5px;padding:8px 18px;font:600 13.5px/1 inherit;cursor:pointer;
  border:1px solid transparent}
button.start{background:var(--green-bg);border-color:#9ccfb4;color:var(--green)}
button.stop{background:var(--red-bg);border-color:#efb5b1;color:var(--red)}
button.resume{background:var(--blue-bg);border-color:#a8c4ea;color:var(--blue)}
button.plain{background:#fff;border-color:var(--rule);color:var(--ink);padding:6px 13px}
button:disabled{opacity:.45;cursor:not-allowed}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
.field{display:flex;flex-direction:column;gap:4px}
.field label{font-size:11.5px;color:var(--dim)}
select,input{background:#fff;border:1px solid var(--rule);border-radius:5px;
  padding:7px 9px;font:13.5px inherit;color:var(--ink)} input{width:84px}

.cards{display:grid;gap:10px;margin-top:14px;
  grid-template-columns:repeat(auto-fit,minmax(130px,1fr))}
.c{background:#f7f9fc;border:1px solid var(--rule);border-radius:6px;padding:10px 12px}
.c.hl{background:var(--amber-bg);border-color:#edd7a0}
.c .k{font-size:11.5px;color:var(--dim);margin-bottom:3px}
.c .v{font-size:22px;font-weight:650;letter-spacing:-.01em}
.c.hl .v{color:var(--amber)}
.track{height:5px;background:#dfe5ee;border-radius:3px;margin-top:7px;overflow:hidden}
.track>i{display:block;height:100%;background:var(--blue)}
.summary{margin-top:10px;font-size:12.5px;color:var(--dim);display:flex;gap:18px;flex-wrap:wrap}
.summary b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}

.cols2{display:grid;gap:0 26px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.panel{background:var(--card);border:1px solid var(--rule);border-radius:7px;
  padding:14px 16px;margin-bottom:14px}
h2{font-size:13.5px;font-weight:650;margin:0 0 10px;color:var(--blue)}
.sentence{font-size:14px;line-height:1.55;margin:0 0 6px;word-break:break-word}
.said{color:var(--dim)} .said.bad{color:var(--red)}
.line{display:flex;justify-content:space-between;gap:14px;padding:4px 0;
  border-bottom:1px solid var(--rule);font-size:13px}
.line:last-child{border-bottom:0}
.line span{color:var(--dim)} .line b{font-variant-numeric:tabular-nums}

table{width:100%;border-collapse:collapse;font-size:13px}
thead th{background:var(--head);color:#fff;text-align:left;font-weight:600;font-size:12px;padding:7px 9px}
thead th:first-child{border-radius:5px 0 0 0} thead th:last-child{border-radius:0 5px 0 0}
td{padding:6px 9px;border-bottom:1px solid var(--rule)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.failed{color:var(--red);font-weight:650} .ok{color:var(--green)}

details{border-top:1px solid var(--rule);padding-top:9px;margin-top:9px}
details:first-of-type{border-top:0;margin-top:0}
summary{cursor:pointer;font-size:13px;font-weight:600;list-style:none;
  display:flex;justify-content:space-between;align-items:center}
summary::-webkit-details-marker{display:none}
summary::after{content:"+";color:var(--dim);font-weight:400}
details[open] summary::after{content:"\\2013"}
pre{background:#0f1622;color:#dbe3ef;border-radius:5px;padding:11px;
  font:12px/1.55 var(--mono);max-height:420px;overflow:auto;white-space:pre-wrap;
  word-break:break-word;margin:9px 0 0}
.tagline{font-size:12px;color:var(--dim);margin:2px 0 0}
.note{font-size:12.5px;color:var(--dim);margin:9px 0 0;min-height:1em}
a{color:var(--blue)}
.logbar{display:flex;gap:8px;align-items:center;margin-top:9px}
</style></head><body>

<div class="bar">
  <h1>SmartScribe ASR Evaluation &mdash; Dashboard</h1>
  <span class="meta" id="runid">Run ID: none</span>
  <span class="chip" id="micChip"><i></i><span id="micChipText">Phone mic</span></span>
</div>

<div class="shell">
  <div class="controls">
    <button class="start" id="start">Start</button>
    <button class="stop" id="stop">Stop</button>
    <button class="resume" id="resume">Resume</button>
    <div class="field"><label for="mic">Mic mode</label>
      <select id="mic"><option value="phone">Phone</option><option value="ble">BLE</option></select></div>
    <div class="field"><label for="limit">Samples</label><input id="limit" placeholder="all"></div>
    <div class="field"><label for="startAt">Start at row</label><input id="startAt" placeholder="0"></div>
    <div class="field"><label for="volume">PC volume</label><input id="volume" placeholder="70"></div>
    <div class="field"><label for="dist">Distance cm</label><input id="dist" placeholder="20"></div>
    <div class="field"><label for="dba">Level dBA</label><input id="dba" placeholder="65"></div>
  </div>
  <p class="note" id="msg"></p>

  <div class="cards">
    <div class="c"><div class="k">Progress</div><div class="v num" id="progress">—</div>
      <div class="track"><i id="track" style="width:0"></i></div>
      <div class="tagline num" id="percent">0 %</div></div>
    <div class="c"><div class="k">Average WER</div><div class="v num" id="wer">—</div></div>
    <div class="c"><div class="k">Average CER</div><div class="v num" id="cer">—</div></div>
    <div class="c"><div class="k">SER</div><div class="v num" id="ser">—</div></div>
    <div class="c hl"><div class="k">Avg time / sample</div><div class="v num" id="rate">—</div></div>
    <div class="c hl"><div class="k">Est. time remaining</div><div class="v num" id="eta">—</div></div>
  </div>

  <div class="summary">
    <span>Status: <b id="state">idle</b></span>
    <span>Total S / D / I: <b id="sdi">—</b></span>
    <span>Completed: <b id="cScored">—</b></span>
    <span>Failed: <b id="cFailed">—</b></span>
    <span>Remaining: <b id="cLeft">—</b></span>
    <span>Elapsed: <b id="elapsed">—</b></span>
  </div>
</div>

<div class="panel">
  <h2>Current sample <span class="tagline" id="curId"></span></h2>
  <p class="sentence" id="curRef">Nothing playing yet.</p>
  <p class="sentence said" id="curHyp"></p>
  <div class="cols2">
    <div class="line"><span>Current WER</span><b id="sWer">—</b></div>
    <div class="line"><span>Current CER</span><b id="sCer">—</b></div>
    <div class="line"><span>Current S / D / I</span><b id="sSdi">—</b></div>
    <div class="line"><span>Result</span><b id="sStatus">—</b></div>
  </div>
</div>

<div class="panel">
  <h2>Details</h2>
  <details><summary>Recent results <span class="tagline" id="recentCount"></span></summary>
    <div style="overflow-x:auto">
    <table style="margin-top:9px"><thead><tr>
      <th>Sample</th><th>Status</th><th class="n">WER</th><th class="n">CER</th>
      <th class="n">S / D / I</th><th class="n">Speaker Analysis (s)</th><th>Tag / reason</th>
    </tr></thead><tbody id="recent">
      <tr><td colspan="7" style="color:var(--dim)">Nothing yet.</td></tr>
    </tbody></table></div>
  </details>
  <details id="logBox"><summary>Complete logs</summary>
        <div class="logbar">
          <button class="plain" id="logRefresh">Refresh</button>
          <a href="/api/log" download="run.log">Download run.log</a>
          <span class="tagline" id="logInfo"></span>
        </div>
        <pre id="log">Open to load the log.</pre>
      </details>
      <details><summary>Failures by cause</summary><div id="failCause"></div></details>
      <details><summary>Where the errors are</summary><div id="analysis"></div></details>
      <details><summary>Run record</summary><div id="meta"></div></details>
      <details><summary>Phone and microphone</summary><div id="checks"></div>
        <p class="note" id="preNote"></p>
        <div class="logbar"><button class="plain" id="recheck">Check again</button></div>
      </details>
  <details><summary>Final report and data</summary><div id="files"></div></details>
</div>

<script>
const $ = id => document.getElementById(id);
// Writing straight to $("x").textContent means one missing element throws and
// every field after it silently keeps its placeholder. set() degrades to a
// no-op instead, and records what was missing.
const missingIds = new Set();
function set(id, value) {
  const el = document.getElementById(id);
  if (!el) { missingIds.add(id); return; }
  el.textContent = value;
}
const esc = t => (t ?? "").toString().replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
const pct = v => (v == null || v === "") ? "—" : Number(v).toFixed(2) + " %";
const dur = s => { if (s == null) return "—"; s = Math.round(s);
  const h = Math.floor(s/3600), m = Math.floor(s%3600/60);
  return h ? `${h} h ${m} m` : (m ? `${m} m ${s%60} s` : `${s} s`); };

let fullTick = 0;

async function send(path, body) {
  const r = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
                              body: JSON.stringify(body || {})});
  $("msg").textContent = (await r.json()).message || "";
  refresh(true);
}
const opts = () => ({mic:$("mic").value, limit:$("limit").value, start:$("startAt").value,
                     volume:$("volume").value, distance_cm:$("dist").value, dba:$("dba").value});
$("start").onclick   = () => send("/api/start", {...opts(), resume:false});
$("resume").onclick  = () => send("/api/start", {...opts(), resume:true});
$("stop").onclick    = () => send("/api/stop", {});
$("recheck").onclick = () => send("/api/recheck", {});
$("mic").onchange = () => refresh(true);

// The log is fetched only when its dropdown is open: a long run writes a lot,
// and carrying it on every two-second poll would be wasteful.
async function loadLog() {
  try {
    const text = await (await fetch("/api/log")).text();
    $("log").textContent = text;
    $("logInfo").textContent = `${text.split("\\n").length} lines`;
    const box = $("log"); box.scrollTop = box.scrollHeight;
  } catch (e) { $("log").textContent = "Could not read the log."; }
}
$("logBox").addEventListener("toggle", () => { if ($("logBox").open) loadLog(); });
$("logRefresh").onclick = e => { e.preventDefault(); loadLog(); };

function recentTable(rows) {
  if (!rows.length) return `<tr><td colspan="7" style="color:var(--dim)">Nothing yet.</td></tr>`;
  return rows.slice().reverse().map(r => {
    const bad = r.status !== "ok" || !(r.hypothesis || "").trim();
    return `<tr><td>${esc(r.id)}</td>
      <td class="${bad?"failed":"ok"}">${bad ? "FAILED" : "DONE"}</td>
      <td class="n">${bad ? "-" : pct(r.WER)}</td>
      <td class="n">${bad ? "-" : pct(r.CER)}</td>
      <td class="n">${bad ? "-" : `${r.substitutions||0} / ${r.deletions||0} / ${r.insertions||0}`}</td>
      <td class="n">${r.speaker_analysis_s || "-"}</td>
      <td class="${bad?"failed":""}">${esc(bad ? (r.app_message || r.status) : r.content_type)}</td>
    </tr>`;
  }).join("");
}

function renderChecks(p, ble) {
  const isBle = $("mic").value === "ble";
  const items = [
    ["Device connected", (p.devices||[]).length > 0,
     (p.devices||[]).length ? `${p.devices[0].model} (${p.devices[0].serial})` : "none"],
    ["App installed", p.app_installed, p.app_version ? "version " + p.app_version : ""],
    ["Debug build", p.debuggable, p.debuggable ? "transcripts readable" : "cannot read transcripts"],
  ];
  if (isBle) items.push(["BLE microphone connected", !!ble.is_connected,
    ble.is_connected ? (ble.connected||[]).map(d => d.name).join(", ") : "not connected"]);
  $("checks").innerHTML = items.map(([k, ok, note]) =>
    `<div class="line"><span>${ok ? "OK" : "not ready"} &mdash; ${k}</span><b>${esc(note)}</b></div>`).join("");
  $("preNote").textContent = p.problem || (isBle && !ble.is_connected
    ? "Connect the BLE microphone in the app before starting." : "Ready to run.");
  const blocked = !!p.problem || (isBle && !ble.is_connected);
  $("start").disabled = blocked; $("resume").disabled = blocked;
}

function renderAnalysis(a) {
  if (!a || !a.groups || !Object.keys(a.groups).length) {
    $("analysis").innerHTML = `<p class="tagline">No scored samples yet.</p>`; return;
  }
  $("analysis").innerHTML = `<p class="tagline">Run average ${a.overall_wer} %</p>` +
    Object.entries(a.groups).map(([tag, groups]) => `
    <div style="margin:10px 0 14px">
      <div class="tagline" style="font-weight:600">${tag.replace(/_/g," ")}</div>
      <table style="margin-top:5px"><thead><tr><th>Value</th><th class="n">Samples</th>
        <th class="n">WER</th><th class="n">Failed</th></tr></thead><tbody>
      ${groups.map(g => `<tr><td>${esc(g.value)}</td><td class="n">${g.n}</td>
        <td class="n">${g.wer==null?"—":g.wer.toFixed(1)+" %"}</td>
        <td class="n">${g.failed||""}</td></tr>`).join("")}
      </tbody></table></div>`).join("");
}

async function refresh(full) {
  try { await draw(full); }
  catch (e) {
    $("msg").textContent = "Dashboard error: " + (e && e.message ? e.message : e)
      + (missingIds.size ? "  (missing: " + [...missingIds].join(", ") + ")" : "");
    console.error(e);
  }
}

async function draw(full) {
  let d;
  try { d = await (await fetch("/api/status?viewer=1&full=" + (full?1:0))).json(); }
  catch (e) { $("msg").textContent = "Lost contact with the dashboard server."; return; }

  const s = d.status||{}, p = s.progress||{}, c = s.cumulative||{},
        t = s.timing||{}, last = s.last_result||{};
  const state = s.state || "idle";
  const micNow = s.mic_mode || $("mic").value;

  set("runid", "Run ID: " + (s.run_id || "none"));
  const bleOk = micNow === "ble"
    ? (s.ble_connected === true || (d.ble && d.ble.is_connected)) : true;
  $("micChip").className = "chip" + (bleOk ? "" : " off");
  set("micChipText", micNow === "ble"
    ? (bleOk ? "BLE connected" : "BLE disconnected") : "Phone mic");

  // The manifest's own numbering when the run knows it, so this card and the
  // console line read the same. Falls back to the within-range count.
  set("progress", (p.row && p.manifest_total)
    ? `${p.row} / ${p.manifest_total}`
    : `${p.completed??0} / ${p.total??0}`);
  set("percent", (p.percent ?? 0).toFixed(1) + " %");
  set("inRange", (p.row && p.manifest_total)
    ? `${p.completed??0} / ${p.total??0} in this run` : "");
  $("track").style.width = (p.percent||0) + "%";
  set("wer", pct(c.WER));
  set("cer", pct(c.CER));
  set("ser", pct(c.SER));
  set("rate", t.avg_per_sample_s ? t.avg_per_sample_s.toFixed(1) + " s" : "—");
  set("eta", dur(t.eta_s));
  set("state", state);
  set("sdi", `${c.substitutions??0} / ${c.deletions??0} / ${c.insertions??0}`);
  set("cScored", p.scored ?? 0);
  set("cFailed", p.failed ?? 0);
  set("cLeft", p.remaining ?? 0);
  set("elapsed", dur(t.elapsed_s));

  const cur = s.current || {};
  set("curId", cur.id ? `sample ${cur.id}` : "");
  set("curRef", cur.reference || "Nothing playing yet.");
  if (last.id) {
    const heard = (last.hypothesis || "").trim();
    $("curHyp").textContent = heard
      ? `Heard on sample ${last.id}: ${heard}`
      : `Sample ${last.id} returned nothing (${last.status})`;
    $("curHyp").className = "sentence said" + (heard ? "" : " bad");
    set("sWer", pct(last.WER));
    set("sCer", pct(last.CER));
    set("sSdi", `${last.substitutions??0} / ${last.deletions??0} / ${last.insertions??0}`);
    set("sStatus", last.status || "—");
  }

  $("recent").innerHTML = recentTable(d.recent_rows || []);
  $("recentCount").textContent = (d.recent_rows || []).length
    ? `last ${d.recent_rows.length}` : "";

  const f = s.failures_by_status || {};
  $("failCause").innerHTML = Object.keys(f).length
    ? Object.entries(f).map(([k,v]) => `<div class="line"><span>${esc(k)}</span><b>${v}</b></div>`).join("")
    : `<p class="tagline">No failures.</p>`;
  const m = s.run_meta || {};
  $("meta").innerHTML = Object.keys(m).length
    ? Object.entries(m).map(([k,v]) =>
      `<div class="line"><span>${k.replace(/_/g," ")}</span><b>${esc(v ?? "—")}</b></div>`).join("")
    : `<p class="tagline">No run started yet.</p>`;

  renderChecks(d.preflight||{}, d.ble||{});
  $("stop").disabled = (state !== "running");

  if (full) {
    renderAnalysis(d.analysis);
    $("files").innerHTML = (d.outputs||[]).length
      ? d.outputs.map(f => `<div class="line"><span><a href="/api/download?name=${
          encodeURIComponent(f.name)}">${esc(f.name)}</a></span><b>${f.size_kb} kB</b></div>`).join("")
      : `<p class="tagline">Nothing written yet.</p>`;
    if ($("logBox").open) loadLog();
  }
}

refresh(true);
setInterval(() => { fullTick++; refresh(fullTick % 3 === 0); }, 2000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/":
            _last_view["at"] = time.time()
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif url.path == "/api/status":
            if query.get("viewer", ["0"])[0] == "1":
                _last_view["at"] = time.time()
            full = query.get("full", ["0"])[0] == "1"
            self._send(json.dumps(payload(full)).encode("utf-8"), "application/json")
        elif url.path == "/api/log":
            text = read_log_full() or "No run has been started yet."
            self._send(text.encode("utf-8"), "text/plain; charset=utf-8")
        elif url.path == "/api/download":
            name = (query.get("name") or [""])[0]
            safe = Path(name).name                   # no directory traversal
            target = next((folder / safe for folder in search_dirs()
                           if (folder / safe).is_file()), None)
            if target is None:
                return self.send_error(404)
            self._send(target.read_bytes(), "application/octet-stream",
                       {"Content-Disposition": f'attachment; filename="{target.name}"'})
        else:
            self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if url.path == "/api/start":
            msg = start_run(body)
        elif url.path == "/api/stop":
            msg = stop_run()
        elif url.path == "/api/recheck":
            preflight(force=True)
            ble_status(force=True)
            msg = "rechecked the phone connection"
        else:
            return self.send_error(404)
        self._send(json.dumps({"message": msg}).encode("utf-8"), "application/json")


def _browser_exe() -> str | None:
    """Find a Chromium-family browser, which is what supports app windows."""
    import shutil
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.append(os.path.join(local, "Google", "Chrome", "Application", "chrome.exe"))
        for path in candidates:
            if os.path.exists(path):
                return path
        return shutil.which("msedge") or shutil.which("chrome")
    if sys.platform == "darwin":
        for path in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                     "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"):
            if os.path.exists(path):
                return path
        return None
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def open_window(url: str, as_tab: bool = False) -> bool:
    """Open the dashboard in its own window rather than a browser tab.

    `--app=` gives a Chromium-family browser a plain window: no tabs, no
    address bar, its own taskbar entry. It looks and behaves like a small
    application, which is what this is. Falls back to an ordinary tab when no
    such browser is available.
    """
    if not as_tab:
        exe = _browser_exe()
        if exe:
            try:
                profile = os.path.join(tempfile.gettempdir(), "smartscribe-dashboard")
                subprocess.Popen(
                    [exe, f"--app={url}", "--window-size=1500,950",
                     f"--user-data-dir={profile}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Control panel for the ASR evaluation")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true",
                    help="start the server without opening a window")
    ap.add_argument("--tab", action="store_true",
                    help="open in a normal browser tab instead of its own window")
    cfg = ap.parse_args()

    try:
        server = ThreadingHTTPServer((cfg.host, cfg.port), Handler)
    except OSError as e:
        print(f"Could not listen on port {cfg.port}: {e}")
        print(f"If a dashboard is already open it is at http://{cfg.host}:{cfg.port}")
        print("Otherwise choose another port with --port 8766")
        return 1

    url = f"http://{cfg.host}:{cfg.port}"
    print(f"Dashboard on {url}")
    print(f"  working folder: {HERE}")
    print("  Ctrl+C stops the dashboard (a running evaluation keeps going)")

    if not cfg.no_browser:
        def open_browser() -> None:
            time.sleep(0.6)
            if not open_window(url, as_tab=cfg.tab):
                print(f"  (could not open a window - go to {url})")
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
