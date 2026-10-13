"""
SmartScribe on-device ASR + Speaker Analysis evaluation harness.

Flow per sample:
  1. Read the top entry of the app's history (baseline).
  2. Tap the mic button (start recording).
  3. Play the ground-truth WAV out loud through the PC speakers (picked up by the
     phone's microphone).
  4. Tap the mic button again (stop recording) once playback finishes (+ a short
     trailing buffer so the tail of the sentence isn't clipped).
  5. Poll the app's internal history.json (via `adb shell run-as ... cat`) until a
     new entry appears (the just-finished recording).
  6. Tap the "Speaker Analysis" button.
  7. Poll history.json until that entry's `standardizedText` is populated.
  8. Strip "Speaker N: " labels, compare against the reference text using the common semantic normalization
     and compute WER/CER/SER/OOV/S-D-I, then append the result.

Usage:
  python evaluate_asr.py calibrate            # take a screenshot to find button coords
  python evaluate_asr.py set-mic-xy X Y       # save the mic button's tap coordinates
  python evaluate_asr.py run [--limit N] [--start N] [--force] [--serial S]
  python evaluate_asr.py summarize            # recompute metrics from results CSV only

Requires: adb (Android Debug Bridge), and the Python packages jiwer, sounddevice,
soundfile (already installed in this environment).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

if sys.platform == "win32":
    # Windows consoles default to cp1252, which chokes on stray Unicode in ASR
    # output/reference text; force UTF-8 so a print() never crashes the run.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import jiwer
import numpy as np
import sounddevice as sd
import soundfile as sf
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from whisper_normalizer.english import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


_CARDINALS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}

_ORDINALS = {
    "zeroth": 0, "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
    "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}
_SCALE_WORDS = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
_NUMBER_WORDS = set(_CARDINALS) | set(_ORDINALS) | set(_SCALE_WORDS) | {"and"}


def _parse_number_phrase(words: list[str]) -> int | None:
    """Parse a contiguous cardinal/ordinal number phrase with scales."""
    if not words or not all(w in _NUMBER_WORDS for w in words):
        return None
    total = 0
    current = 0
    saw_scale = False
    for w in words:
        if w == "and":
            continue
        if w in _ORDINALS:
            current += _ORDINALS[w]
        elif w in _CARDINALS:
            current += _CARDINALS[w]
        else:
            saw_scale = True
            scale = _SCALE_WORDS[w]
            if current == 0:
                current = 1
            if scale == 100:
                current *= 100
            else:
                total += current * scale
                current = 0
    if not saw_scale:
        if len(words) == 1:
            return current
        values = [(_ORDINALS[w] if w in _ORDINALS else _CARDINALS.get(w))
                  for w in words if w != "and"]
        if len(values) == 2 and values[0] >= 20 and values[1] < 10:
            return values[0] + values[1]  # twenty six -> 26
        if len(values) == 3 and values[0] >= 20 and values[1] >= 20 and values[2] < 10:
            return values[0] * 100 + values[1] + values[2]  # twenty twenty six -> 2026
        # Do not collapse "eight twenty" into 28; in speech this commonly
        # represents a time such as 8:20.
        return None
    return total + current


def _canonicalize_number_words(text: str) -> str:
    tokens = text.lower().split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Numeric time, e.g. 8:20 -> 8 20 so it can match "eight twenty".
        m = re.fullmatch(r"(\d{1,2}):(\d{1,2})", tok)
        if m:
            out.extend([str(int(m.group(1))), str(int(m.group(2)))])
            i += 1
            continue
        # Numeric ordinal, e.g. 15th -> 15.
        m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", tok)
        if m:
            out.append(str(int(m.group(1))))
            i += 1
            continue
        if tok in _ORDINALS:
            out.append(str(_ORDINALS[tok]))
            i += 1
            continue
        if tok in _NUMBER_WORDS:
            j = i
            while j < len(tokens) and tokens[j] in _NUMBER_WORDS:
                j += 1
            phrase = tokens[i:j]
            parsed = _parse_number_phrase(phrase)
            if parsed is not None:
                out.append(str(parsed))
                i = j
                continue
            # No scale: canonicalize each simple number separately. This makes
            # "eight twenty" match "8:20" rather than becoming "28".
            consumed = False
            for w in phrase:
                if w == "and":
                    continue
                if w in _ORDINALS:
                    out.append(str(_ORDINALS[w]))
                elif w in _CARDINALS:
                    out.append(str(_CARDINALS[w]))
                else:
                    out.append(w)
                consumed = True
            if consumed:
                i = j
                continue
        out.append(tok)
        i += 1
    return " ".join(out)


def normalize(text: str) -> str:
    """Canonical ASR comparison form.

    Uses Whisper's English normalizer for case/punctuation, then applies a
    semantic number normalizer so numeric and spoken forms compare equally:
    15 == fifteenth, 2026 == two thousand and twenty six, and 8:20 == eight
    twenty. This normalized text is used consistently for WER, CER, SER and
    S/D/I alignment counts.
    """
    base = _normalizer(text or "")
    return _canonicalize_number_words(base)

# ─── Config ──────────────────────────────────────────────────────────────────

# Path to adb. Taken from the ADB_PATH environment variable so the repository
# holds no one person's machine layout; falls back to "adb" on PATH.
ADB = os.environ.get("ADB_PATH") or shutil.which("adb") or "adb"
PACKAGE = "com.example.speechtotext"
ACTIVITY = f"{PACKAGE}/.MainActivity"

DATASET_DIR = Path(__file__).resolve().parent

# Dataset versions are numbered, and a new one exists because the old one was
# wrong - so the highest version wins rather than the first name on a list.
# --manifest overrides everything.
MANIFEST_CANDIDATES = ["dataset_v<N>.csv", "metadata_balanced.csv", "manifest.csv"]


def default_manifest() -> Path:
    versioned = sorted(
        DATASET_DIR.glob("dataset_v*.csv"),
        key=lambda p: (int(m.group(1)) if (m := re.search(r"_v(\d+)", p.stem)) else -1),
        reverse=True,
    )
    if versioned:
        return versioned[0]
    for name in ("metadata_balanced.csv", "manifest.csv"):
        candidate = DATASET_DIR / name
        if candidate.exists():
            return candidate
    return DATASET_DIR / "manifest.csv"


MANIFEST_PATH = default_manifest()
CALIBRATION_JSON = DATASET_DIR / "calibration.json"
SCREENSHOT_PATH = DATASET_DIR / "calibration_screenshot.png"

# Every `run` writes to its OWN new, timestamped results file rather than
# overwriting the last one -- past runs are never destroyed. These two globals
# point at whichever file is "current": `cmd_run` mints a fresh path per run;
# standalone `full-metrics`/`summarize` resolve to the most recently written
# one unless told otherwise. Read/written via `global` in the functions below.
RESULTS_CSV = DATASET_DIR / "results_detailed.csv"       # legacy pre-timestamp name, kept as a fallback
SUMMARY_JSON = DATASET_DIR / "summary.json"


def new_results_csv_path(ts: str) -> Path:
    return DATASET_DIR / f"results_detailed_{ts}.csv"


def new_full_metrics_csv_path(ts: str) -> Path:
    return DATASET_DIR / f"full_metrics_{ts}.csv"


def all_results_csvs() -> list[Path]:
    candidates = list(DATASET_DIR.glob("results_detailed_*.csv"))
    legacy = DATASET_DIR / "results_detailed.csv"
    if legacy.exists():
        candidates.append(legacy)
    return candidates


def latest_results_csv() -> Path | None:
    candidates = all_results_csvs()
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_results_csv(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    latest = latest_results_csv()
    if latest is None:
        raise SystemExit(
            f"No results files found in {DATASET_DIR} (looked for results_detailed_*.csv). Run `run` first."
        )
    return latest

MAX_JOIN_SEGMENTS = 8          # more than this from one clip means something is wrong
MAX_HYP_WORD_RATIO = 5.0       # a transcript this much longer than the reference is not a transcript

PRE_RECORD_SETTLE_S = 0.8      # wait after tapping mic-on, before playing audio
POST_RECORD_TAIL_S = 0.8       # wait after audio ends, before tapping mic-off
POLL_INTERVAL_S = 0.5
NEW_ENTRY_TIMEOUT_S = 15.0     # waiting for the stopped recording to appear in history
ANALYSIS_TIMEOUT_S = 90.0      # waiting for speaker analysis to finish
INTER_SAMPLE_DELAY_S = 0.3

# Filled by cmd_run. Stored beside the results so a number from six months ago
# can still be explained: which app build, which dataset, which microphone, and
# the acoustic conditions it was measured under.
RUN_META: dict = {}

# Anchored to the line start, this only caught the first label when the app
# returned a whole batch on one line - "Speaker 2:" in the middle was then
# scored as two words the reference never contained. Matches anywhere now,
# but still requires a number, so "please reduce the speaker volume" is safe.
SPEAKER_LABEL_RE = re.compile(r"\bspeakers?\s*#?\s*\d+\s*[:\-\u2013\u2014]?\s*", re.I)

# ─── ADB helpers ─────────────────────────────────────────────────────────────


def _adb_base(serial: str | None) -> list[str]:
    base = [ADB]
    if serial:
        base += ["-s", serial]
    return base


ADB_RETRIES = 2            # attempts after the first, for transient failures


def _adb_once(args: list[str], serial: str | None, timeout: float):
    return subprocess.run(_adb_base(serial) + args, capture_output=True,
                          text=True, timeout=timeout)


def restart_adb_server(serial: str | None) -> bool:
    """Kill and restart the ADB server, then wait for the phone to reappear.

    A dropped cable or a sleeping USB port leaves the server holding a dead
    handle: every later command fails until it is restarted. Doing this once
    automatically is the difference between a run losing one sample and losing
    the rest of the night.
    """
    print("  [adb] connection lost; restarting the ADB server...")
    try:
        subprocess.run([ADB, "kill-server"], capture_output=True, timeout=20)
        time.sleep(1.0)
        subprocess.run([ADB, "start-server"], capture_output=True, timeout=30)
        wait = ["wait-for-device"]
        subprocess.run(_adb_base(serial) + wait, capture_output=True, timeout=45)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  [adb] restart failed: {e}")
        return False
    back = serial in list_devices() if serial else bool(list_devices())
    print("  [adb] device is back." if back else "  [adb] device did not come back.")
    return back


def adb(args: list[str], serial: str | None = None, timeout: float = 30.0) -> str:
    """Run one adb command, retrying transient failures.

    A single failed command is usually a blip - a busy device, a slow dump.
    A command that fails because the device is gone needs the server
    restarting, which is tried once before giving up.
    """
    last = ""
    for attempt in range(ADB_RETRIES + 1):
        try:
            result = _adb_once(args, serial, timeout)
        except subprocess.TimeoutExpired:
            last = f"timed out after {timeout:.0f}s"
            result = None
        if result is not None and result.returncode == 0:
            return result.stdout
        if result is not None:
            last = result.stderr.strip() or f"exit code {result.returncode}"

        gone = any(w in last.lower() for w in
                   ("device not found", "device offline", "no devices",
                    "device unauthorized", "closed", "protocol fault"))
        if attempt < ADB_RETRIES:
            if gone:
                if not restart_adb_server(serial):
                    break
            else:
                time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"adb {' '.join(args)} failed: {last}")


def adb_shell(shell_cmd: str, serial: str | None = None, timeout: float = 30.0) -> str:
    return adb(["shell", shell_cmd], serial=serial, timeout=timeout)


def list_devices() -> list[str]:
    out = subprocess.run([ADB, "devices"], capture_output=True, text=True).stdout
    devices = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if line and "\tdevice" in line:
            devices.append(line.split("\t")[0])
    return devices


def ensure_single_device(serial: str | None) -> str:
    devices = list_devices()
    if not devices:
        raise SystemExit("No adb device found. Connect the phone (USB debugging) and retry.")
    if serial:
        if serial not in devices:
            raise SystemExit(f"Device {serial} not found. Connected: {devices}")
        return serial
    if len(devices) > 1:
        raise SystemExit(f"Multiple devices connected ({devices}); pass --serial to pick one.")
    return devices[0]


def tap(x: int, y: int, serial: str | None = None) -> None:
    adb_shell(f"input tap {x} {y}", serial=serial)


def is_overlay_open(serial: str | None) -> bool:
    """The session-detail overlay is full-screen, so the RECORD & VAD / SESSIONS tab
    header (always present otherwise) disappears while it's open."""
    root = dump_ui_xml(serial)
    texts = {node.get("text", "").strip() for node in root.iter("node")}
    return "RECORD & VAD" not in texts and "SESSIONS" not in texts


def close_session_overlay(serial: str | None) -> None:
    """Tapping Speaker Analysis on the home screen opens the full-screen session-detail
    overlay (it sets viewModel.selectedTranscription, which covers the mic button). Only
    call this when that overlay is actually known to be open -- pressing BACK when it's
    NOT open would exit/minimize the app instead (no other back-stack entry)."""
    adb_shell("input keyevent 4", serial=serial)  # KEYCODE_BACK
    time.sleep(0.3)


def launch_app(serial: str | None) -> None:
    """Force-stop first: MainActivity has no launchMode set (defaults to
    'standard'), so repeated `am start` calls stack up new Activity instances
    instead of reusing one -- this caused the intermittent good/bad flakiness
    across repeated test runs. Force-stopping guarantees exactly one clean
    instance every time."""
    adb_shell(f"am force-stop {PACKAGE}", serial=serial)
    time.sleep(0.5)
    adb_shell(f"am start -n {ACTIVITY}", serial=serial)
    time.sleep(2.5)


def keep_screen_awake(serial: str | None) -> None:
    """A phone whose screen locks mid-run (timeout, or a PIN lock) breaks every
    tap silently for the rest of the batch with no error -- an overnight run
    once lost 852 consecutive samples this way. `stayon` keeps the screen on
    for as long as the phone stays on USB power, same as the 'stay awake while
    charging' developer option, without needing to touch phone settings."""
    try:
        adb_shell("svc power stayon true", serial=serial)
    except RuntimeError:
        pass


def is_on_app_home_screen(serial: str | None) -> bool:
    """Ground truth: are we actually looking at the app's RECORD & VAD screen
    (mic button visible, idle)? This checks for the app's own UI directly,
    rather than inferring from `dumpsys window` text -- that string format
    varies across Android versions/OEMs and was silently wrong on a device
    where the lock-screen detection never fired despite the screen being
    locked. 'PHONE MIC' is unique to this tab (not shown on Sessions)."""
    try:
        root = dump_ui_xml(serial)
    except Exception:
        return False
    texts = {n.get("text", "").strip() for n in root.iter("node")}
    return "RECORD & VAD" in texts and "PHONE MIC" in texts


def ensure_screen_ready(serial: str | None) -> bool:
    """Escalating recovery when the app's home screen isn't showing: wake the
    display, then fall back to a full app relaunch (`am start -n <activity>`
    names the exact component, so it reliably brings SmartScribe back
    regardless of what's currently in front -- a crash, an ANR dialog,
    another app, or a plain screen-off). Deliberately does NOT attempt a
    swipe gesture to dismiss a lock screen: on gesture-navigation Android, a
    vertical swipe starting near the bottom edge is the same physical motion
    as the system's home/recent-apps gesture, and was observed switching to a
    completely different app instead of unlocking anything -- worse than
    doing nothing. Verifies success against the app's actual UI at every
    step instead of guessing. Returns False only if nothing brings the home
    screen back (e.g. a genuine PIN/pattern lock, which cannot be bypassed
    programmatically -- keep the phone's screen lock set to 'None' for
    unattended runs)."""
    if is_on_app_home_screen(serial):
        return True

    try:
        power = adb_shell("dumpsys power", serial=serial)
        if "mWakefulness=Awake" not in power:
            adb_shell("input keyevent 224", serial=serial)  # KEYCODE_WAKEUP
            time.sleep(0.5)
    except RuntimeError:
        pass
    if is_on_app_home_screen(serial):
        return True

    launch_app(serial)
    keep_screen_awake(serial)
    if is_overlay_open(serial):
        close_session_overlay(serial)
    return is_on_app_home_screen(serial)


def ble_stack_devices(serial: str | None) -> list[str]:
    """Devices the phone's Bluetooth stack reports as connected.

    Only a partial answer: that dump lists classic profile connections (audio,
    headset) reliably, but a BLE GATT link opened by an app frequently does not
    appear in it at all. Useful as corroboration, not as the decider.
    """
    try:
        dump = adb_shell("dumpsys bluetooth_manager", serial=serial, timeout=20.0)
    except RuntimeError:
        return []
    names, section = [], None
    for raw in dump.splitlines():
        line = raw.strip()
        low = line.lower()
        if "connected devices:" in low or "connected gatt devices" in low:
            section = "connected"
            continue
        if "bonded devices:" in low or low.startswith("gatt server"):
            section = None
            continue
        m = re.match(r"([0-9A-F]{2}(?::[0-9A-F]{2}){5})\s*(.*)", line, re.I)
        if m and section == "connected":
            names.append(f"{m.group(2).strip() or '(unnamed)'}  {m.group(1)}")
        elif not line:
            section = None
    return names


def ble_connected_in_app(serial: str | None) -> tuple[bool, str]:
    """Ask the app, since the app's own view decides whether it will record.

    Looks for the states only a connected screen shows: a Disconnect control,
    the word Connected, or a device name sitting where the scan list was.
    Returns (connected, what was seen) so the reason can be shown rather than
    just a yes or no.
    """
    try:
        root = dump_ui_xml(serial)
    except Exception as e:
        return False, f"could not read the screen ({e})"

    labels = []
    for node in root.iter("node"):
        txt = (node.get("text", "") + " " + node.get("content-desc", "")).strip()
        if txt:
            labels.append(txt)
    joined = " | ".join(labels).upper()

    if "DISCONNECT" in joined:
        return True, "the app is showing a Disconnect control"
    m = re.search(r"CONNECTED(?:\s*(?:TO|:)?\s*([^|]{0,40}))?", joined)
    if m and "NOT CONNECTED" not in joined and "DISCONNECTED" not in joined:
        detail = (m.group(1) or "").strip(" :-")
        return True, f"the app is showing Connected{(' to ' + detail) if detail else ''}"
    return False, "no connected state visible on the app screen"


def ble_connected_devices(serial: str | None) -> list[str]:
    """Kept for callers that just want a list; prefers the stack, falls back to
    what the app is showing."""
    stack = ble_stack_devices(serial)
    if stack:
        return stack
    ok, detail = ble_connected_in_app(serial)
    return [detail] if ok else []


def choose_mic(serial: str | None, preset: str | None) -> str | None:
    """Ask which microphone this run uses, unless --mic already said.

    Returns None if the person backs out, so the caller can stop cleanly rather
    than defaulting to the phone mic and quietly measuring the wrong thing.
    """
    mic = preset
    if mic is None:
        if not sys.stdin.isatty():
            return "phone"          # scripted run, no one to ask
        print("\nWhich microphone should this run use?")
        print("  1) Phone microphone")
        print("  2) BLE microphone")
        while True:
            try:
                choice = input("Choose 1 or 2 [1]: ").strip() or "1"
            except (EOFError, KeyboardInterrupt):
                print()
                return None
            if choice in ("1", "phone"):
                mic = "phone"
                break
            if choice in ("2", "ble"):
                mic = "ble"
                break
            print("  Type 1 or 2.")

    if mic == "phone":
        print("Using the phone microphone.")
        return mic

    def ble_report() -> tuple[bool, str]:
        stack = ble_stack_devices(serial)
        in_app, detail = ble_connected_in_app(serial)
        lines = []
        lines.append(f"    Bluetooth stack : {'; '.join(stack) if stack else 'nothing listed'}")
        lines.append(f"    App screen      : {detail}")
        return (bool(stack) or in_app), "\n".join(lines)

    ok, report = ble_report()
    if ok:
        print("BLE microphone connected.")
        print(report)
        return mic

    print("\nOpening SmartScribe and scanning for BLE devices...")
    open_ble_scan(serial)

    while True:
        ok, report = ble_report()
        if ok:
            print("\nBLE microphone connected.")
            print(report)
            return mic

        print("\nNot seeing a connection yet:")
        print(report)
        found = ble_names_on_screen(serial)
        if found:
            print("  Devices listed in the app:")
            for name in found[:12]:
                print(f"    {name}")
            print("  Tap the one you want on the phone to connect it.")

        print("\n  A BLE link opened by the app does not always show up in the")
        print("  phone's Bluetooth dump. If the app itself says it is connected,")
        print("  press c to go ahead.")
        try:
            again = input("Enter to re-check, s to scan again, c to continue anyway, q to quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if again == "q":
            return None
        if again == "c":
            print("Continuing with the BLE microphone on your confirmation.")
            return mic
        if again == "s":
            open_ble_scan(serial)


def is_recording(serial: str | None) -> bool:
    """The app runs a foreground service while it records, so its presence in
    the service dump is the cheapest honest answer. Audio played into a phone
    that never started recording comes back as an empty transcript, which looks
    exactly like a total ASR failure -- worth one adb call to rule out."""
    try:
        out = adb_shell(f"dumpsys activity services {PACKAGE}", serial=serial, timeout=15.0)
    except RuntimeError:
        return False
    return "TranscriptionService" in out and "isForeground=true" in out


def grant_record_audio(serial: str | None) -> None:
    try:
        adb_shell(f"pm grant {PACKAGE} android.permission.RECORD_AUDIO", serial=serial)
    except RuntimeError:
        pass  # already granted, or not a runtime permission on this API level


def app_version(serial: str | None) -> str:
    try:
        out = adb_shell(f"dumpsys package {PACKAGE} | grep versionName",
                        serial=serial, timeout=15.0)
    except RuntimeError:
        return "unknown"
    for line in out.splitlines():
        if "versionName" in line:
            return line.strip().split("=", 1)[-1].strip() or "unknown"
    return "unknown"


def device_model(serial: str | None) -> str:
    try:
        return adb_shell("getprop ro.product.model", serial=serial, timeout=15.0).strip()
    except RuntimeError:
        return "unknown"


MIN_BATTERY_PCT = 20
MIN_PHONE_FREE_MB = 500
MIN_PC_FREE_MB = 1000


def phone_battery_pct(serial: str | None) -> int | None:
    try:
        out = adb_shell("dumpsys battery", serial=serial, timeout=15.0)
    except RuntimeError:
        return None
    m = re.search(r"^\s*level:\s*(\d+)", out, re.MULTILINE)
    return int(m.group(1)) if m else None


def phone_free_mb(serial: str | None) -> int | None:
    """Free space in the app's own data partition. The app keeps a copy of
    every recording, so a long run fills the phone even though the PC is fine."""
    try:
        out = adb_shell("df /data", serial=serial, timeout=15.0)
    except RuntimeError:
        return None
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4 and parts[3].isdigit():
            return int(parts[3]) // 1024          # df reports 1K blocks
    return None


def pc_free_mb() -> int | None:
    try:
        import shutil
        return shutil.disk_usage(DATASET_DIR).free // (1024 * 1024)
    except Exception:
        return None


def check_resources(serial: str | None, loud: bool = True) -> str | None:
    """Returns a reason to stop, or None if there is room to carry on."""
    battery = phone_battery_pct(serial)
    phone_free = phone_free_mb(serial)
    pc_free = pc_free_mb()

    if loud:
        parts = []
        if battery is not None:
            parts.append(f"battery {battery}%")
        if phone_free is not None:
            parts.append(f"phone free {phone_free} MB")
        if pc_free is not None:
            parts.append(f"PC free {pc_free} MB")
        if parts:
            print("Resources: " + ", ".join(parts))

    if battery is not None and battery < MIN_BATTERY_PCT:
        return (f"phone battery is {battery}%, below the {MIN_BATTERY_PCT}% limit. "
                "Charge it, then resume.")
    if phone_free is not None and phone_free < MIN_PHONE_FREE_MB:
        return (f"the phone has {phone_free} MB free, below the "
                f"{MIN_PHONE_FREE_MB} MB limit. Clear saved sessions in the app, "
                "then resume.")
    if pc_free is not None and pc_free < MIN_PC_FREE_MB:
        return (f"the PC has {pc_free} MB free, below the {MIN_PC_FREE_MB} MB "
                "limit. Free some space, then resume.")
    return None


def check_debuggable(serial: str | None) -> None:
    out = adb_shell(f"run-as {PACKAGE} id", serial=serial)
    if "not debuggable" in out.lower() or "package not found" in out.lower():
        raise SystemExit(
            f"'adb shell run-as {PACKAGE}' failed ({out.strip()!r}). "
            "Install the DEBUG build variant, not release."
        )


# ─── UI location (Speaker Analysis button found by its visible text) ────────


def dump_ui_xml(serial: str | None) -> ET.Element:
    adb_shell("uiautomator dump /sdcard/window_dump.xml", serial=serial)
    local = DATASET_DIR / "_window_dump.xml"
    adb(["pull", "/sdcard/window_dump.xml", str(local)], serial=serial)
    tree = ET.parse(local)
    return tree.getroot()


def _bounds_center(bounds: str) -> tuple[int, int]:
    # bounds looks like "[x1,y1][x2,y2]"
    nums = [int(n) for n in re.findall(r"-?\d+", bounds)]
    x1, y1, x2, y2 = nums
    return (x1 + x2) // 2, (y1 + y2) // 2


# Which button produces the text being scored, and which stored field it
# lands in. Speaker Analysis writes standardizedText; a plain Transcript button
# may write any of these, so the fields are tried in turn and whichever fills
# in first is used.
ANALYSIS_MODES = {
    "speaker": {"labels": ["SPEAKER ANALYSIS"],
                "fields": ["standardizedText"]},
    # The transcript button may write a new field, or it may simply display
    # text the app already saved when recording stopped. In the second case
    # nothing changes after the press, so waiting for a change would hang. Here
    # the saved transcript is accepted as the answer once the grace period
    # passes without a new field appearing.
    "transcript": {"labels": ["TRANSCRIPT", "TRANSCRIPTION", "FULL TRANSCRIPT",
                              "GET TRANSCRIPT", "SHOW TRANSCRIPT"],
                   "fields": ["transcriptText", "fullTranscript", "refinedText",
                              "standardizedText"],
                   "fallback_field": "text",
                   "grace_s": 25.0},
}
ANALYSIS_MODE = "speaker"          # overridden by --analysis


def analysis_labels() -> list[str]:
    return ANALYSIS_MODES[ANALYSIS_MODE]["labels"]


def analysis_fields() -> list[str]:
    return ANALYSIS_MODES[ANALYSIS_MODE]["fields"]


def analysis_fallback() -> str | None:
    """Field to accept unchanged if the button produced nothing new."""
    return ANALYSIS_MODES[ANALYSIS_MODE].get("fallback_field")


def analysis_grace() -> float:
    """How long to wait for a new field before using the fallback."""
    return float(ANALYSIS_MODES[ANALYSIS_MODE].get("grace_s", 0.0))


def find_analysis_button_xy(serial: str | None,
                            labels: list[str] | None = None) -> tuple[int, int]:
    """Centre of the button that produces the text we score."""
    labels = [lab.strip().upper() for lab in (labels or analysis_labels())]
    root = dump_ui_xml(serial)
    parent_map = {child: parent for parent in root.iter() for child in parent}
    target = None
    for wanted in labels:                      # exact first, then contains
        for node in root.iter("node"):
            if node.get("text", "").strip().upper() == wanted:
                target = node
                break
        if target is not None:
            break
    if target is None:
        for node in root.iter("node"):
            label = (node.get("text", "") + " " + node.get("content-desc", "")).strip().upper()
            if label and any(w in label for w in labels):
                target = node
                break
    if target is None:
        raise RuntimeError(
            f"Could not find a {' or '.join(labels)} button on screen. "
            "Make sure the app is on the home (RECORD & VAD) tab and not "
            "mid-recording or mid-analysis."
        )
    node = target
    while node is not None and node.get("clickable") != "true":
        node = parent_map.get(node)
    if node is None:
        raise RuntimeError(f"Found {labels[0]} text but no clickable ancestor.")
    return _bounds_center(node.get("bounds"))


def find_speaker_analysis_xy(serial: str | None) -> tuple[int, int]:
    """Kept for callers that specifically want Speaker Analysis."""
    return find_analysis_button_xy(serial, ANALYSIS_MODES["speaker"]["labels"])


# Messages the app shows instead of a transcript, mapped to a status worth
# reading in the report.
APP_MESSAGES = [
    (re.compile(r"speech\s+not\s+recognis|speech\s+not\s+recogniz", re.I), "speech_not_recognised"),
    (re.compile(r"no\s+speech\s+(detected|found)", re.I), "no_speech_detected"),
    (re.compile(r"(didn.?t|did\s+not)\s+catch\s+that", re.I), "speech_not_recognised"),
    (re.compile(r"too\s+short", re.I), "recording_too_short"),
    (re.compile(r"microphone\s+(permission|unavailable|in\s+use)", re.I), "microphone_unavailable"),
    (re.compile(r"(model|engine)\s+(not\s+ready|loading)", re.I), "engine_not_ready"),
    (re.compile(r"connect\s+the\s+ble", re.I), "ble_not_connected"),
    (re.compile(r"error|failed", re.I), "app_error"),
]


def app_message_on_screen(serial: str | None) -> str | None:
    """Read any explanatory message the app is showing, as a status string."""
    try:
        root = dump_ui_xml(serial)
    except Exception:
        return None
    blob = " | ".join(
        (n.get("text", "") + " " + n.get("content-desc", "")).strip()
        for n in root.iter("node")
        if (n.get("text") or n.get("content-desc"))
    )
    for pattern, status in APP_MESSAGES:
        if pattern.search(blob):
            return status
    return None


def find_text_xy(serial: str | None, *candidates: str,
                 root: ET.Element | None = None) -> tuple[int, int] | None:
    """Centre of the first clickable element whose text or description matches
    one of `candidates` (case-insensitive, substring). Returns None if absent,
    so callers can decide whether that is a problem."""
    root = root if root is not None else dump_ui_xml(serial)
    parents = {child: parent for parent in root.iter() for child in parent}
    wanted = [c.strip().upper() for c in candidates]
    for node in root.iter("node"):
        label = (node.get("text", "") + " " + node.get("content-desc", "")).strip().upper()
        if not label:
            continue
        if any(w in label for w in wanted):
            n = node
            while n is not None and n.get("clickable") != "true":
                n = parents.get(n)
            if n is not None and n.get("bounds"):
                return _bounds_center(n.get("bounds"))
    return None


def ble_names_on_screen(serial: str | None) -> list[str]:
    """Device names the app is currently listing. Read straight off the screen
    because the scan results live in the app, not in the phone's BT stack."""
    root = dump_ui_xml(serial)
    skip = {"BLE MIC", "PHONE MIC", "SCAN", "SCANNING", "SCAN DEVICES", "CONNECT",
            "DISCONNECT", "RECORD & VAD", "SESSIONS", "SMARTSCRIBE",
            "LIVE OPUS VAD TRANSCRIPTION", "SPEAKER ANALYSIS", "REFINE WITH AI"}
    out = []
    for node in root.iter("node"):
        text = node.get("text", "").strip()
        if not text or text.upper() in skip or len(text) > 48:
            continue
        if re.fullmatch(r"[\d:.\s%/-]+", text):      # timers, counters
            continue
        if node.get("clickable") == "true" or re.search(r"[0-9A-F]{2}:[0-9A-F]{2}", text, re.I):
            if text not in out:
                out.append(text)
    return out


def open_ble_scan(serial: str | None) -> bool:
    """Put the app on the BLE microphone source and start a scan, so the list
    of nearby devices is on screen ready to be tapped.

    cmd_run has already launched the app by this point, so relaunching would
    force-stop a working screen and throw away any connection the person had
    already made. Only start it if it is not already in front of us.
    """
    root = None
    try:
        root = dump_ui_xml(serial)
        texts = {n.get("text", "").strip() for n in root.iter("node")}
        showing = "PHONE MIC" in texts or "BLE MIC" in texts
    except Exception:
        showing = False

    if not showing:
        launch_app(serial)
        time.sleep(1.0)
        root = None

    ble_xy = find_text_xy(serial, "BLE MIC", root=root)
    if ble_xy is None:
        print("  Could not find the BLE MIC selector on screen.")
        print("  Open SmartScribe on the RECORD & VAD tab and try again.")
        return False
    tap(*ble_xy, serial=serial)
    time.sleep(1.2)

    scan_xy = find_text_xy(serial, "SCAN")   # fresh dump: the tap changed the screen
    if scan_xy is None:
        print("  Switched to BLE MIC. No scan button found - the app may scan "
              "automatically.")
        return True
    tap(*scan_xy, serial=serial)
    print("  Scanning for nearby devices…")
    time.sleep(5.0)          # the app's scan window
    return True


# ─── Mic button (auto-detected -- no per-phone calibration needed) ─────────
#
# The mic button has no text/content-desc, so it can't be found by text search
# the way Speaker Analysis is. Instead: it's the only roughly-square clickable
# element sitting directly above the (text-findable) Speaker Analysis button --
# that layout relationship holds on any screen resolution/density, so this
# works across phones without manual setup.


def find_mic_xy(serial: str | None) -> tuple[int, int]:
    root = dump_ui_xml(serial)
    parent_map = {child: parent for parent in root.iter() for child in parent}

    screen_node = next(root.iter("node"), None)
    screen_nums = [int(n) for n in re.findall(r"-?\d+", screen_node.get("bounds", ""))] if screen_node is not None else []
    screen_w = (screen_nums[2] - screen_nums[0]) if len(screen_nums) == 4 else 1080  # sane fallback

    # The mic button is found by its position relative to the analysis button
    # below it. Which button that is depends on the run: a transcript run may
    # not have SPEAKER ANALYSIS on screen at all, so every known analysis label
    # is accepted as the anchor.
    anchors = []
    for mode in ANALYSIS_MODES.values():
        anchors.extend(mode["labels"])
    anchors = [a.upper() for a in dict.fromkeys(anchors)]

    sa_text_node = None
    for wanted in anchors:
        for node in root.iter("node"):
            if node.get("text", "").strip().upper() == wanted:
                sa_text_node = node
                break
        if sa_text_node is not None:
            break
    if sa_text_node is None:
        raise RuntimeError(
            f"Could not find any of {', '.join(anchors[:4])} on screen to anchor "
            "mic-button detection. Make sure the app is on the home "
            "(RECORD & VAD) tab, not mid-recording or mid-analysis."
        )
    sa_node = sa_text_node
    while sa_node is not None and sa_node.get("clickable") != "true":
        sa_node = parent_map.get(sa_node)
    if sa_node is None:
        raise RuntimeError("Found the analysis button text but no clickable ancestor.")
    sa_top = [int(n) for n in re.findall(r"-?\d+", sa_node.get("bounds"))][1]

    # The mic button is an 80dp circle; on a typical ~360-430dp-wide phone that's
    # roughly 18-22% of screen width. Using a width FRACTION (not fixed pixels)
    # keeps this correct from a cheap 240px-wide screen to a 1440px-wide flagship.
    min_w, max_w = 0.12 * screen_w, 0.40 * screen_w

    candidates = []
    for node in root.iter("node"):
        if node.get("clickable") != "true":
            continue
        nums = [int(n) for n in re.findall(r"-?\d+", node.get("bounds", ""))]
        if len(nums) != 4:
            continue
        x1, y1, x2, y2 = nums
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0 or y2 > sa_top:
            continue
        aspect = w / h
        if not (0.7 <= aspect <= 1.4):  # roughly square/circular, not a wide row
            continue
        if not (min_w <= w <= max_w):
            continue
        candidates.append((sa_top - y2, x1, y1, x2, y2))  # sort key: gap above SA button

    if not candidates:
        raise RuntimeError(
            "Could not locate the mic button (no square clickable element found "
            "above Speaker Analysis)."
        )
    candidates.sort(key=lambda c: c[0])
    _, x1, y1, x2, y2 = candidates[0]
    return (x1 + x2) // 2, (y1 + y2) // 2


def load_calibration() -> dict:
    if CALIBRATION_JSON.exists():
        return json.loads(CALIBRATION_JSON.read_text())
    return {}


def save_calibration(data: dict) -> None:
    CALIBRATION_JSON.write_text(json.dumps(data, indent=2))


def get_mic_xy(serial: str | None) -> tuple[int, int]:
    """Auto-detect first (works on any phone); fall back to a manually saved
    override from `set-mic-xy` only if auto-detection fails."""
    try:
        xy = find_mic_xy(serial)
        print(f"Mic button auto-detected at {xy}.")
        return xy
    except RuntimeError as e:
        cal = load_calibration()
        if "mic_x" in cal and "mic_y" in cal:
            print(f"Auto-detection failed ({e}); using saved override {cal['mic_x'], cal['mic_y']}.")
            return cal["mic_x"], cal["mic_y"]
        raise SystemExit(
            f"Could not auto-detect the mic button: {e}\n"
            "As a fallback, run `python evaluate_asr.py calibrate`, find its pixel "
            "center in the screenshot, then `python evaluate_asr.py set-mic-xy X Y`."
        )


def cmd_calibrate(serial: str | None) -> None:
    ensure_single_device(serial)
    launch_app(serial)
    raw = subprocess.run(
        _adb_base(serial) + ["exec-out", "screencap", "-p"],
        capture_output=True,
        timeout=30,
    ).stdout
    SCREENSHOT_PATH.write_bytes(raw)
    size_out = adb_shell("wm size", serial=serial).strip()
    print(f"Screenshot saved to: {SCREENSHOT_PATH}")
    print(f"Device screen size: {size_out}")
    print("Open the screenshot, find the mic button's center pixel (x, y), then run:")
    print("  python evaluate_asr.py set-mic-xy X Y")
    try:
        sa_xy = find_speaker_analysis_xy(serial)
        print(f"(Speaker Analysis button auto-detected at: {sa_xy} -- no calibration needed for it)")
    except RuntimeError as e:
        print(f"(Speaker Analysis button not detected right now: {e})")


def cmd_set_mic_xy(x: int, y: int) -> None:
    cal = load_calibration()
    cal["mic_x"] = x
    cal["mic_y"] = y
    save_calibration(cal)
    print(f"Saved mic button coordinates: ({x}, {y}) -> {CALIBRATION_JSON}")


# ─── App state (history.json via run-as) ────────────────────────────────────


def read_history(serial: str | None) -> list[dict]:
    """history.json doesn't exist until the app's first-ever recording is saved
    (a fresh install has nothing but the auto-generated profileInstalled file) --
    treat that the same way the app itself does: an empty history, not an error."""
    try:
        out = adb_shell(f"run-as {PACKAGE} cat files/history.json", serial=serial)
    except RuntimeError as e:
        if "no such file or directory" in str(e).lower():
            return []
        raise
    out = out.strip()
    if not out:
        return []
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        time.sleep(0.3)
        out2 = adb_shell(f"run-as {PACKAGE} cat files/history.json", serial=serial).strip()
        return json.loads(out2) if out2 else []


def top_entry_id(history: list[dict]) -> str | None:
    return history[0]["id"] if history else None


def find_entry(history: list[dict], entry_id: str) -> dict | None:
    for item in history:
        if item.get("id") == entry_id:
            return item
    return None


def strip_speaker_labels(text: str) -> str:
    """Remove every "Speaker N" tag and flatten the result to one sentence."""
    if not text:
        return ""
    cleaned = SPEAKER_LABEL_RE.sub(" ", text.replace("\n", " "))
    return re.sub(r"\s+", " ", cleaned).strip()


# ─── Audio playback ──────────────────────────────────────────────────────────


PLAYBACK_TARGET_DBFS = -20.0   # every file leaves the speaker at this level
PLAYBACK_MAX_BOOST_DB = 20.0   # never amplify a near-silent file into noise


# The output device is pinned at the start of a run. If it changes mid-run the
# calibrated level from the setup (65 dBA at the phone) no longer holds, and
# every sample after that is measured under conditions nobody recorded - so the
# run stops rather than carrying on producing numbers that look fine.
PLAYBACK_DEVICE: dict = {}


def current_output_device() -> dict:
    try:
        index = sd.default.device[1]
        info = sd.query_devices(index, "output")
        return {"index": int(index), "name": str(info.get("name", "")).strip(),
                "samplerate": float(info.get("default_samplerate", 0))}
    except Exception as e:
        return {"index": None, "name": "", "samplerate": 0.0, "error": str(e)}


def pin_output_device() -> dict:
    """Record which speaker this run is calibrated against."""
    PLAYBACK_DEVICE.clear()
    PLAYBACK_DEVICE.update(current_output_device())
    if PLAYBACK_DEVICE.get("name"):
        print(f"Audio output: {PLAYBACK_DEVICE['name']} "
              f"({PLAYBACK_DEVICE['samplerate']:.0f} Hz)")
    else:
        print("Audio output: could not be identified "
              f"({PLAYBACK_DEVICE.get('error', 'unknown')})")
    return PLAYBACK_DEVICE


def output_device_changed() -> str | None:
    """Returns a description of the change, or None if it is the same device."""
    if not PLAYBACK_DEVICE.get("name"):
        return None
    now = current_output_device()
    if not now.get("name"):
        return f"the output device is gone ({now.get('error', 'not available')})"
    if now["name"] != PLAYBACK_DEVICE["name"]:
        return (f"the output device changed from {PLAYBACK_DEVICE['name']!r} "
                f"to {now['name']!r}")
    return None


def _device_samplerate(device=None) -> float | None:
    """The rate the output device actually runs at."""
    try:
        target = device if device is not None else sd.default.device[1]
        return float(sd.query_devices(target, "output")["default_samplerate"])
    except Exception:
        return None


def _resample(data, src_rate: float, dst_rate: float):
    """Linear resample. Plain, but adequate for speech playback, and it removes
    any dependence on what the sound card is willing to do with an odd rate."""
    if src_rate == dst_rate or src_rate <= 0 or dst_rate <= 0:
        return data
    n_out = int(round(len(data) * dst_rate / src_rate))
    if n_out <= 1:
        return data
    src_t = np.arange(len(data), dtype=np.float64)
    dst_t = np.linspace(0, len(data) - 1, n_out, dtype=np.float64)
    return np.interp(dst_t, src_t, data).astype("float32")


def play_audio_blocking(path: Path, measured_rms_dbfs: float | None = None,
                        target_dbfs: float = PLAYBACK_TARGET_DBFS) -> tuple[float, float]:
    """Play one file at a fixed loudness. Returns (seconds played, gain in dB).

    Two things here fail silently if they are wrong.

    Level: peak normalisation was the old behaviour and it is the wrong knob. A
    clip with one loud transient gets pulled quiet, one without gets pushed
    loud, so identical speech arrives at the phone at different levels and the
    difference lands in the WER. Equalising on RMS fixes the speech itself,
    with a peak check so the gain never introduces clipping.

    Rate: the files are 16 kHz and most output devices run at 48 kHz. If the
    stream opens at a rate the device will not take, the audio plays at the
    wrong speed and finishes early - the phone hears a fraction of the sentence
    and the recogniser gets blamed for the missing words. The audio is
    resampled to the device's own rate here, and the elapsed time is compared
    with the file length afterwards, so a short playback is reported instead of
    being scored as a transcription failure.
    """
    data, file_rate = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if not data.size:
        return 0.0, 0.0

    expected_s = len(data) / float(file_rate)

    rms = float(np.sqrt(np.mean(np.square(data))))
    if measured_rms_dbfs not in (None, ""):
        current_db = float(measured_rms_dbfs)
    else:
        current_db = 20 * np.log10(rms) if rms > 1e-9 else -120.0

    gain_db = max(-PLAYBACK_MAX_BOOST_DB,
                  min(PLAYBACK_MAX_BOOST_DB, target_dbfs - current_db))
    data = data * (10 ** (gain_db / 20))

    peak = float(np.max(np.abs(data)))
    if peak > 0.99:
        data = data * (0.99 / peak)
        gain_db += 20 * np.log10(0.99 / peak)

    play_rate = float(file_rate)
    device_rate = _device_samplerate()
    if device_rate and abs(device_rate - float(file_rate)) > 1:
        data = _resample(data, float(file_rate), device_rate)
        play_rate = device_rate

    start = time.monotonic()
    sd.play(data, play_rate)
    sd.wait()
    elapsed = time.monotonic() - start

    if expected_s > 0.5 and elapsed < expected_s * 0.9:
        print(f"  [audio] played {elapsed:.1f}s of a {expected_s:.1f}s file "
              f"({elapsed / expected_s * 100:.0f}%)  file rate {file_rate} Hz, "
              f"device rate {device_rate or 'unknown'} Hz")

    return elapsed, round(float(gain_db), 1)


# ─── Manifest / results I/O ──────────────────────────────────────────────────


# Column spellings differ between the hand-made manifest (id/text/path) and the
# balanced dataset exported from metadata (file/transcription). Map both onto
# one shape here so nothing downstream has to care which file it was given.
_ID_KEYS = ("id", "sample_id", "global_index")
_TEXT_KEYS = ("text", "transcription", "sentence", "reference")
_PATH_KEYS = ("path", "file", "audio_file", "filename")

# Carried into every result row, so failures can be grouped by speaker or by
# recording condition rather than only counted.
TAG_COLUMNS = ["block", "source", "gender", "age_group", "native_place_state",
               "primary_language", "content_type", "length_bucket",
               "speaking_rate", "snr_band", "category", "difficulty"]


def _pick(cols: dict, keys: tuple) -> str | None:
    for k in keys:
        if k in cols:
            return cols[k]
    return None


def check_references(rows: list[dict], manifest: Path) -> None:
    """Refuse to start if the ground truth is not ground truth.

    A corrupted reference does not look like a bug while the run is going: the
    app transcribes correctly and is scored wrong, sample after sample, and the
    result is a plausible-looking report built on nothing. Cheap to check once,
    expensive to discover afterwards.
    """
    tabbed = [r for r in rows if "\t" in (r["text"] or "")]
    pathlike = [r for r in rows
                if re.match(r"^\s*\S+\.(?:wav|mp3|flac)\b", r["text"] or "", re.I)]
    empty = [r for r in rows if not (r["text"] or "").strip()]
    bad = {id(r) for r in tabbed + pathlike}

    if bad:
        print(f"\nThe references in {manifest.name} are not clean:")
        if tabbed:
            print(f"  {len(tabbed)} rows contain tab characters")
        if pathlike:
            print(f"  {len(pathlike)} rows start with an audio file name")
        for r in (tabbed + pathlike)[:3]:
            print(f"    id={r['id']}  {r['text'][:70]!r}")
        print("\nThese would be scored as the reference text, so the app would be")
        print("marked wrong for every word it did not invent. Repair them first:")
        print("    python fix_ground_truth.py --input <manifest> --output metadata_fixed.csv")
        print("    python balance_dataset.py --input metadata_fixed.csv --output dataset_v2.csv")
        raise SystemExit("Stopped: references need repair.")

    if empty:
        print(f"[warn]  {len(empty)} rows have no reference text and will be skipped")


def check_audio_present(rows: list[dict], manifest: Path, sample: int = 200) -> None:
    """Confirm the audio is actually where the manifest says before starting.

    A missing file scores 100% WER, which is indistinguishable in the report
    from the recogniser failing completely. Checking a sample up front turns a
    26,000-sample disaster into a one-line message.
    """
    probe = rows[:sample]
    missing = [r for r in probe if not r["audio_path"].exists()]
    if not missing:
        return

    share = len(missing) / len(probe)
    first = missing[0]["audio_path"]
    print(f"\n{len(missing)} of the first {len(probe)} audio files were not found.")
    print(f"  Manifest : {manifest}")
    print(f"  Looked for: {first}")
    print("  The path in the manifest is relative, so it is resolved against the")
    print("  manifest's folder. Put the manifest beside the audio folder, or pass")
    print("  --manifest with the copy that sits next to it.")
    if share > 0.25:
        raise SystemExit("Stopped: the audio could not be found.")
    print("  Continuing - the missing ones will be marked missing_audio.")


def load_manifest(path: Path | None = None) -> list[dict]:
    """Returns rows with id / text / audio_path resolved, plus the tag columns."""
    manifest = path or MANIFEST_PATH
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}\n"
                         f"Looked for {', '.join(MANIFEST_CANDIDATES)} in {DATASET_DIR}.")
    with open(manifest, encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise SystemExit(f"{manifest} has no rows.")

    cols = {c.lower().strip(): c for c in raw[0]}
    id_col = _pick(cols, _ID_KEYS)
    text_col = _pick(cols, _TEXT_KEYS)
    path_col = _pick(cols, _PATH_KEYS)
    if not (text_col and path_col):
        raise SystemExit(f"{manifest}: need a text column and an audio path column.\n"
                         f"Found: {list(raw[0])}")

    # The manifest may live beside the audio, or a folder away from it, or be
    # pointed at from somewhere else entirely. Try the obvious roots rather
    # than failing 26,000 times over a folder level.
    roots = []
    for candidate in (manifest.parent, manifest.parent.parent, DATASET_DIR,
                      DATASET_DIR.parent, Path.cwd(), Path.cwd().parent):
        if candidate not in roots:
            roots.append(candidate)

    rows = []
    for i, r in enumerate(raw, 1):
        rel = (r.get(path_col) or "").strip().replace("\\", "/")
        audio = Path(rel)
        if not audio.is_absolute():
            for root in roots:
                candidate = root / rel
                if candidate.exists():
                    audio = candidate
                    break
            else:
                audio = manifest.parent / rel
        rows.append({
            "id": (r.get(id_col) or "").strip() if id_col else f"{i:06d}",
            "text": (r.get(text_col) or "").strip(),
            "audio_path": audio,
            "duration_s": (r.get("duration_s") or r.get("duration") or "").strip(),
            "rms_dbfs": (r.get("rms_dbfs") or "").strip(),
            **{t: (r.get(cols.get(t, ""), "") or "").strip() for t in TAG_COLUMNS},
        })
    check_references(rows, manifest)
    check_audio_present(rows, manifest)
    return rows


def load_all_results() -> dict[str, dict]:
    """id -> row, deduped (last occurrence wins if duplicates ever slipped in)."""
    if not RESULTS_CSV.exists():
        return {}
    with open(RESULTS_CSV, encoding="utf-8") as f:
        rows = {}
        for row in csv.DictReader(f):
            rows[row["id"]] = row
        return rows


RESULT_FIELDS = [
    "timestamp", "run_id", "mic_mode", "id", "duration_s", "playback_gain_db",
    "reference", "hypothesis", "normalized_reference", "normalized_hypothesis",
    "status", "app_message", "wait_s", "speaker_analysis_s",
    "segments", "app_stopped_early", "played_s", "played_fraction",
    "analysis_field", "raw_hypothesis",
    "WER", "CER", "SER", "OOV_rate", "substitutions", "deletions", "insertions",
] + TAG_COLUMNS


class ResultWriteError(RuntimeError):
    """Raised when a scored sample could not be saved."""


def append_result(row: dict) -> None:
    """Upsert by id so retries never leave duplicate rows.

    Written to a temporary file and moved into place, so a failure part-way
    through cannot leave a half-written results file. A sample that cannot be
    saved stops the run: continuing would score samples that are not being
    recorded anywhere, which is worse than stopping.
    """
    existing = load_all_results()
    existing[row["id"]] = row
    tmp = RESULTS_CSV.with_suffix(".csv.tmp")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            for r in existing.values():
                writer.writerow(r)
        tmp.replace(RESULTS_CSV)
    except OSError as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        free = pc_free_mb()
        raise ResultWriteError(
            f"could not write {RESULTS_CSV.name}: {e}"
            + (f" (PC free space {free} MB)" if free is not None else "")) from e


STATUS_JSON = DATASET_DIR / "status.json"
STOP_REQUEST = DATASET_DIR / "stop.request"


def notify(title: str, message: str) -> None:
    """Put a message where someone away from the terminal will see it.

    Best effort and never blocking: the run must be able to exit whether or not
    a window can be shown, so the dialog is raised on its own thread and the
    console always gets the same text.
    """
    print("\n" + "!" * 68)
    print(f"  {title}")
    for line in message.splitlines():
        print(f"  {line}")
    print("!" * 68)

    def popup() -> None:
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.user32.MessageBoxW(0, message, title, 0x30 | 0x1000)
                return
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showwarning(title, message)
            root.destroy()
        except Exception:
            pass

    threading.Thread(target=popup, daemon=True).start()


# Wall-clock time this run began. A stop request written before it belongs to
# an earlier run and must be ignored, or a single forgotten file stops every
# run that follows it.
RUN_STARTED_AT = 0.0


def stop_requested() -> bool:
    if not STOP_REQUEST.exists():
        return False
    try:
        written = STOP_REQUEST.stat().st_mtime
    except OSError:
        return False
    if RUN_STARTED_AT and written < RUN_STARTED_AT - 1.0:
        clear_stop_request()      # stale: left over from a previous run
        print("  (cleared a stop request left over from an earlier run)")
        return False
    return True


def clear_stop_request() -> None:
    try:
        STOP_REQUEST.unlink()
    except OSError:
        pass
DASHBOARD_PY = DATASET_DIR / "dashboard.py"
DASHBOARD_PORT = 8765


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
    import webbrowser
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


def launch_dashboard(port: int = DASHBOARD_PORT, force_window: bool = False) -> None:
    """Make sure a dashboard is up AND visible before the run starts.

    Three cases, and the middle one is the one that used to go wrong: a
    dashboard process left over from an earlier run still holds the port, so
    nothing new starts, but the browser tab was closed long ago and the window
    never appears. Finding the port busy is not the same as the person being
    able to see it, so we check what is actually listening and open a tab at it.
    """
    import socket
    import urllib.error
    import urllib.request

    if not DASHBOARD_PY.exists():
        print(f"  (dashboard.py not found in {DATASET_DIR}; running without it)")
        return

    def port_open(p: int) -> bool:
        with socket.socket() as probe:
            probe.settimeout(0.6)
            return probe.connect_ex(("127.0.0.1", p)) == 0

    def dashboard_info(p: int) -> dict | None:
        """Returns the dashboard's status, or None if that port is something
        else. Also tells us whether a browser tab is already watching."""
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{p}/api/status", timeout=3) as r:
                body = r.read()
        except (urllib.error.URLError, OSError, ValueError):
            return None
        if not any(k in body for k in (b"preflight", b"controlled_by_dashboard",
                                       b"results_file")):
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    def is_our_dashboard(p: int) -> bool:
        return dashboard_info(p) is not None

    def show(p: int) -> None:
        url = f"http://127.0.0.1:{p}"
        opened = open_window(url)
        print(f"  Dashboard: {url}" + ("" if opened else "  (open this yourself)"))

    requested = port
    try:
        if port_open(port):
            info = dashboard_info(port)
            if info is not None:
                # A window that polled a few seconds ago is one that already
                # exists - opening another every run is how you end up with six
                # of them. A closed window stops polling, so the gap is short:
                # long enough to recognise a live window, short enough that a
                # closed one is noticed on the next run.
                seen = info.get("viewer_seen_s_ago")
                if force_window:
                    show(port)
                elif seen is not None and seen < 15:
                    print(f"  Dashboard is already open at http://127.0.0.1:{port} "
                          f"(check the taskbar; use --show-dashboard to open another)")
                else:
                    show(port)
                return
            # something else owns the port; move to the next free one
            for alt in range(requested + 1, requested + 12):
                if not port_open(alt):
                    port = alt
                    break
            else:
                print(f"  (ports {requested}-{requested + 11} are all busy; "
                      "start the dashboard yourself with --port)")
                return
            print(f"  Port {requested} is used by something else; "
                  f"starting the dashboard on {port}.")

        kwargs: dict = {"cwd": str(DATASET_DIR),
                        "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([sys.executable, str(DASHBOARD_PY), "--port", str(port),
                          "--no-browser"], **kwargs)

        # wait for it to answer, then open the tab from here, so the browser
        # never lands on a page that is not serving yet
        for _ in range(30):
            time.sleep(0.4)
            if port_open(port) and is_our_dashboard(port):
                show(port)
                return
        print(f"  (dashboard did not come up on port {port}; "
              f"try running: python dashboard.py --port {port})")
    except Exception as e:
        print(f"  (could not start the dashboard: {e})")


def last_result(done: list[dict]) -> dict:
    """The most recently finished sample, in full. The dashboard shows the
    reference being played and, beside it, what the app actually returned for
    the one before -- which is the pair you watch to notice a run going wrong."""
    if not done:
        return {}
    r = done[-1]
    return {
        "id": r.get("id"), "status": r.get("status"),
        "reference": r.get("reference"), "hypothesis": r.get("hypothesis"),
        "WER": r.get("WER"), "CER": r.get("CER"), "SER": r.get("SER"),
        "substitutions": r.get("substitutions"), "deletions": r.get("deletions"),
        "insertions": r.get("insertions"),
        "content_type": r.get("content_type") or r.get("category"),
        "snr_band": r.get("snr_band"), "source": r.get("source"),
        "playback_gain_db": r.get("playback_gain_db"),
    }


def write_status(state: str, done: list[dict], total: int, current: dict | None,
                 started_at: float) -> None:
    """One small file, rewritten after every sample. The dashboard polls this
    and nothing else: it only reads, so it can never slow down or crash a run,
    and whatever finished survives if the run dies."""
    scored = [r for r in done if r.get("status") == "ok" and (r.get("hypothesis") or "").strip()]
    failed = [r for r in done if r not in scored]

    def avg(key: str) -> float:
        vals = [float(r[key]) for r in scored if str(r.get(key, "")).strip() not in ("", "None")]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def total_of(key: str) -> int:
        return int(sum(float(r.get(key) or 0) for r in scored))

    elapsed = time.monotonic() - started_at
    rate = elapsed / len(done) if done else 0.0
    remaining = total - len(done)

    payload = {
        "state": state,
        "run_id": RUN_META.get("run_id", ""),
        "mic_mode": RUN_META.get("mic_mode", ""),
        "ble_connected": RUN_META.get("ble_connected"),
        "ble_detail": RUN_META.get("ble_detail", ""),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "progress": {"completed": len(done), "total": total, "remaining": remaining,
                     "scored": len(scored), "failed": len(failed),
                     "percent": round(len(done) / total * 100, 1) if total else 0.0},
        "cumulative": {
            "WER": avg("WER"), "CER": avg("CER"),
            "SER": round(sum(1 for r in scored if float(r.get("SER") or 0)) / len(scored) * 100, 2)
                   if scored else 0.0,
            "substitutions": total_of("substitutions"),
            "deletions": total_of("deletions"),
            "insertions": total_of("insertions"),
        },
        "timing": {"elapsed_s": round(elapsed, 1),
                   "avg_per_sample_s": round(rate, 1),
                   "eta_s": round(rate * remaining, 1)},
        "current": current or {},
        "last_result": last_result(done),
        "failures_by_status": {k: sum(1 for r in failed if r.get("status") == k)
                               for k in {r.get("status") for r in failed}},
        "recent": [{"id": r.get("id"), "WER": r.get("WER"), "CER": r.get("CER"),
                    "status": r.get("status")} for r in done[-10:]],
        "run_meta": RUN_META,
    }
    tmp = STATUS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS_JSON)   # atomic, so the dashboard never reads half a file


def print_running(done: list[dict], total: int, started_at: float) -> None:
    scored = [r for r in done if r.get("status") == "ok" and (r.get("hypothesis") or "").strip()]
    if not scored:
        print(f"  Running   : no scored samples yet ({len(done)}/{total} attempted)")
        return
    wer = sum(float(r["WER"]) for r in scored) / len(scored)
    cer = sum(float(r["CER"]) for r in scored) / len(scored)
    ser = sum(1 for r in scored if float(r.get("SER") or 0)) / len(scored) * 100
    sub = int(sum(float(r.get("substitutions") or 0) for r in scored))
    dele = int(sum(float(r.get("deletions") or 0) for r in scored))
    ins = int(sum(float(r.get("insertions") or 0) for r in scored))
    elapsed = time.monotonic() - started_at
    rate = elapsed / len(done) if done else 0.0
    print(f"  Running   : WER={wer:.2f}%  CER={cer:.2f}%  SER={ser:.1f}%  "
          f"S={sub} D={dele} I={ins}")
    print(f"  Progress  : {len(done)}/{total} done, {len(done) - len(scored)} failed, "
          f"{total - len(done)} left  |  {elapsed / 60:.1f} min elapsed, "
          f"{rate:.1f}s per sample")


# ─── Per-sample pipeline ─────────────────────────────────────────────────────


class PollAborted:
    """Returned by poll_until when its abort predicate fired."""

    def __init__(self, reason):
        self.reason = reason


def poll_until(predicate, timeout: float, interval: float = POLL_INTERVAL_S,
               serial: str | None = None, keepalive_every: float = 10.0,
               abort_if=None, abort_every: float = 4.0):
    """`abort_if` is checked while waiting. When it returns a value, the wait
    stops immediately and that value comes back wrapped in PollAborted.

    The checks are front-loaded rather than evenly spaced, because the thing
    being watched for is usually a toast: the app shows "Speech not recognised"
    the moment recording stops and it is gone two or three seconds later. A
    check every four seconds frequently arrives after the message has already
    faded, and then the wait runs to its full timeout for a message that will
    never appear again. So: look almost immediately, keep looking closely for
    the first few seconds, then settle into the slower interval.
    """
    """`serial` set -> sends a harmless wake keyevent every `keepalive_every`
    seconds while waiting. A 90s analysis wait with zero touch events was long
    enough for the screen to time out and lock mid-sample on some phones,
    regardless of `keep_screen_awake` (which only holds while the phone
    registers as actively charging -- not guaranteed on every cable/hub)."""
    start = time.monotonic()
    deadline = start + timeout
    last_keepalive = start
    # seconds after the start at which to look for a transient message
    schedule = [0.3, 1.2, 2.2, 3.4, 5.0, 7.0]
    next_abort = schedule.pop(0) if abort_if is not None else None

    while time.monotonic() < deadline:
        result = predicate()
        if result is not None:
            return result
        if abort_if is not None and (time.monotonic() - start) >= next_abort:
            reason = abort_if()
            if reason is not None:
                return PollAborted(reason)
            next_abort = (schedule.pop(0) if schedule
                          else (time.monotonic() - start) + abort_every)
        if serial and time.monotonic() - last_keepalive >= keepalive_every:
            try:
                adb_shell("input keyevent 224", serial=serial)  # KEYCODE_WAKEUP
            except RuntimeError:
                pass
            last_keepalive = time.monotonic()
        time.sleep(interval)
    return None


def blank_result(row: dict, status: str = "ok") -> dict:
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": RUN_META.get("run_id", ""),
        "mic_mode": RUN_META.get("mic_mode", "phone"),
        "id": row["id"],
        "duration_s": row.get("duration_s", ""),
        "reference": row["text"],
        "hypothesis": "",
        "playback_gain_db": "",
        "status": status,
        "app_message": "",
        "wait_s": "",
        "speaker_analysis_s": "",
        "segments": "",
        "app_stopped_early": "",
        "played_s": "",
        "played_fraction": "",
        "analysis_field": "",
        "raw_hypothesis": "",
    }
    for t in TAG_COLUMNS:
        result[t] = row.get(t, "")
    return result


def process_sample(row: dict, mic_xy: tuple[int, int], serial: str | None) -> dict:
    audio_path = row["audio_path"]
    result = blank_result(row)

    if not audio_path.exists():
        result["status"] = "missing_audio"
        return result

    if not ensure_screen_ready(serial):
        result["status"] = "screen_locked"
        return result

    before_id = top_entry_id(read_history(serial))

    tap(*mic_xy, serial=serial)
    time.sleep(PRE_RECORD_SETTLE_S)

    if not is_recording(serial):
        # One retry: the first tap sometimes lands while the screen is settling.
        tap(*mic_xy, serial=serial)
        time.sleep(PRE_RECORD_SETTLE_S)
        if not is_recording(serial):
            result["status"] = "record_not_started"
            return result

    played_s, gain_db = play_audio_blocking(audio_path, row.get("rms_dbfs") or None)
    result["playback_gain_db"] = gain_db
    result["played_s"] = round(played_s, 2)
    expected = float(row.get("duration") or row.get("duration_s") or 0)
    if expected > 0.5:
        result["played_fraction"] = round(played_s / expected, 2)
        if played_s < expected * 0.9:
            result["app_message"] = (f"only {played_s:.1f}s of a {expected:.1f}s "
                                     f"file was played")
    time.sleep(POST_RECORD_TAIL_S)

    # If the app has already stopped on its own, it ended the utterance before
    # the audio finished - its voice-activity detector closed the segment on a
    # pause. That truncates the transcript, and it is an app-side behaviour, so
    # it is worth recording rather than silently scoring the fragment.
    still_recording = is_recording(serial)
    result["app_stopped_early"] = "" if still_recording else "yes"
    tap(*mic_xy, serial=serial)

    # If this read fails or comes back empty, every entry in the app's history
    # looks new a moment later, and the whole night's transcripts get joined
    # into one hypothesis. Retry, and record whether we can trust it.
    baseline_ids = {e.get("id") for e in read_history(serial)}
    baseline_ok = bool(baseline_ids)
    if not baseline_ok:
        time.sleep(1.0)
        baseline_ids = {e.get("id") for e in read_history(serial)}
        baseline_ok = bool(baseline_ids)

    def _new_entry():
        hist = read_history(serial)
        new_id = top_entry_id(hist)
        if new_id and new_id != before_id:
            return new_id
        return None

    t_wait = time.monotonic()

    # The app puts its verdict on screen the instant recording stops, so look
    # once straight away - before any polling - while the toast is still up.
    immediate = app_message_on_screen(serial)
    if immediate:
        result["status"] = immediate
        result["app_message"] = immediate
        result["wait_s"] = round(time.monotonic() - t_wait, 1)
        return result

    # A stop asked for while this sample is in flight should land in seconds,
    # not after the next 90-second wait finishes.
    def _abort_reason():
        if stop_requested():
            return "stopped_by_request"
        return app_message_on_screen(serial)

    new_id = poll_until(_new_entry, timeout=NEW_ENTRY_TIMEOUT_S, serial=serial,
                        abort_if=_abort_reason)
    if isinstance(new_id, PollAborted):
        result["status"] = new_id.reason
        result["app_message"] = new_id.reason
        result["wait_s"] = round(time.monotonic() - t_wait, 1)
        return result
    result["wait_s"] = round(time.monotonic() - t_wait, 1)
    if new_id is None:
        # The app often has a reason on screen - "speech not recognised", "no
        # speech detected". Recording it turns a blank no_new_recording row
        # into something that says what actually happened.
        said = app_message_on_screen(serial)
        result["status"] = said or "no_new_recording"
        if said:
            result["app_message"] = said
        return result

    # If the recording produced no text, Speaker Analysis has nothing to label
    # and will never fill standardizedText - so waiting the full timeout tells
    # us nothing we do not already know.
    history_now = read_history(serial)
    fresh = [e for e in history_now if e.get("id") not in baseline_ids]
    # Chronological: the app writes newest-first, and a split recording has to
    # be rejoined in the order it was spoken.
    fresh.sort(key=lambda e: str(e.get("timestamp") or ""))
    result["segments"] = len(fresh)

    # One played file can legitimately split into a handful of sessions. It
    # cannot split into hundreds - that means the baseline was wrong, and
    # joining would hand back everything the app has ever recorded.
    if len(fresh) > MAX_JOIN_SEGMENTS or (len(fresh) > 1 and not baseline_ok):
        result["app_message"] = (f"{len(fresh)} sessions looked new, which is "
                                 "implausible for one clip; scored the newest only")
        result["segments"] = 1
        fresh = [e for e in fresh if e.get("id") == new_id] or fresh[-1:]

    if len(fresh) > 1:
        # The voice-activity detector split one played file into several
        # sessions. Scoring only one of them loses most of the sentence, so
        # they are rejoined. Speaker Analysis labels a single session, so it is
        # skipped here and the live transcript is used.
        joined = " ".join((e.get("text") or "").strip() for e in fresh
                          if (e.get("text") or "").strip())
        result["hypothesis"] = strip_speaker_labels(joined)
        result["app_message"] = f"{len(fresh)} VAD segments joined"
        result["status"] = "ok" if joined.strip() else "empty_transcript"
        close_session_overlay(serial)
        return result

    saved = find_entry(history_now, new_id) or {}
    if not (saved.get("text") or "").strip():
        said = app_message_on_screen(serial)
        result["status"] = said or "empty_transcript"
        result["app_message"] = said or "the saved session had no text"
        close_session_overlay(serial)
        return result

    t_analysis = time.monotonic()
    sa_xy = find_analysis_button_xy(serial)
    tap(*sa_xy, serial=serial)

    before_fields = {f: (saved.get(f) or "").strip() for f in analysis_fields()}

    def _analysis_done():
        entry = find_entry(read_history(serial), new_id)
        if not entry:
            return None
        for field in analysis_fields():
            value = (entry.get(field) or "").strip()
            # "text" already holds the live transcript, so it only counts as a
            # result if pressing the button actually changed it.
            if value and value != before_fields.get(field, ""):
                entry["_analysis_field"] = field
                return entry
        return None

    wait_for = ANALYSIS_TIMEOUT_S
    fallback = analysis_fallback()
    if fallback and analysis_grace():
        wait_for = min(ANALYSIS_TIMEOUT_S, analysis_grace())

    final_entry = poll_until(_analysis_done, timeout=wait_for, serial=serial,
                             abort_if=_abort_reason)

    if final_entry is None and fallback:
        # Nothing new appeared. If the app already holds a transcript for this
        # session, that is what the button was showing, so score it rather than
        # time out on a result that was there all along.
        entry = find_entry(read_history(serial), new_id) or {}
        existing = (entry.get(fallback) or "").strip()
        if existing:
            entry["_analysis_field"] = fallback
            final_entry = entry
            result["app_message"] = (f"no new field after {wait_for:.0f}s; "
                                     f"scored the saved {fallback}")
    result["speaker_analysis_s"] = round(time.monotonic() - t_analysis, 1)
    if isinstance(final_entry, PollAborted):
        result["status"] = final_entry.reason
        result["app_message"] = final_entry.reason
        close_session_overlay(serial)
        return result
    if final_entry is None:
        result["status"] = "analysis_timeout"
        result["app_message"] = (f"no labelled text after "
                                 f"{result['speaker_analysis_s']:.0f}s")
        close_session_overlay(serial)
        return result

    field = final_entry.get("_analysis_field") or analysis_fields()[0]
    result["analysis_field"] = field
    result["raw_hypothesis"] = final_entry.get(field) or ""
    # Speaker labels come off whichever field produced the text: the measure is
    # transcription accuracy, so a diarisation split must not cost words.
    result["hypothesis"] = strip_speaker_labels(result["raw_hypothesis"])
    close_session_overlay(serial)
    return result


# ─── Summary ─────────────────────────────────────────────────────────────────


def cmd_summarize(source: Path | None = None) -> None:
    global RESULTS_CSV, SUMMARY_JSON
    RESULTS_CSV = resolve_results_csv(source)
    print(f"Reading results from: {RESULTS_CSV.name}")
    with open(RESULTS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    scored = [r for r in rows if r["status"] == "ok" and r["hypothesis"]]
    refs = [normalize(r["reference"]) for r in scored]
    hyps = [normalize(r["hypothesis"]) for r in scored]

    summary = {
        "total_samples": len(rows),
        "scored_samples": len(scored),
        "failed_samples": len(rows) - len(scored),
        "failure_breakdown": {},
        "overall_wer": round(jiwer_wer(refs, hyps), 4) if refs else None,
        "overall_cer": round(jiwer_cer(refs, hyps), 4) if refs else None,
        "by_category": {},
        "by_difficulty": {},
    }

    for r in rows:
        if r["status"] != "ok" or not r["hypothesis"]:
            summary["failure_breakdown"][r["status"]] = summary["failure_breakdown"].get(r["status"], 0) + 1

    for key, bucket in (("category", "by_category"), ("difficulty", "by_difficulty")):
        groups: dict[str, list[dict]] = {}
        for r in scored:
            groups.setdefault(r[key], []).append(r)
        for name, items in groups.items():
            g_refs = [normalize(i["reference"]) for i in items]
            g_hyps = [normalize(i["hypothesis"]) for i in items]
            summary[bucket][name] = {
                "count": len(items),
                "wer": round(jiwer_wer(g_refs, g_hyps), 4),
                "cer": round(jiwer_cer(g_refs, g_hyps), 4),
            }

    SUMMARY_JSON = DATASET_DIR / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {SUMMARY_JSON}")


# ─── Full metric suite ───────────────────────────────────────────────────────
#
# Computes WER/CER/SER/OOV and S/D/I from the same semantic normalization used
# during per-sample evaluation. Vocabulary coverage is intentionally omitted.

FULL_METRICS_CSV = DATASET_DIR / "full_metrics.csv"  # reassigned per-invocation, see new_full_metrics_csv_path
FULL_METRICS_REPORT_PREFIX = "smartscribe_asr_eval_"
APP_NAME_FOR_REPORT = "SmartScribe Speaker Analysis (on-device ASR)"

BENCHMARK = {
    "WER %":          [("Excellent", "< 5 %"), ("Good", "5 - 15 %"), ("Acceptable", "15 - 25 %"), ("Poor", "25 - 40 %"), ("Very Poor", "> 40 %")],
    "CER %":          [("Excellent", "< 3 %"), ("Good", "3 - 10 %"), ("Acceptable", "10 - 20 %"), ("Poor", "> 20 %")],
    "SER %":          [("Excellent", "< 10 %"), ("Good", "10 - 30 %"), ("Acceptable", "30 - 50 %"), ("Poor", "> 50 %")],
}


def oov_rate(hyp: str, ref_n: str) -> float:
    """Percentage of normalized hypothesis words absent from this sample's reference."""
    hyp_words = normalize(hyp).split()
    ref_words = set(ref_n.split())
    if not hyp_words:
        return 0.0
    return round(sum(1 for w in hyp_words if w not in ref_words) / len(hyp_words) * 100, 4)


def _alignment_counts(ref_n: str, hyp_n: str) -> tuple[int, int, int]:
    """Return word-level substitutions, deletions and insertions for normalized text."""
    try:
        out = jiwer.process_words(ref_n, hyp_n)
        return out.substitutions, out.deletions, out.insertions
    except Exception:
        return 0, 0, 0


def compute_full_metrics(ref: str, hyp: str) -> dict:
    """Compute all retained metrics from the same normalized ref/hyp pair."""
    ref_n, hyp_n = normalize(ref), normalize(hyp)
    if not hyp.strip():
        ref_words = ref_n.split()
        return {
            "normalized_reference": ref_n,
            "normalized_hypothesis": hyp_n,
            "WER": 100.0, "CER": 100.0, "SER": 1,
            "OOV_rate": 0.0,
            "substitutions": 0,
            "deletions": len(ref_words),
            "insertions": 0,
        }
    wer = round(jiwer_wer(ref_n, hyp_n) * 100, 4)
    cer = round(jiwer_cer(ref_n, hyp_n) * 100, 4)
    ser = 0 if wer == 0.0 else 1
    s, d, i = _alignment_counts(ref_n, hyp_n)
    return {
        "normalized_reference": ref_n,
        "normalized_hypothesis": hyp_n,
        "WER": wer, "CER": cer, "SER": ser,
        "OOV_rate": oov_rate(hyp, ref_n),
        "substitutions": s, "deletions": d, "insertions": i,
    }

def label_wer(v: float) -> str:
    if v < 5: return "Excellent"
    if v < 15: return "Good"
    if v < 25: return "Acceptable"
    if v < 40: return "Poor"
    return "Very Poor"


def label_cer(v: float) -> str:
    if v < 3: return "Excellent"
    if v < 10: return "Good"
    if v < 20: return "Acceptable"
    return "Poor"


def label_ser(v: float) -> str:
    if v < 10: return "Excellent"
    if v < 30: return "Good"
    if v < 50: return "Acceptable"
    return "Poor"



def _avg(lst: list[float]) -> float:
    return round(sum(lst) / len(lst), 4) if lst else 0.0


def _read_results_with_retry() -> list[dict]:
    """A `run` in progress rewrites results_detailed.csv on every sample (upsert);
    retry a couple of times if we catch it mid-write."""
    for _ in range(5):
        try:
            with open(RESULTS_CSV, encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except PermissionError:
            time.sleep(0.3)
    raise SystemExit(f"Could not read {RESULTS_CSV} (locked?). Close any program that has it open.")


def build_full_summary(rows: list[dict]) -> tuple[dict, list[dict]]:
    per_sample = []
    for r in rows:
        m = compute_full_metrics(r["reference"], r["hypothesis"])
        per_sample.append({**r, **m})

    wers = [p["WER"] for p in per_sample]
    cers = [p["CER"] for p in per_sample]
    sers = [p["SER"] for p in per_sample]
    oovs = [p["OOV_rate"] for p in per_sample]
    subs = [p["substitutions"] for p in per_sample]
    dels = [p["deletions"] for p in per_sample]
    inss = [p["insertions"] for p in per_sample]
    ser_pct = round(sum(sers) / len(sers) * 100, 4) if sers else 0.0

    summary = {
        "total": len(rows),
        "evaluated": len(per_sample),
        "empty_or_failed": sum(1 for p in per_sample if not p["hypothesis"].strip()),
        "WER_avg": _avg(wers), "CER_avg": _avg(cers), "SER_pct": ser_pct,
        "OOV_avg": _avg(oovs),
        "total_sub": sum(subs), "total_del": sum(dels), "total_ins": sum(inss),
    }
    return summary, per_sample


def print_full_summary(s: dict) -> None:
    W = 72
    print(f"\n{'=' * W}")
    print(f"  {APP_NAME_FOR_REPORT} - EVALUATION SUMMARY")
    print(f"{'=' * W}")
    print(f"  Evaluated            : {s['evaluated']} / {s['total']}")
    print(f"  Empty/failed output  : {s['empty_or_failed']}")
    print("\n  -- ACCURACY")
    print(f"  Avg WER %            : {s['WER_avg']:.4f} %   [{label_wer(s['WER_avg'])}]")
    print(f"  Avg CER %            : {s['CER_avg']:.4f} %   [{label_cer(s['CER_avg'])}]")
    print(f"  Sentence Error Rate  : {s['SER_pct']:.4f} %   [{label_ser(s['SER_pct'])}]")
    print(f"  Avg OOV Rate         : {s['OOV_avg']:.4f} %")
    print(f"  Total Substitutions  : {s['total_sub']}")
    print(f"  Total Deletions      : {s['total_del']}")
    print(f"  Total Insertions     : {s['total_ins']}")


def format_per_sample_lines(per_sample: list[dict]) -> list[str]:
    W = 72
    total = len(per_sample)
    lines: list[str] = []
    for i, p in enumerate(per_sample, 1):
        lines.append("")
        lines.append("-" * W)
        lines.append(f"  [{i:04d}/{total:04d}]  id={p.get('id', '')}  REFERENCE:")
        lines.append(f"  {p['reference']}")
        lines.append("-" * W)
        if p["hypothesis"].strip():
            lines.append(f"  App output : {p['hypothesis']}")
            lines.append(f"  WER={p['WER']:.2f}% [{label_wer(p['WER'])}]  "
                          f"CER={p['CER']:.2f}% [{label_cer(p['CER'])}]  "
                          f"S={p['substitutions']} D={p['deletions']} I={p['insertions']}")
        else:
            lines.append(f"  App output : (none -- status={p['status']})")
            lines.append("  WER=100.00% [Very Poor]  CER=100.00% [Poor]")
    return lines


def save_full_txt_report(s: dict, per_sample: list[dict], ts: str, ts_file: str) -> Path:
    W2 = 58
    lines: list[str] = []

    def sep(c="="):
        lines.append(c * W2)

    def row(label, value, lbl=""):
        lines.append(f"  {label:<32} {value:>14}{'   ' + lbl if lbl else ''}")

    sep()
    lines.append(f"  {APP_NAME_FOR_REPORT} - EVALUATION REPORT")
    lines.append(f"  Date      : {ts}")
    lines.append(f"  Sentences : {s['total']}")
    sep()
    lines.append("")
    lines.append(f"  {'Files Evaluated':<32} {s['evaluated']:>7} / {s['total']}")
    lines.append(f"  {'Empty/failed output':<32} {s['empty_or_failed']:>14}")
    lines.append("")

    sep("-")
    lines.append("  ACCURACY")
    sep("-")
    row("Avg WER %", f"{s['WER_avg']:.4f} %", label_wer(s['WER_avg']))
    row("Avg CER %", f"{s['CER_avg']:.4f} %", label_cer(s['CER_avg']))
    row("Sentence Error Rate %", f"{s['SER_pct']:.4f} %", label_ser(s['SER_pct']))
    row("Avg OOV Rate %", f"{s['OOV_avg']:.4f} %")
    lines.append("")
    row("Total Substitutions", f"{s['total_sub']}")
    row("Total Deletions", f"{s['total_del']}")
    row("Total Insertions", f"{s['total_ins']}")
    lines.append("")

    sep("-")
    lines.append("  INDUSTRY BENCHMARK RANGES")
    sep("-")
    for metric, bands in BENCHMARK.items():
        lines.append(f"  {metric}:")
        for label, rng in bands:
            lines.append(f"    {label:<14}  {rng}")
        lines.append("")

    sep()
    lines.append("  PER-SAMPLE DETAIL")
    sep()
    lines.extend(format_per_sample_lines(per_sample))
    lines.append("")

    sep()
    lines.append("  END OF REPORT")
    sep()

    out_path = DATASET_DIR / f"{FULL_METRICS_REPORT_PREFIX}{ts_file}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def print_per_sample_detail(per_sample: list[dict]) -> None:
    for line in format_per_sample_lines(per_sample):
        print(line)


def cmd_full_metrics(detail: bool = True, source: Path | None = None) -> None:
    global RESULTS_CSV, FULL_METRICS_CSV
    RESULTS_CSV = resolve_results_csv(source)
    print(f"Reading results from: {RESULTS_CSV.name}")

    rows = _read_results_with_retry()
    if not rows:
        raise SystemExit(f"No results yet at {RESULTS_CSV}")

    summary, per_sample = build_full_summary(rows)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    ts_file = time.strftime("%Y%m%d_%H%M%S")

    # A NEW timestamped file every call -- past snapshots (e.g. a mid-run check
    # versus the final one) are never overwritten.
    FULL_METRICS_CSV = new_full_metrics_csv_path(ts_file)
    with open(FULL_METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for p in per_sample:
            writer.writerow({k: p.get(k, "") for k in RESULT_FIELDS})

    if detail:
        print_per_sample_detail(per_sample)

    print_full_summary(summary)
    report_path = save_full_txt_report(summary, per_sample, ts, ts_file)
    print(f"\nPer-sample metrics : {FULL_METRICS_CSV}")
    print(f"Report saved       : {report_path}")


# ─── Listen (audibility check) ───────────────────────────────────────────────


def cmd_listen(sample_id: str | None) -> None:
    manifest = load_manifest()
    row = manifest[0] if sample_id is None else next((r for r in manifest if r["id"] == sample_id), None)
    if row is None:
        raise SystemExit(f"No sample with id={sample_id!r} in manifest.")
    audio_path = row["audio_path"]
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")
    print(f"Playing id={row['id']}: {row['text']!r} ({row['duration_s']}s)")
    print("Stand the phone where it'll sit during the real run and confirm you can hear this clearly.")
    elapsed, gain_db = play_audio_blocking(audio_path, row.get("rms_dbfs") or None)
    print(f"Done ({elapsed:.2f}s elapsed, {gain_db:+.1f} dB applied to reach "
          f"{PLAYBACK_TARGET_DBFS:.0f} dBFS).")


# ─── Run command ─────────────────────────────────────────────────────────────


MAX_CONSECUTIVE_FAILURES = 5


def build_report(results: Path | None) -> None:
    """Write the Excel report for a finished run.

    Imported here rather than at the top so a missing openpyxl stops the report
    and not the evaluation: losing hours of measurement because a reporting
    dependency is absent would be the wrong trade.
    """
    if results is None or not Path(results).exists():
        return
    try:
        import make_report
        out = make_report.build(Path(results))
        print(f"Report: {out.name}")
    except ImportError:
        print("  (make_report.py or openpyxl not found; no Excel report written)")
    except Exception as e:
        print(f"  (report could not be built: {type(e).__name__}: {e})")


def cmd_run(serial: str | None, limit: int | None, start: int,
            manifest: Path | None = None, mic_mode: str | None = None,
            resume: bool = False, conditions: dict | None = None,
            dashboard: bool = True, force_window: bool = False) -> None:
    global RESULTS_CSV, RUN_META
    serial = ensure_single_device(serial)
    check_debuggable(serial)
    grant_record_audio(serial)
    launch_app(serial)
    keep_screen_awake(serial)

    if is_overlay_open(serial):
        print("Session-detail overlay was left open from a previous run; closing it.")
        close_session_overlay(serial)

    mic_mode = choose_mic(serial, mic_mode)
    if mic_mode is None:
        print("Stopped before starting - no microphone chosen.")
        return

    # Opened now rather than earlier: the microphone question needs the
    # terminal, and a browser window stealing focus mid-question is a nuisance.
    if dashboard:
        launch_dashboard(force_window=force_window)

    manifest_path = manifest or MANIFEST_PATH
    print(f"Dataset: {manifest_path.name}")
    manifest_all = load_manifest(manifest_path)
    rows = manifest_all[start:]
    if limit:
        rows = rows[:limit]

    ts = time.strftime("%Y%m%d_%H%M%S")
    done: list[dict] = []
    already: set[str] = set()

    # Continuing is the default. Re-running after an interruption should pick
    # up where it stopped rather than silently redoing hours of work; starting
    # over is the deliberate choice, with --fresh.
    if resume:
        previous = latest_results_csv()
        if previous and previous.exists():
            RESULTS_CSV = previous
            with open(previous, encoding="utf-8") as f:
                done = list(csv.DictReader(f))
            already = {r["id"] for r in done if r.get("status") == "ok"
                       and (r.get("hypothesis") or "").strip()}
            retry = len(done) - len(already)
            print(f"Continuing {previous.name}: {len(already)} already scored"
                  + (f", {retry} failed rows will be retried." if retry else "."))
            print("  (use --fresh to start a new results file instead)")
        else:
            RESULTS_CSV = new_results_csv_path(ts)
            print(f"No previous run found. Results: {RESULTS_CSV.name}")
    else:
        RESULTS_CSV = new_results_csv_path(ts)
        print(f"Starting fresh. Results will be saved to: {RESULTS_CSV.name}")

    RUN_META = {
        "run_id": f"{ts}_{mic_mode}",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "app_version": app_version(serial),
        "device_model": device_model(serial),
        "device_serial": serial,
        "dataset_file": manifest_path.name,
        "dataset_rows": len(manifest_all),
        "samples_this_run": len(rows),
        "mic_mode": mic_mode,
        "playback_target_dbfs": PLAYBACK_TARGET_DBFS,
        "output_device": PLAYBACK_DEVICE.get("name", ""),
        **(conditions or {}),
    }
    print(f"Run id {RUN_META['run_id']}  |  app {RUN_META['app_version']}  "
          f"|  {RUN_META['device_model']}  |  mic {mic_mode}")
    missing = [k for k in ("pc_volume", "distance_cm", "measured_dba")
               if not RUN_META.get(k)]
    if missing:
        print(f"  note: acoustic conditions not recorded ({', '.join(missing)}). "
              "Pass --volume / --distance-cm / --dba so this run can be compared "
              "with others.")
    print(f"{len(rows)} samples to process.")

    pin_output_device()
    stop_reason = check_resources(serial)
    if stop_reason:
        print(f"\nNot starting: {stop_reason}")
        return

    mic_xy = get_mic_xy(serial)
    started_at = time.monotonic()
    consecutive_failures = 0
    global RUN_STARTED_AT
    RUN_STARTED_AT = time.time()
    clear_stop_request()

    if mic_mode == "ble":
        ok, detail = ble_connected_in_app(serial)
        RUN_META["ble_connected"] = ok or bool(ble_stack_devices(serial))
        RUN_META["ble_detail"] = detail

    # Write one status file before the first sample, so the dashboard has
    # something to show the moment the browser opens.
    write_status("running", [], len(rows), None, started_at)

    pending = [r for r in rows if r["id"] not in already]
    if not pending:
        print(f"\nAll {len(rows)} requested samples are already scored in "
              f"{RESULTS_CSV.name}. Nothing to do.\n"
              "  --fresh            start a new results file\n"
              f"  --start {len(already)}      skip past what is already done\n"
              "  --limit N          take more samples")
        write_status("completed", done, len(rows), None, started_at)
        return
    if len(pending) < len(rows):
        print(f"{len(rows) - len(pending)} of these were already scored; "
              f"{len(pending)} to do.")

    processed = 0
    for i, row in enumerate(rows, 1):
        if row["id"] in already:
            continue
        processed += 1

        # Stop can be asked for from the dashboard, from another terminal, or
        # by deleting nothing more than a file - so it works regardless of who
        # started this run.
        if stop_requested():
            clear_stop_request()
            write_status("paused", done, len(rows), None, started_at)
            print(f"\nStop requested. {len(done)} samples are saved in "
                  f"{RESULTS_CSV.name}; re-run with --resume to continue.")
            build_report(RESULTS_CSV)
            return

        # In BLE mode the link is checked before every sample. A microphone
        # that drops out mid-run does not announce itself: the app simply
        # records nothing, and every remaining sample scores as a total
        # failure. Better to stop and say so.
        if mic_mode == "ble":
            ok, detail = ble_connected_in_app(serial)
            if not ok and not ble_stack_devices(serial):
                RUN_META["ble_connected"] = False
                RUN_META["ble_detail"] = detail
                write_status("stopped_ble_disconnected", done, len(rows), None, started_at)
                notify("BLE microphone disconnected",
                       f"The BLE microphone is no longer connected.\n\n"
                       f"Checked before sample {i} of {len(rows)}.\n"
                       f"{detail}\n\n"
                       f"{len(done)} samples are saved in {RESULTS_CSV.name}.\n"
                       f"Reconnect the microphone and re-run with --resume.")
                return
            RUN_META["ble_connected"] = True
            RUN_META["ble_detail"] = detail

        # The speaker is the one thing that cannot be checked after the fact:
        # if it changed, the calibrated level is gone and every later sample is
        # measured under conditions nobody wrote down.
        changed = output_device_changed()
        if changed:
            write_status("paused", done, len(rows), None, started_at)
            notify("Audio output changed",
                   f"{changed}.\n\nThe level calibration no longer applies, so "
                   f"the run has stopped before sample {i}.\n"
                   f"{len(done)} samples are saved in {RESULTS_CSV.name}.\n"
                   "Restore the output device, then re-run to continue.")
            build_report(RESULTS_CSV)
            return

        # Battery and free space, every 25 samples: cheap, and all three are
        # slow failures that would otherwise end a long run in the small hours.
        if processed % 25 == 1 and processed > 1:
            stop_reason = check_resources(serial, loud=False)
            if stop_reason:
                write_status("paused", done, len(rows), None, started_at)
                notify("Run paused", f"{stop_reason}\n\n"
                       f"{len(done)} samples are saved in {RESULTS_CSV.name}.")
                build_report(RESULTS_CSV)
                return

        tag = row.get("category") or row.get("content_type") or ""
        mic_note = ("BLE mic connected" if mic_mode == "ble" else "phone mic")
        print(f"\n[{i}/{len(rows)}] id={row['id']} {tag}  |  {mic_note}")
        print(f"  REFERENCE: {row['text']}")
        write_status("running", done, len(rows),
                     {"index": i, "id": row["id"], "reference": row["text"]}, started_at)
        try:
            result = process_sample(row, mic_xy, serial)
        except KeyboardInterrupt:
            write_status("paused", done, len(rows), None, started_at)
            print(f"\nInterrupted. {len(done)} samples are saved in {RESULTS_CSV.name}; "
                  "re-run with --resume to carry on from here.")
            return
        except Exception as e:
            result = blank_result(row, status=f"error:{e}")

        if result.get("status") == "stopped_by_request":
            clear_stop_request()
            write_status("paused", done, len(rows), None, started_at)
            print(f"\nStop requested. {len(done)} samples are saved in "
                  f"{RESULTS_CSV.name}; re-run to continue from here.")
            build_report(RESULTS_CSV)
            return

        # A hypothesis many times longer than its reference is not a bad
        # transcription, it is the wrong text. Scoring it would put a four-digit
        # WER into the averages and poison every figure in the run.
        ref_words = len(result["reference"].split())
        hyp_words = len(result["hypothesis"].split())
        if hyp_words > max(25, MAX_HYP_WORD_RATIO * ref_words):
            result["app_message"] = (f"transcript was {hyp_words} words for a "
                                     f"{ref_words}-word reference; not scored")
            result["status"] = "implausible_transcript"
            result["hypothesis"] = ""

        full_m = compute_full_metrics(result["reference"], result["hypothesis"])
        result.update(full_m)
        try:
            append_result(result)
        except ResultWriteError as e:
            write_status("paused", done, len(rows), None, started_at)
            notify("Results could not be saved",
                   f"{e}\n\nThe run stopped at sample {i} rather than scoring "
                   "samples that are not being recorded. Free some space, then "
                   "re-run to continue.")
            return
        done = [r for r in done if r.get("id") != result["id"]] + [result]

        if result["hypothesis"].strip():
            print(f"  App output : {result['hypothesis']}")
            notes = []
            if result.get("app_stopped_early") == "yes":
                notes.append("the app stopped recording before the audio finished")
            if str(result.get("segments") or "") not in ("", "0", "1"):
                notes.append(f"{result['segments']} segments joined")
            if notes:
                print(f"               note: {'; '.join(notes)}")
        else:
            timing = []
            if result.get("wait_s") not in ("", None):
                timing.append(f"waited {result['wait_s']}s for the session")
            if result.get("speaker_analysis_s") not in ("", None):
                timing.append(f"{result['speaker_analysis_s']}s in speaker analysis")
            note = result.get("app_message") or ""
            print(f"  App output : (none -- status={result['status']})")
            if note:
                print(f"               app said: {note}")
            if timing:
                print(f"               {', '.join(timing)}")
        print(f"  WER={full_m['WER']:.2f}% [{label_wer(full_m['WER'])}]  "
              f"CER={full_m['CER']:.2f}% [{label_cer(full_m['CER'])}]  "
              f"S={full_m['substitutions']} D={full_m['deletions']} I={full_m['insertions']}")
        print_running(done, len(rows), started_at)
        write_status("running", done, len(rows), None, started_at)

        if result["status"] != "ok" or not result["hypothesis"].strip():
            consecutive_failures += 1
            if consecutive_failures >= 3:
                write_status("paused", done, len(rows), None, started_at)
                print("\nThree failures in a row -- pausing. The phone has most likely "
                      "stopped hearing the speaker, locked, or lost adb.\n"
                      "Fix it, then re-run with --resume.")
                return
        else:
            consecutive_failures = 0

        time.sleep(INTER_SAMPLE_DELAY_S)

    write_status("completed", done, len(rows), None, started_at)
    build_report(RESULTS_CSV)
    print(f"\nFinished {processed} samples this run "
          f"({len(done)} in the results file) in "
          f"{(time.monotonic() - started_at) / 60:.1f} min.")
    print(f"Results: {RESULTS_CSV.name}")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", default=None, help="adb device serial (auto-detected if only one device)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("calibrate", help="Take a screenshot to help find the mic button's coordinates")

    p_set = sub.add_parser("set-mic-xy", help="Save the mic button's tap coordinates")
    p_set.add_argument("x", type=int)
    p_set.add_argument("y", type=int)

    p_run = sub.add_parser("run", help="Run the evaluation over the manifest")
    p_run.add_argument("--limit", type=int, default=None, help="Only process N samples")
    p_run.add_argument("--start", type=int, default=0, help="Skip the first N manifest rows")
    p_run.add_argument("--manifest", default=None,
                       help=f"Dataset CSV (default: first of {', '.join(MANIFEST_CANDIDATES)})")
    p_run.add_argument("--mic", choices=["phone", "ble"], default=None,
                       help="Which microphone this run uses (asked if omitted); recorded "
                            "against every row so phone and BLE results are never "
                            "averaged together")
    p_run.add_argument("--fresh", action="store_true",
                       help="Start a new results file instead of continuing the "
                            "most recent one (continuing is the default)")
    p_run.add_argument("--resume", action="store_true",
                       help=argparse.SUPPRESS)   # kept so older commands still work
    p_run.add_argument("--volume", default=None, help="PC volume setting, for the run record")
    p_run.add_argument("--distance-cm", default=None, help="Speaker-to-phone distance in cm")
    p_run.add_argument("--dba", default=None, help="Level measured at the phone, in dBA")
    p_run.add_argument("--no-dashboard", action="store_true",
                       help="Do not start the live dashboard with this run")
    p_run.add_argument("--analysis", choices=sorted(ANALYSIS_MODES), default="speaker",
                       help="which button produces the text to score: 'speaker' "
                            "taps SPEAKER ANALYSIS, 'transcript' taps the "
                            "TRANSCRIPT button")
    p_run.add_argument("--show-dashboard", action="store_true",
                       help="Always open a dashboard window, even if one is already open")

    p_sum = sub.add_parser("summarize", help="Recompute WER/CER from the latest results file")
    p_sum.add_argument("--file", default=None, help="Specific results_detailed_*.csv to read (default: latest)")

    p_full = sub.add_parser("full-metrics", help="Compute WER/CER/SER/OOV/S-D-I "
                                                  "per sample using semantic number normalization; "
                                                  "read-only, safe alongside a running 'run'")
    p_full.add_argument("--no-detail", action="store_true", help="Skip the per-sample printout, aggregate only")
    p_full.add_argument("--file", default=None, help="Specific results_detailed_*.csv to read (default: latest)")

    p_listen = sub.add_parser("listen", help="Play one sample locally to check audibility (no adb needed)")
    p_listen.add_argument("--id", default=None, help="Manifest id to play (default: first row)")

    args = parser.parse_args()

    if args.command == "calibrate":
        cmd_calibrate(args.serial)
    elif args.command == "set-mic-xy":
        cmd_set_mic_xy(args.x, args.y)
    elif args.command == "run":
        global ANALYSIS_MODE
        ANALYSIS_MODE = getattr(args, "analysis", "speaker")
        print(f"Scoring the output of the "
              f"{'TRANSCRIPT' if ANALYSIS_MODE == 'transcript' else 'SPEAKER ANALYSIS'} button.")
        cmd_run(args.serial, args.limit, args.start,
                manifest=Path(args.manifest) if args.manifest else None,
                mic_mode=args.mic, resume=not args.fresh,
                conditions={"pc_volume": args.volume,
                            "distance_cm": args.distance_cm,
                            "measured_dba": args.dba},
                dashboard=not args.no_dashboard,
                force_window=args.show_dashboard)
    elif args.command == "summarize":
        cmd_summarize(source=Path(args.file) if args.file else None)
    elif args.command == "full-metrics":
        cmd_full_metrics(detail=not args.no_detail, source=Path(args.file) if args.file else None)
    elif args.command == "listen":
        cmd_listen(args.id)


if __name__ == "__main__":
    main()
