# SmartScribe ASR evaluation

Automated accuracy testing for the SmartScribe Android app. A PC plays reference
audio, the phone transcribes it, and the transcript is scored against the known
text - with nobody in the room.

The point is that a change in the numbers can be attributed to the app rather
than to the way it was tested, so the playback level, the acoustic conditions
and the app build are recorded with every run.

## What is here

| Script | What it does |
|---|---|
| `evaluate_asr.py` | The harness: records, plays a clip, reads the transcript back, scores it |
| `dashboard.py` | Live control panel in a browser window: start, stop, resume, and watch a run |
| `make_report.py` | Excel report: summary with bar charts, plus one row per sample |

`evaluate_asr.py` runs on its own. The other two are optional: without
`make_report.py` a run prints a note and skips the Excel report, and without
`dashboard.py` everything works from the terminal.

## Setup

```bash
pip install -r requirements.txt
```

Point the scripts at adb if it is not on your PATH:

```bash
# Windows
set ADB_PATH=C:\path\to\platform-tools\adb.exe
# macOS and Linux
export ADB_PATH=/path/to/platform-tools/adb
```

The phone needs USB debugging on and a **debug** build of the app installed -
the harness reads the app's own session store over `run-as`, which a release
build refuses.

## The dataset is not in this repository

The corpus is about 5.4 GB of audio and the manifest runs to tens of thousands
of rows. Keep both on shared storage. The scripts find the audio by the path in
each manifest row, resolved against the manifest's own folder, so putting the
manifest beside the audio folder is enough.

CI rejects any pull request containing audio files or anything over 5 MB. A
large file committed once stays in the history forever and every future clone
pays for it.

## Running an evaluation

```bash
python dashboard.py          # opens a window; start, stop and resume from there
```

or from the terminal:

```bash
python evaluate_asr.py run --limit 10 --volume 70 --distance-cm 20 --dba 65
```

Pass the acoustic conditions. Without them the run has no record of how it was
measured, and its numbers cannot be compared with any other run.

Continuing is the default: re-run the same command after an interruption and it
picks up where it stopped. `--fresh` starts a new results file.

`--analysis transcript` scores the TRANSCRIPT button; `--analysis speaker`
scores SPEAKER ANALYSIS. The two are never comparable to each other, so keep
them in separate results files.

## Output

Each run writes a timestamped results CSV, an Excel report, and `status.json`
for the dashboard. The report has a summary sheet with bar charts (error rates,
S/D/I, WER by category) and a per-sample sheet.

Two WER figures are reported and they do not agree. The corpus average - all
errors over all reference words - is the headline, because it weights every
word equally and matches published benchmarks. The per-sample average is shown
beside it; it lets a three-word sentence count as much as a thirty-word one.

## Tests and CI

```bash
pip install -r requirements-dev.txt
pytest tests -v
```

The audio layer is stubbed, so everything that does not need hardware runs on
any machine and in CI: normalisation, the metric counts, manifest loading and
the safety guards. The device-dependent paths are verified by a short
calibration run against a real phone.

CI runs on every push and pull request:

- **Tests** on Python 3.10 and 3.12
- **Lint** with ruff - real errors fail the build, style opinions do not
- **Large-file guard** - rejects committed audio or anything over 5 MB

### Known bugs, recorded as expected failures

Three tests are marked `xfail` because they describe real defects rather than
wrong expectations. They are strict, so each will fail the build the moment it
is fixed and the marker should then be removed:

1. **Times do not match their spoken form.** `eight twenty` normalises to `820`
   while `8:20` normalises to `8 20`, so a correct transcript of a time scores
   as wrong. Affects the `times` category.
2. **Consecutive number words are glued together.** `one two three` becomes
   `123`. Where one side has digits and the other words, they cannot match.
   Affects `numbers` and `phone_numbers`.
3. **Normalisation is not idempotent.** A trailing `...` survives the first
   pass as a bare `.` token and is scored as a word.

All three are in the number-handling path, between the Whisper normaliser and
the script's own number canonicaliser.
"# ASR_MERTICS_EVALAUTION" 
