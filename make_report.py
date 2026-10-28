#!/usr/bin/env python3
"""
Build the evaluation report from a results CSV.

    python make_report.py                          # newest results file
    python make_report.py --results results_detailed_20261009_154940.csv

Produces one workbook with two sheets:

  Summary     overall WER, CER, SER and S/D/I as figures and as bar charts,
              plus WER by category and the failure breakdown
  Per sample  one row per sample: id, category, reference, ASR output, metrics

WER and CER are per-sample averages: the mean of each sample's own percentage.
Worth knowing what that means when the numbers are quoted - a three-word
sentence counts as much as a thirty-word one, so a run full of short samples
reads differently from one of long ones. The corpus-level figure (all errors
over all reference words) is still computed in `summarise` if it is ever wanted
back in the report.

Failed and empty samples are counted but never averaged - a sample that
produced no transcript is not a sample the recogniser got wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent

FONT = "Calibri"
HDR_FILL = PatternFill("solid", fgColor="1F3864")
HDR_FONT = Font(name=FONT, size=10, bold=True, color="FFFFFF")
BODY = Font(name=FONT, size=10)
BOLD = Font(name=FONT, size=10, bold=True)
TITLE = Font(name=FONT, size=14, bold=True, color="1F3864")
SUB = Font(name=FONT, size=9, italic=True, color="595959")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def num(row: dict, key: str) -> float | None:
    raw = str(row.get(key, "")).strip()
    if raw in ("", "None"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def latest_results() -> Path | None:
    files = sorted(HERE.glob("results_detailed_*.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if files:
        return files[0]
    legacy = HERE / "results_detailed.csv"
    return legacy if legacy.exists() else None


def run_meta_for(results: Path) -> dict:
    """App version, device, dataset and acoustic conditions for this run.

    Taken from status.json, which the run writes for the dashboard anyway, so
    the report keeps its header without a separate run file sitting in the
    folder. An older run_*.json is still read if one happens to be there.
    """
    status = HERE / "status.json"
    if status.exists():
        try:
            meta = json.loads(status.read_text(encoding="utf-8")).get("run_meta") or {}
            if meta:
                return meta
        except (OSError, json.JSONDecodeError):
            pass
    stamp = re.search(r"(\d{8}_\d{6})", results.name)
    if stamp:
        for candidate in HERE.glob(f"run_{stamp.group(1)}*.json"):
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def summarise(rows: list[dict]) -> dict:
    scored = [r for r in rows
              if r.get("status") == "ok" and (r.get("hypothesis") or "").strip()]
    failed = [r for r in rows if r not in scored]

    sub = sum(int(num(r, "substitutions") or 0) for r in scored)
    dele = sum(int(num(r, "deletions") or 0) for r in scored)
    ins = sum(int(num(r, "insertions") or 0) for r in scored)

    # Reference length drives the corpus average, so take it from the
    # normalised text that was actually scored where that was recorded.
    ref_words = 0
    for r in scored:
        text = (r.get("normalized_reference") or r.get("reference") or "")
        ref_words += len(text.split())

    wer_micro = ((sub + dele + ins) / ref_words * 100) if ref_words else 0.0
    wer_macro = (sum(num(r, "WER") or 0 for r in scored) / len(scored)) if scored else 0.0
    cer_macro = (sum(num(r, "CER") or 0 for r in scored) / len(scored)) if scored else 0.0
    ser = (sum(1 for r in scored if (num(r, "SER") or 0) > 0) / len(scored) * 100) if scored else 0.0

    by_cat: dict[str, list[float]] = defaultdict(list)
    cat_fail: Counter = Counter()
    for r in scored:
        cat = (r.get("content_type") or r.get("category") or "").strip() or "untagged"
        w = num(r, "WER")
        if w is not None:
            by_cat[cat].append(w)
    for r in failed:
        cat = (r.get("content_type") or r.get("category") or "").strip() or "untagged"
        cat_fail[cat] += 1

    categories = sorted(
        ({"category": c, "n": len(v), "wer": round(sum(v) / len(v), 2),
          "failed": cat_fail.get(c, 0)} for c, v in by_cat.items()),
        key=lambda d: -d["wer"])

    return {
        "total": len(rows), "scored": len(scored), "failed": len(failed),
        "wer_micro": round(wer_micro, 2), "wer_macro": round(wer_macro, 2),
        "cer_macro": round(cer_macro, 2), "ser": round(ser, 2),
        "substitutions": sub, "deletions": dele, "insertions": ins,
        "reference_words": ref_words,
        "categories": categories,
        "failure_causes": Counter(
            (r.get("app_message") or r.get("status") or "unknown") for r in failed),
    }


def header(ws, row: int, labels: list[str], widths: list[int]) -> None:
    for c, (label, w) in enumerate(zip(labels, widths, strict=False), 1):
        cell = ws.cell(row=row, column=c, value=label)
        cell.fill, cell.font = HDR_FILL, HDR_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[row].height = 26


def write_summary(ws, s: dict, meta: dict, results: Path) -> None:
    """Summary sheet: tables down the left, the chart to the right, and the
    values the chart plots parked out of sight in columns R and S."""
    ws.sheet_view.showGridLines = False
    ws["A1"] = "SmartScribe ASR evaluation - summary"
    ws["A1"].font = TITLE
    ws["A2"] = (f"{results.name}   |   {s['scored']} samples scored, "
                f"{s['failed']} failed, {s['total']} attempted")
    ws["A2"].font = SUB

    for col, width in zip("ABCD", (30, 14, 62, 12), strict=False):
        ws.column_dimensions[col].width = width

    # ── run record ────────────────────────────────────────────────────────
    row = 4
    ws.cell(row=row, column=1, value="Run").font = BOLD
    row += 1
    for key in ("run_id", "started_at", "app_version", "device_model",
                "dataset_file", "mic_mode", "pc_volume", "distance_cm",
                "measured_dba"):
        if meta.get(key):
            ws.cell(row=row, column=1, value=key.replace("_", " ")).font = BODY
            ws.cell(row=row, column=2, value=str(meta[key])).font = BODY
            row += 1
    if row == 5:
        ws.cell(row=row, column=1, value="no run record found").font = SUB
        row += 1

    # ── headline figures ──────────────────────────────────────────────────
    row += 1
    header(ws, row, ["Metric", "Value", "Meaning"], [30, 14, 58])
    row += 1
    for label, value, meaning in [
        ("Average WER %", s["wer_macro"],
         "Mean of the individual sample WERs."),
        ("Average CER %", s["cer_macro"], "Character error rate, averaged per sample."),
        ("SER %", s["ser"], "Share of samples with at least one error."),
        ("Substitutions", s["substitutions"], "Reference word replaced by a different word."),
        ("Deletions", s["deletions"], "Reference word missing from the output."),
        ("Insertions", s["insertions"], "Extra word the recogniser added."),
        ("Reference words", s["reference_words"], "Total words scored against."),
        ("Samples scored", s["scored"], "Excludes failed and empty samples."),
        ("Samples failed", s["failed"], "Counted, never averaged into accuracy."),
    ]:
        ws.cell(row=row, column=1, value=label).font = BODY
        cell = ws.cell(row=row, column=2, value=value)
        cell.font = BOLD
        cell.number_format = "0.00" if isinstance(value, float) else "0"
        cell.alignment = Alignment(horizontal="right")
        meaning_cell = ws.cell(row=row, column=3, value=meaning)
        meaning_cell.font = SUB
        meaning_cell.alignment = Alignment(vertical="center", wrap_text=False)
        for c in range(1, 4):
            ws.cell(row=row, column=c).border = BORDER
        row += 1

    # ── the charts and the values behind them ─────────────────────────────
    # Two charts, not one. Rates are percentages and S/D/I are counts: on a
    # shared axis the 5,290 substitutions flattened a 25.55% bar to a line.
    # The source values sit in columns R and S of this sheet rather than on a
    # separate tab, because Excel will not plot data from a hidden sheet and
    # the first thing it drops is the category labels under the bars.
    src_label, src_value = 18, 19          # columns R and S
    rates = [("Average WER %", s["wer_macro"]),
             ("Average CER %", s["cer_macro"]),
             ("Sentence error rate %", s["ser"])]
    counts = [("Substitutions", s["substitutions"]),
              ("Deletions", s["deletions"]),
              ("Insertions", s["insertions"])]

    def block(top: int, title: str, pairs: list) -> tuple[int, int]:
        cell = ws.cell(row=top, column=src_label, value=title)
        cell.font = BOLD
        for i, (name, value) in enumerate(pairs, start=top + 1):
            ws.cell(row=i, column=src_label, value=name).font = BODY
            v = ws.cell(row=i, column=src_value, value=value)
            v.font = BODY
            v.number_format = "0.00" if isinstance(value, float) else "0"
        return top + 1, top + len(pairs)

    r_top, r_bot = block(3, "Accuracy", rates)
    c_top, c_bot = block(r_bot + 2, "Error counts", counts)
    ws.column_dimensions["R"].width = 24
    ws.column_dimensions["S"].width = 12

    def bar(title: str, top: int, bottom: int, axis_title: str,
            y_max: float | None = None) -> BarChart:
        """One series per chart, so a distinct colour per bar carries no
        meaning on its own - each bar is named below it and labelled with its
        own value above it."""
        ch = BarChart()
        ch.type = "col"
        ch.title = title
        ch.legend = None
        ch.varyColors = True
        ch.dLbls = DataLabelList()
        ch.dLbls.showVal = True
        ch.dLbls.showSerName = False
        ch.dLbls.showCatName = False
        ch.dLbls.showLegendKey = False
        # Without these, some Excel versions hide the axes, which takes the
        # measure names away from under the bars.
        ch.x_axis.delete = False
        ch.y_axis.delete = False
        ch.y_axis.title = axis_title
        if y_max is not None:
            # Fixed 0-100 for the percentages. Letting Excel scale to the data
            # makes a 25% bar fill the chart on one run and a quarter of it on
            # the next, so two reports cannot be compared by eye.
            ch.y_axis.scaling.min = 0
            ch.y_axis.scaling.max = y_max
        ch.height, ch.width = 10, 17
        ch.add_data(Reference(ws, min_col=src_value, min_row=top, max_row=bottom),
                    titles_from_data=False)
        ch.set_categories(Reference(ws, min_col=src_label, min_row=top, max_row=bottom))
        return ch

    ws.add_chart(bar("Accuracy: WER, CER and SER", r_top, r_bot, "percent", 100), "F4")
    ws.add_chart(bar("Error counts: substitutions, deletions, insertions",
                     c_top, c_bot, "words"), "F24")
    ws.column_dimensions["E"].width = 3


DETAIL_COLUMNS = [
    ("id", "Sample ID", 12),
    ("content_type", "Category", 16),
    ("mic_mode", "Mic", 8),
    ("reference", "Reference", 60),
    ("hypothesis", "ASR output", 60),
    ("WER", "WER %", 10),
    ("CER", "CER %", 10),
    ("SER", "SER", 7),
    ("substitutions", "S", 6),
    ("deletions", "D", 6),
    ("insertions", "I", 6),
    ("speaker_analysis_s", "Analysis s", 11),
    ("analysis_field", "From field", 16),
    ("status", "Status", 20),
    ("app_message", "Note", 34),
]


BAND_FILL = PatternFill("solid", fgColor="F2F5FA")


def write_detail(ws, rows: list[dict]) -> None:
    """One row per sample, as a bordered table.

    Every cell is bordered and the rows are banded, because this sheet is read
    across: finding the ASR output that belongs to a given reference in an
    unruled grid of long text is the sort of thing that gets misread.
    """
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A2"
    header(ws, 1, [label for _, label, _ in DETAIL_COLUMNS],
           [w for _, _, w in DETAIL_COLUMNS])

    wrap = {"reference", "hypothesis", "app_message"}
    numeric = {"WER", "CER", "speaker_analysis_s"}
    integer = {"SER", "substitutions", "deletions", "insertions"}

    for i, r in enumerate(rows, start=2):
        for c, (key, _, _) in enumerate(DETAIL_COLUMNS, 1):
            value = r.get(key, "")
            if key in numeric:
                value = num(r, key)
            elif key in integer:
                v = num(r, key)
                value = int(v) if v is not None else ""
            cell = ws.cell(row=i, column=c, value=value)
            cell.font = BODY
            cell.border = BORDER
            cell.alignment = Alignment(
                vertical="top",
                horizontal="right" if key in numeric | integer else "left",
                wrap_text=key in wrap)
            if key in numeric:
                cell.number_format = "0.00"
            if i % 2 == 0:
                cell.fill = BAND_FILL

    last_col = get_column_letter(len(DETAIL_COLUMNS))
    ws.auto_filter.ref = f"A1:{last_col}{len(rows) + 1}"


def build(results: Path, out: Path | None = None) -> Path:
    with results.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{results} has no rows")

    s = summarise(rows)
    meta = run_meta_for(results)
    # One file per exit, never overwritten: the run id says which run it
    # belongs to, the timestamp says when during that run it was taken. Naming
    # it after the results CSV meant a resumed run overwrote its own earlier
    # reports, losing every snapshot but the last.
    run_id = str(meta.get("run_id") or "").strip()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"report_{run_id}_{stamp}.xlsx" if run_id else f"report_{stamp}.xlsx"
    out = out or results.parent / name

    wb = openpyxl.Workbook()
    summary_ws = wb.active
    # Rename BEFORE the charts are built. openpyxl writes the sheet name into
    # every chart reference at the moment the Reference is created, so renaming
    # afterwards leaves each chart pointing at a sheet that no longer exists -
    # the frames and titles still draw, with no bars in them.
    summary_ws.title = "Summary"
    write_summary(summary_ws, s, meta, results)
    write_detail(wb.create_sheet("Per sample"), rows)
    wb.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the evaluation report")
    ap.add_argument("--results", type=Path, default=None,
                    help="results CSV (default: the newest one)")
    ap.add_argument("--out", type=Path, default=None)
    cfg = ap.parse_args()

    results = cfg.results or latest_results()
    if results is None or not results.exists():
        print("No results file found. Run an evaluation first.")
        return 1

    out = build(results, cfg.out)
    print(f"Report written: {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())