#!/usr/bin/env python3
"""
Build the evaluation report from a results CSV.

    python make_report.py                          # newest results file
    python make_report.py --results results_detailed_20261009_154940.csv

Produces one workbook with two sheets:

  Summary     overall WER, CER, SER and S/D/I as figures and as bar charts,
              plus WER by category and the failure breakdown
  Per sample  one row per sample: id, category, reference, ASR output, metrics

Two averages are reported, because they answer different questions and they do
not agree. The corpus average, sum of all errors over sum of all reference
words, is the headline: it weights every word equally and is the figure used in
published ASR benchmarks. The per-sample average is the mean of the individual
WER percentages, which lets a three-word sentence count as much as a thirty-word
one. Showing only the second is how a run full of short samples ends up
reporting a number nobody can reproduce.

Failed and empty samples are counted but never averaged - a sample that
produced no transcript is not a sample the recogniser got wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl
from openpyxl.chart import BarChart, Reference
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
    ws.sheet_view.showGridLines = False
    ws["A1"] = "SmartScribe ASR evaluation - summary"
    ws["A1"].font = TITLE
    ws["A2"] = (f"{results.name}   |   {s['scored']} samples scored, "
                f"{s['failed']} failed, {s['total']} attempted")
    ws["A2"].font = SUB

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
    header(ws, row, ["Metric", "Value", "Meaning"], [26, 12, 62])
    row += 1
    figures = [
        ("WER % (corpus)", s["wer_micro"],
         "All errors divided by all reference words. The headline figure."),
        ("WER % (per sample)", s["wer_macro"],
         "Mean of the individual sample WERs. Short samples count as much as long ones."),
        ("CER % (per sample)", s["cer_macro"], "Character error rate, averaged per sample."),
        ("SER %", s["ser"], "Share of samples with at least one error."),
        ("Substitutions", s["substitutions"], "Reference word replaced by a different word."),
        ("Deletions", s["deletions"], "Reference word missing from the output."),
        ("Insertions", s["insertions"], "Extra word the recogniser added."),
        ("Reference words", s["reference_words"], "Total words scored against."),
        ("Samples scored", s["scored"], "Excludes failed and empty samples."),
        ("Samples failed", s["failed"], "Counted, never averaged into accuracy."),
    ]
    for label, value, meaning in figures:
        ws.cell(row=row, column=1, value=label).font = BODY
        cell = ws.cell(row=row, column=2, value=value)
        cell.font = BOLD
        cell.number_format = "0.00" if isinstance(value, float) else "0"
        ws.cell(row=row, column=3, value=meaning).font = SUB
        for c in range(1, 4):
            ws.cell(row=row, column=c).border = BORDER
        row += 1
    metrics_bottom = row - 1

    # ── chart data, kept out of the way to the right ──────────────────────
    data_col = 8
    ws.cell(row=4, column=data_col, value="Error rates").font = BOLD
    rates = [("WER (corpus)", s["wer_micro"]), ("WER (per sample)", s["wer_macro"]),
             ("CER", s["cer_macro"]), ("SER", s["ser"])]
    for i, (name, value) in enumerate(rates, start=5):
        ws.cell(row=i, column=data_col, value=name)
        ws.cell(row=i, column=data_col + 1, value=value)

    counts_top = 11
    ws.cell(row=counts_top, column=data_col, value="Error counts").font = BOLD
    for i, (name, value) in enumerate(
            [("Substitutions", s["substitutions"]), ("Deletions", s["deletions"]),
             ("Insertions", s["insertions"])], start=counts_top + 1):
        ws.cell(row=i, column=data_col, value=name)
        ws.cell(row=i, column=data_col + 1, value=value)

    cat_top = counts_top + 5
    ws.cell(row=cat_top, column=data_col, value="WER by category").font = BOLD
    top_cats = s["categories"][:12]
    for i, c in enumerate(top_cats, start=cat_top + 1):
        ws.cell(row=i, column=data_col, value=c["category"])
        ws.cell(row=i, column=data_col + 1, value=c["wer"])

    # ── charts ────────────────────────────────────────────────────────────
    chart1 = BarChart()
    chart1.type = "col"
    chart1.title = "Error rates (%)"
    chart1.y_axis.title = "%"
    chart1.legend = None
    chart1.height, chart1.width = 7.5, 12
    chart1.add_data(Reference(ws, min_col=data_col + 1, min_row=5, max_row=8), titles_from_data=False)
    chart1.set_categories(Reference(ws, min_col=data_col, min_row=5, max_row=8))
    ws.add_chart(chart1, f"A{metrics_bottom + 3}")

    chart2 = BarChart()
    chart2.type = "col"
    chart2.title = "Substitutions, deletions, insertions"
    chart2.y_axis.title = "words"
    chart2.legend = None
    chart2.height, chart2.width = 7.5, 12
    chart2.add_data(Reference(ws, min_col=data_col + 1, min_row=counts_top + 1,
                              max_row=counts_top + 3), titles_from_data=False)
    chart2.set_categories(Reference(ws, min_col=data_col, min_row=counts_top + 1,
                                    max_row=counts_top + 3))
    ws.add_chart(chart2, f"F{metrics_bottom + 3}")

    if top_cats:
        chart3 = BarChart()
        chart3.type = "bar"              # horizontal: category names stay readable
        chart3.title = "WER by category (%), worst first"
        chart3.x_axis.title = "%"
        chart3.legend = None
        chart3.height = max(7.5, 1.6 + 0.55 * len(top_cats))
        chart3.width = 16
        chart3.add_data(Reference(ws, min_col=data_col + 1, min_row=cat_top + 1,
                                  max_row=cat_top + len(top_cats)), titles_from_data=False)
        chart3.set_categories(Reference(ws, min_col=data_col, min_row=cat_top + 1,
                                        max_row=cat_top + len(top_cats)))
        ws.add_chart(chart3, f"A{metrics_bottom + 20}")

    # ── category table under the charts ───────────────────────────────────
    table_row = metrics_bottom + 40
    ws.cell(row=table_row - 1, column=1, value="WER by category").font = BOLD
    header(ws, table_row, ["Category", "Samples", "WER %", "Failed"], [26, 12, 12, 12])
    table_row += 1
    for c in s["categories"]:
        ws.cell(row=table_row, column=1, value=c["category"]).font = BODY
        ws.cell(row=table_row, column=2, value=c["n"]).font = BODY
        cell = ws.cell(row=table_row, column=3, value=c["wer"])
        cell.font = BODY
        cell.number_format = "0.00"
        ws.cell(row=table_row, column=4, value=c["failed"]).font = BODY
        for col in range(1, 5):
            ws.cell(row=table_row, column=col).border = BORDER
        table_row += 1

    if s["failure_causes"]:
        table_row += 2
        ws.cell(row=table_row - 1, column=1, value="Failures by cause").font = BOLD
        header(ws, table_row, ["Cause", "Samples"], [52, 12])
        table_row += 1
        for cause, n in s["failure_causes"].most_common():
            ws.cell(row=table_row, column=1, value=str(cause)[:120]).font = BODY
            ws.cell(row=table_row, column=2, value=n).font = BODY
            for col in (1, 2):
                ws.cell(row=table_row, column=col).border = BORDER
            table_row += 1


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


def write_detail(ws, rows: list[dict]) -> None:
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A2"
    header(ws, 1, [label for _, label, _ in DETAIL_COLUMNS],
           [w for _, _, w in DETAIL_COLUMNS])
    for i, r in enumerate(rows, start=2):
        for c, (key, _, _) in enumerate(DETAIL_COLUMNS, 1):
            value = r.get(key, "")
            if key in ("WER", "CER", "speaker_analysis_s"):
                value = num(r, key)
            elif key in ("SER", "substitutions", "deletions", "insertions"):
                v = num(r, key)
                value = int(v) if v is not None else ""
            cell = ws.cell(row=i, column=c, value=value)
            cell.font = BODY
            cell.alignment = Alignment(vertical="top",
                                       wrap_text=key in ("reference", "hypothesis",
                                                         "app_message"))
            if key in ("WER", "CER"):
                cell.number_format = "0.00"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(DETAIL_COLUMNS))}{len(rows) + 1}"


def build(results: Path, out: Path | None = None) -> Path:
    with results.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{results} has no rows")

    s = summarise(rows)
    meta = run_meta_for(results)
    out = out or results.with_name(results.stem.replace("results_detailed",
                                                        "report") + ".xlsx")

    wb = openpyxl.Workbook()
    write_summary(wb.active, s, meta, results)
    wb.active.title = "Summary"
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
