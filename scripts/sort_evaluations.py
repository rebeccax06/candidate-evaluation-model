#!/usr/bin/env python3
"""Sort exported evaluation rows (CSV) into an organized folder tree.

Organizes the ``result`` JSON of each evaluation into:

    <output>/holistic/<year>/<candidate>.json      # holistic evaluations, by year done
    <output>/role_based/<role>/<candidate>.json    # role-specific evals (clinician/engineer/phd)
    <output>/<other_type>/<candidate>.json          # anything else (e.g. interview_selection)

The "year" comes from the evaluation's own ``metadata.timestamp`` when available,
falling back to the row's ``created_at`` column.

Local role-specific result JSON files can also be ingested with --results-dir; only
evaluations carrying a recognized ``role`` are added (into role_based/<role>/).

Usage:
    python scripts/sort_evaluations.py /path/to/evaluations_rows.csv [--output organized_evaluations]
    python scripts/sort_evaluations.py /path/to/evaluations_rows.csv --results-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Roles recognized by the candidate evaluator (see candidate_evaluator/utils/role_results.py)
ROLE_NAMES = {"clinician", "engineer", "phd"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        help="Path to the exported evaluations CSV (optional if --results-dir is used).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Directory of local result *.json files to ingest (can be repeated). "
            "Role-specific evaluations are routed into role_based/<role>/."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("organized_evaluations"),
        help="Output directory for the sorted JSON files (default: organized_evaluations).",
    )
    parser.add_argument(
        "--year-map",
        type=Path,
        default=Path("candidates_by_year.txt"),
        help=(
            "Optional file mapping candidates to a year. Format: lines like '2025:' / '2026:' "
            "followed by candidate filenames. Used as the primary source for the holistic year "
            "(default: candidates_by_year.txt if present)."
        ),
    )
    return parser.parse_args()


def sanitize(name: str) -> str:
    """Make a string safe to use as a filename."""
    name = (name or "").strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unknown"


def normalize_key(value: str) -> str:
    """Normalize a candidate id / filename for matching (drop .pdf, strip whitespace)."""
    value = (value or "").strip()
    if value.lower().endswith(".pdf"):
        value = value[:-4]
    return re.sub(r"\s+", "", value).lower()


def load_year_map(path: Path) -> dict[str, str]:
    """Parse candidates_by_year.txt into {normalized_candidate: year}."""
    mapping: dict[str, str] = {}
    if not path or not path.exists():
        return mapping
    current_year: str | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry:
                continue
            if entry.endswith(":") and entry[:-1].strip().isdigit():
                current_year = entry[:-1].strip()
                continue
            if current_year:
                mapping.setdefault(normalize_key(entry), current_year)
    return mapping


def year_for(result: dict, row: dict, year_map: dict[str, str]) -> str:
    """Determine the year the evaluation was performed.

    Prefers the explicit candidates_by_year.txt mapping, then the evaluation's own
    metadata timestamp, then the row's created_at column.
    """
    candidate = row.get("candidate_id") or row.get("candidate_name") or ""
    mapped = year_map.get(normalize_key(candidate))
    if mapped:
        return mapped

    timestamp = (result.get("metadata") or {}).get("timestamp") or ""
    match = re.match(r"(\d{4})", timestamp)
    if match:
        return match.group(1)
    created = row.get("created_at") or ""
    match = re.match(r"(\d{4})", created)
    return match.group(1) if match else "unknown_year"


def role_for(result: dict) -> str | None:
    """Return canonical role if this is a role-specific evaluation, else None."""
    role = result.get("role")
    if role and str(role).lower() in ROLE_NAMES:
        return str(role).lower()
    return None


def destination(result: dict, row: dict, output: Path, year_map: dict[str, str]) -> Path:
    """Compute the target directory for an evaluation."""
    role = role_for(result)
    if role:
        return output / "role_based" / role

    eval_type = (row.get("evaluation_type") or "").strip().lower()
    if eval_type in ("", "holistic", "holistic_enhanced"):
        return output / "holistic" / year_for(result, row, year_map)
    return output / sanitize(eval_type)


def write_result(result: dict, target_dir: Path, candidate: str) -> None:
    """Write a single result JSON into target_dir, avoiding name clashes."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{sanitize(candidate)}.json"
    counter = 2
    while out_path.exists():
        out_path = target_dir / f"{sanitize(candidate)}_{counter}.json"
        counter += 1
    with out_path.open("w", encoding="utf-8") as out_handle:
        json.dump(result, out_handle, indent=2, ensure_ascii=False)


def candidate_from_result(result: dict, fallback: str) -> str:
    """Best-effort candidate identifier from a result JSON."""
    candidate = result.get("candidate")
    if isinstance(candidate, dict):
        return candidate.get("candidate_id") or candidate.get("name") or fallback
    return fallback


def process_csv(csv_path: Path, output: Path, year_map: dict[str, str], summary: Counter) -> tuple[int, int]:
    csv.field_size_limit(10 ** 8)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    written = skipped = 0
    for row in rows:
        raw = row.get("result")
        if not raw:
            skipped += 1
            continue
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            skipped += 1
            continue

        target_dir = destination(result, row, output, year_map)
        candidate = row.get("candidate_id") or row.get("candidate_name") or row.get("id") or "unknown"
        write_result(result, target_dir, candidate)
        written += 1
        summary[str(target_dir.relative_to(output))] += 1
    return written, skipped


def process_results_dir(directory: Path, output: Path, summary: Counter) -> tuple[int, int]:
    """Ingest local *.json result files; only role-specific ones are added."""
    written = skipped = 0
    for json_path in sorted(directory.glob("*.json")):
        try:
            with json_path.open(encoding="utf-8") as handle:
                result = json.load(handle)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        if not isinstance(result, dict):
            skipped += 1
            continue

        role = role_for(result)
        if not role:
            # Holistic local files come from the CSV/DB export; skip to avoid duplicates.
            skipped += 1
            continue

        target_dir = output / "role_based" / role
        candidate = candidate_from_result(result, json_path.stem)
        write_result(result, target_dir, candidate)
        written += 1
        summary[str(target_dir.relative_to(output))] += 1
    return written, skipped


def main() -> int:
    args = parse_args()

    if not args.csv_path and not args.results_dir:
        print("error: provide a CSV path and/or at least one --results-dir", file=sys.stderr)
        return 1

    year_map = load_year_map(args.year_map)
    if year_map:
        print(f"Loaded year mapping for {len(year_map)} candidates from {args.year_map}\n")

    summary: Counter[str] = Counter()
    written = skipped = 0

    if args.csv_path:
        if not args.csv_path.exists():
            print(f"error: CSV not found: {args.csv_path}", file=sys.stderr)
            return 1
        w, s = process_csv(args.csv_path, args.output, year_map, summary)
        written += w
        skipped += s

    for directory in args.results_dir:
        if not directory.exists():
            print(f"warning: results dir not found, skipping: {directory}", file=sys.stderr)
            continue
        w, s = process_results_dir(directory, args.output, summary)
        print(f"Added {w} role-specific evaluations from {directory}/")
        written += w
        skipped += s

    print(f"\nSorted {written} evaluations into {args.output}/ ({skipped} skipped)\n")
    for folder in sorted(summary):
        print(f"  {folder:<28} {summary[folder]:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
