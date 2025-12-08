#!/usr/bin/env python3
"""Fetch GitHub traffic stats and build a persistent CSV plus trend chart."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import requests

API_ROOT = "https://api.github.com"


@dataclass
class DailyMetrics:
    date: str
    views: int
    unique_views: int
    clones: int
    unique_clones: int


def _fetch(endpoint: str, repo: str, token: str) -> Mapping:
    url = f"{API_ROOT}/repos/{repo}/{endpoint}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def _normalize_daily(items: Iterable[Mapping], count_key: str) -> Dict[str, Dict[str, int]]:
    daily: Dict[str, Dict[str, int]] = {}
    for item in items:
        # GitHub returns timestamps like "2024-12-05T00:00:00Z".
        date_key = item["timestamp"][:10]
        counts = daily.setdefault(date_key, {"count": 0, "uniques": 0})
        counts["count"] = max(counts["count"], int(item[count_key]))
        counts["uniques"] = max(counts["uniques"], int(item["uniques"]))
    return daily


def _load_existing(path: Path) -> Dict[str, DailyMetrics]:
    if not path.exists():
        return {}
    existing: Dict[str, DailyMetrics] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            existing[row["date"]] = DailyMetrics(
                date=row["date"],
                views=int(row["views"]),
                unique_views=int(row["unique_views"]),
                clones=int(row["clones"]),
                unique_clones=int(row["unique_clones"]),
            )
    return existing


def _merge(existing: Dict[str, DailyMetrics], updates: Dict[str, DailyMetrics]) -> Dict[str, DailyMetrics]:
    merged = existing.copy()
    for date_key, metrics in updates.items():
        if date_key in merged:
            prev = merged[date_key]
            merged[date_key] = DailyMetrics(
                date=date_key,
                views=max(prev.views, metrics.views),
                unique_views=max(prev.unique_views, metrics.unique_views),
                clones=max(prev.clones, metrics.clones),
                unique_clones=max(prev.unique_clones, metrics.unique_clones),
            )
        else:
            merged[date_key] = metrics
    return merged


def _write_csv(metrics: Dict[str, DailyMetrics], path: Path) -> None:
    ordered_dates = sorted(metrics.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["date", "views", "unique_views", "clones", "unique_clones"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for date_key in ordered_dates:
            record = metrics[date_key]
            writer.writerow(
                {
                    "date": record.date,
                    "views": record.views,
                    "unique_views": record.unique_views,
                    "clones": record.clones,
                    "unique_clones": record.unique_clones,
                }
            )


def _latest(metrics: Dict[str, DailyMetrics], limit: int = 7) -> Sequence[DailyMetrics]:
    if not metrics:
        return []
    ordered_dates = sorted(metrics.keys())
    recent_dates = ordered_dates[-limit:]
    return [metrics[date_key] for date_key in recent_dates]


def _render_table(metrics: Dict[str, DailyMetrics], max_rows: int = 7) -> str:
    header = "| Date | Views | Unique views | Clones | Unique clones |\n| --- | --- | --- | --- | --- |"
    if not metrics:
        return header + "\n| - | - | - | - | - |"
    rows = [
        f"| {item.date} | {item.views} | {item.unique_views} | {item.clones} | {item.unique_clones} |"
        for item in _latest(metrics, max_rows)
    ]
    return header + "\n" + "\n".join(rows)


def _update_readme(readme_path: Path, metrics: Dict[str, DailyMetrics], image_path: Path) -> None:
    start_marker = "<!-- traffic:start -->"
    end_marker = "<!-- traffic:end -->"
    block = f"{start_marker}\n{_render_table(metrics)}\n\n![Traffic trend]({image_path.as_posix()})\n{end_marker}"

    if not readme_path.exists():
        print(f"README not found at {readme_path}, skip embedding metrics.")
        return

    content = readme_path.read_text(encoding="utf-8")
    if start_marker in content and end_marker in content:
        pre, rest = content.split(start_marker, 1)
        _, post = rest.split(end_marker, 1)
        new_content = pre + block + post
    else:
        new_content = content.rstrip() + "\n\n" + block + "\n"

    readme_path.write_text(new_content, encoding="utf-8")


def _plot(metrics: Dict[str, DailyMetrics], output_path: Path) -> None:
    if not metrics:
        print("No traffic data available to plot.")
        return

    ordered = [metrics[key] for key in sorted(metrics.keys())]
    dates = [datetime.strptime(item.date, "%Y-%m-%d") for item in ordered]
    view_counts = [item.views for item in ordered]
    view_uniques = [item.unique_views for item in ordered]
    clone_counts = [item.clones for item in ordered]
    clone_uniques = [item.unique_clones for item in ordered]

    plt.figure(figsize=(10, 6))
    plt.plot(dates, view_counts, label="Views", linewidth=2)
    plt.plot(dates, view_uniques, label="Unique views", linestyle="--", linewidth=1.5)
    plt.plot(dates, clone_counts, label="Downloads (clones)", linewidth=2)
    plt.plot(dates, clone_uniques, label="Unique downloaders", linestyle="--", linewidth=1.5)
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.title("PaddleMaterials repository traffic")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def collect_metrics(repo: str, token: str) -> Dict[str, DailyMetrics]:
    views_resp = _fetch("traffic/views?per=day", repo, token)
    clones_resp = _fetch("traffic/clones?per=day", repo, token)

    view_daily = _normalize_daily(views_resp.get("views", []), count_key="count")
    clone_daily = _normalize_daily(clones_resp.get("clones", []), count_key="count")

    # Combine the two sources into a consistent DailyMetrics map.
    consolidated: Dict[str, DailyMetrics] = {}
    all_dates = set(view_daily.keys()) | set(clone_daily.keys())
    for date_key in all_dates:
        views = view_daily.get(date_key, {"count": 0, "uniques": 0})
        clones = clone_daily.get(date_key, {"count": 0, "uniques": 0})
        consolidated[date_key] = DailyMetrics(
            date=date_key,
            views=views["count"],
            unique_views=views["uniques"],
            clones=clones["count"],
            unique_clones=clones["uniques"],
        )
    return consolidated


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist GitHub traffic metrics and draw a trend plot.")
    parser.add_argument("--repo", required=True, help="Repository in owner/name format.")
    parser.add_argument(
        "--output-dir",
        default="output/traffic",
        help="Directory for CSV and chart outputs (default: output/traffic).",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"),
        help="GitHub token with repo access (defaults to GITHUB_TOKEN env).",
    )
    parser.add_argument(
        "--readme",
        default=None,
        help="Path to README to embed the latest traffic table and chart (optional).",
    )
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Missing GitHub token (set GITHUB_TOKEN or pass --token).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "traffic_metrics.csv"
    plot_path = output_dir / "traffic_trend.png"

    latest = collect_metrics(repo=args.repo, token=args.token)
    existing = _load_existing(csv_path)
    merged = _merge(existing, latest)
    _write_csv(merged, csv_path)
    _plot(merged, plot_path)
    if args.readme:
        _update_readme(Path(args.readme), merged, plot_path)
    print(f"Wrote {csv_path} and {plot_path}")


if __name__ == "__main__":
    main()
