from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


EXPERIMENT_COLUMNS = [
    "date",
    "experiment",
    "seed",
    "params",
    "metric_name",
    "metric_value",
    "seconds",
    "peak_ram_mb",
]


def log_experiment(
    experiment: str,
    seed: int,
    params: dict[str, Any],
    metric_name: str,
    metric_value: float,
    seconds: float,
    peak_ram_mb: float,
    csv_path: str | Path = "results/experiments.csv",
) -> Path:
    """يسجل تجربة واحدة في ملف CSV بسيط وخفيف."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "seed": seed,
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "seconds": seconds,
        "peak_ram_mb": peak_ram_mb,
    }

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EXPERIMENT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return path

