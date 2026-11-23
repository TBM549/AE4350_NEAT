"""Aggregate validation-selected sensitivity controllers; no training or selection on evaluation."""

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sensitivity"


def model_runs():
    """Discover the newest complete run per setting and seed from model folders."""
    baseline, sensitivity = [], []
    selected = {}
    pattern = r"model_sensitivity_(matched_baseline|compatibility_threshold|node_add_prob|conn_add_prob|weight_mutate_power|survival_threshold)(?:_([0-9.]+))?_seed_(10|20|30)_.*"
    for folder in sorted((ROOT / "neat_models").iterdir()):
        match = re.fullmatch(pattern, folder.name)
        if match and folder.is_dir():
            parameter, value, seed = match.groups()
            is_baseline = parameter == "matched_baseline"
            if is_baseline != (value is None):
                continue
            row = dict(model=folder.name, training_seed=int(seed),
                       parameter="baseline" if is_baseline else parameter,
                       value="" if is_baseline else float(value))
            if not all((folder / name).is_file() for name in ("winner_genome.pkl", "validation.json", "evaluation.json")) or not list(folder.glob("config_*.ini")):
                continue
            key = (row["parameter"], row["value"], row["training_seed"])
            recency = (folder.stat().st_mtime_ns, folder.name)
            if key not in selected or recency > selected[key][0]:
                selected[key] = (recency, row)
    for _, row in sorted(selected.values(), key=lambda item: item[1]["model"]):
        (baseline if row["parameter"] == "baseline" else sensitivity).append(row)
    if len(baseline) != 3 or len(sensitivity) != 60:
        raise ValueError("Reports need 3 baseline and 60 sensitivity runs; run the full sweep first.")
    groups = {}
    for row in baseline + sensitivity:
        groups.setdefault((row["parameter"], row["value"]), []).append(row["training_seed"])
    if len(groups) != 21 or any(sorted(seeds) != [10, 20, 30] for seeds in groups.values()):
        raise ValueError("Each of the 21 report settings must have exactly seeds 10, 20, 30.")
    return dict(baseline=baseline, sensitivity=sensitivity)


def run():
    source = model_runs()
    rows = []
    for run in source["baseline"] + source["sensitivity"]:
        folder = ROOT / "neat_models" / run["model"]
        validation_path, evaluation_path = folder / "validation.json", folder / "evaluation.json"
        validation = json.loads(validation_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
        rows.append(dict(**run, validation_capture_pct=100*validation["winner"]["catch_rate"],
                         evaluation_capture_pct=100*evaluation["catch_rate"]))
    summary = []
    settings = list(dict.fromkeys((r["parameter"], r["value"]) for r in rows))
    for parameter, value in settings:
        group = [r for r in rows if (r["parameter"], r["value"]) == (parameter, value)]
        assert sorted(r["training_seed"] for r in group) == [10, 20, 30]
        result = dict(parameter=parameter, value=value)
        for metric in ("validation_capture_pct", "evaluation_capture_pct"):
            values = [r[metric] for r in group]
            result[metric+"_mean"] = float(np.mean(values))
            result[metric+"_sd"] = float(np.std(values, ddof=1))
        summary.append(result)
    assert len(rows) == 63 and len(summary) == 21
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, data in (("sensitivity_runs.csv", rows), ("sensitivity_summary.csv", summary)):
        with (OUT / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print("Aggregated 3 matched baseline and 60 sensitivity controllers.")


if __name__ == "__main__":
    run()
