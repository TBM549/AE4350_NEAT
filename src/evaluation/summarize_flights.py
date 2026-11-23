"""Count baseline outcomes by attempts and failure end time."""

from collections import Counter
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results/baseline"


def write_csv(name, rows, fields):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(seed=None):
    with (OUT / "episodes.csv").open() as stream:
        flights = list(csv.DictReader(stream))
    if seed is not None:
        flights = [flight for flight in flights if int(flight["training_seed"]) == seed]
    if not flights:
        raise ValueError(f"No flights found for seed {seed}.")
    attempt_distances = sorted({float(flight["attempt_distance_m"]) for flight in flights})
    suffix = "" if seed is None else f"_seed_{seed}"
    counts = Counter()
    failures = []
    for flight in flights:
        attempts = int(flight["attempts"])
        success = flight["caught"] == "True"
        assert attempts >= 0 and (not success or attempts >= 1)
        outcome = "success" if success else "failure"
        bucket = "6+" if attempts >= 6 else str(attempts)
        if not success and attempts == 0 and float(flight["minimum_distance_m"]) > 1.0:
            bucket = "-1"
        counts[outcome, bucket] += 1
        counts[outcome, "total"] += 1
        if not success:
            failures.append(flight)
    rows = [dict(outcome=outcome, attempts=bucket, flights=counts[outcome, bucket],
                 percentage_total=round(100*counts[outcome, bucket]/len(flights), 2))
            for outcome, buckets in (("success", ("total", "1", "2", "3", "4", "5", "6+")),
                                     ("failure", ("total", "-1", "0", "1", "2", "3", "4", "5", "6+")))
            for bucket in buckets]
    assert sum(row["flights"] for row in rows if row["attempts"] != "total") == len(flights)
    for outcome in ("success", "failure"):
        assert sum(row["flights"] for row in rows
                   if row["outcome"] == outcome and row["attempts"] != "total") == counts[outcome, "total"]
    write_csv(f"attempt_counts{suffix}.csv", rows, ["outcome", "attempts", "flights", "percentage_total"])


if __name__ == "__main__":
    run()
