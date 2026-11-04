#!/usr/bin/env python3

import csv
import statistics
from collections import defaultdict


INPUT = "results/week2/baseline_results.csv"

groups = defaultdict(list)


with open(INPUT, encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        algorithm = row["algorithm"]

        if algorithm == "gpu_delta":
            value = row["gpu_section_ms"]
        else:
            value = row["runtime_ms"]

        if not value:
            continue

        groups[
            (
                row["dataset"],
                algorithm
            )
        ].append(
            float(value)
        )


print(
    f"{'Dataset':<12}"
    f"{'Algorithm':<15}"
    f"{'N':>5}"
    f"{'Median(ms)':>14}"
    f"{'Mean(ms)':>14}"
    f"{'Min(ms)':>12}"
    f"{'Max(ms)':>12}"
)


for key in sorted(groups):

    dataset, algorithm = key

    values = groups[key]

    print(
        f"{dataset:<12}"
        f"{algorithm:<15}"
        f"{len(values):>5}"
        f"{statistics.median(values):>14.4f}"
        f"{statistics.mean(values):>14.4f}"
        f"{min(values):>12.4f}"
        f"{max(values):>12.4f}"
    )
