#!/usr/bin/env python3

import csv
import glob
import os
import re


INPUT = "results/week2/baselines"
OUTPUT = "results/week2/baseline_results.csv"


def value(pattern, text):
    match = re.search(pattern, text)

    if not match:
        return ""

    return match.group(1)


rows = []


for path in sorted(glob.glob(f"{INPUT}/*.txt")):

    filename = os.path.basename(path)

    match = re.match(
        r"(.+)_source(\d+)_rep(\d+)_(dijkstra|cpu_delta|gpu_delta)\.txt",
        filename
    )

    if not match:
        continue


    dataset = match.group(1)
    source = int(match.group(2))
    repetition = int(match.group(3))
    algorithm = match.group(4)


    with open(path, encoding="utf-8") as file:
        text = file.read()


    row = {
        "dataset": dataset,
        "source": source,
        "repetition": repetition,
        "algorithm": algorithm,
        "vertices": value(
            r"Vertices\s*:\s*(\d+)",
            text
        ),
        "edges": value(
            r"Edges\s*:\s*(\d+)",
            text
        ),
        "delta": value(
            r"Delta\s*:\s*(\d+)",
            text
        ),
        "runtime_ms": "",
        "gpu_total_ms": "",
        "gpu_setup_ms": "",
        "gpu_section_ms": "",
        "gpu_d2h_ms": "",
        "edges_examined": value(
            r"Edges examined\s*:\s*(\d+)",
            text
        ),
        "successful_relaxations": value(
            r"Successful relaxations\s*:\s*(\d+)",
            text
        ),
        "mismatches": value(
            r"Mismatches\s*:\s*(\d+)",
            text
        ),
        "correctness": value(
            r"Correctness\s*:\s*(\w+)",
            text
        )
    }


    if algorithm == "dijkstra":

        row["runtime_ms"] = value(
            r"Runtime\s*:\s*([0-9.eE+-]+)",
            text
        )


    elif algorithm == "cpu_delta":

        row["runtime_ms"] = value(
            r"Delta CPU\s*:\s*([0-9.eE+-]+)",
            text
        )


    elif algorithm == "gpu_delta":

        row["gpu_total_ms"] = value(
            r"GPU total\s*:\s*([0-9.eE+-]+)",
            text
        )

        row["gpu_setup_ms"] = value(
            r"H2D/setup\s*:\s*([0-9.eE+-]+)",
            text
        )

        row["gpu_section_ms"] = value(
            r"GPU section\s*:\s*([0-9.eE+-]+)",
            text
        )

        row["gpu_d2h_ms"] = value(
            r"D2H\s*:\s*([0-9.eE+-]+)",
            text
        )


    rows.append(row)


fields = [
    "dataset",
    "source",
    "repetition",
    "algorithm",
    "vertices",
    "edges",
    "delta",
    "runtime_ms",
    "gpu_total_ms",
    "gpu_setup_ms",
    "gpu_section_ms",
    "gpu_d2h_ms",
    "edges_examined",
    "successful_relaxations",
    "mismatches",
    "correctness"
]


with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(rows)


print(
    f"Wrote {len(rows)} rows to {OUTPUT}"
)
