#!/usr/bin/env python3

import glob
import os
import re
import statistics
from collections import defaultdict


folder = "results/week2/delta_sweep"

groups = defaultdict(
    lambda: {
        "time": [],
        "total": [],
        "iterations": [],
        "edges": []
    }
)


def get(pattern, text):
    m = re.search(pattern, text)
    return m.group(1) if m else None


for path in glob.glob(f"{folder}/*.txt"):

    name = os.path.basename(path)

    m = re.match(
        r"(.+)_delta(\d+)_rep(\d+)\.txt",
        name
    )

    if not m:
        continue


    dataset = m.group(1)
    delta = int(m.group(2))


    with open(path, encoding="utf-8") as f:
        text = f.read()


    correctness = get(
        r"Correctness\s*:\s*(\w+)",
        text
    )

    if correctness != "PASS":
        continue


    key = (dataset, delta)


    groups[key]["time"].append(
        float(
            get(
                r"GPU section\s*:\s*([0-9.eE+-]+)",
                text
            )
        )
    )


    groups[key]["total"].append(
        float(
            get(
                r"GPU total\s*:\s*([0-9.eE+-]+)",
                text
            )
        )
    )


    groups[key]["iterations"].append(
        int(
            get(
                r"Iterations\s*:\s*(\d+)",
                text
            )
        )
    )


    groups[key]["edges"].append(
        int(
            get(
                r"Edges examined\s*:\s*(\d+)",
                text
            )
        )
    )


print(
    f"{'Dataset':<10}"
    f"{'Delta':>8}"
    f"{'GPU med(ms)':>14}"
    f"{'Total med':>14}"
    f"{'Iterations':>14}"
    f"{'Edges':>14}"
)


for key in sorted(groups):

    dataset, delta = key

    values = groups[key]

    print(
        f"{dataset:<10}"
        f"{delta:>8}"
        f"{statistics.median(values['time']):>14.3f}"
        f"{statistics.median(values['total']):>14.3f}"
        f"{int(statistics.median(values['iterations'])):>14}"
        f"{int(statistics.median(values['edges'])):>14}"
    )
