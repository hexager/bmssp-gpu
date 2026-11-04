#!/usr/bin/env python3

import argparse
import random


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--edges", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.vertices < 2:
        raise ValueError("vertices must be >= 2")

    max_edges = args.vertices * (args.vertices - 1)

    if args.edges < 1 or args.edges > max_edges:
        raise ValueError(
            f"edges must be between 1 and {max_edges}"
        )

    random.seed(args.seed)

    edges = set()

    while len(edges) < args.edges:
        u = random.randrange(args.vertices)
        v = random.randrange(args.vertices)

        if u == v:
            continue

        if (u, v) in edges:
            continue

        weight = random.randint(1, 50)

        edges.add((u, v, weight))

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("c Random directed weighted graph\n")
        f.write(f"c Seed: {args.seed}\n")
        f.write(
            f"p sp {args.vertices} {len(edges)}\n"
        )

        for u, v, w in sorted(edges):
            f.write(
                f"a {u + 1} {v + 1} {w}\n"
            )

    print(
        f"Generated seed={args.seed} "
        f"V={args.vertices} "
        f"E={len(edges)}"
    )


if __name__ == "__main__":
    main()
