#!/usr/bin/env python3
# Copied unchanged from Siddharth's bmssp-baselines (Week 2) so RMAT18/20 match RMAT16 exactly.

import argparse
import random


def generate_rmat_edge(scale, a, b, c, d):
    u = 0
    v = 0

    step = 1 << (scale - 1)

    while step > 0:

        r = random.random()

        if r < a:
            pass

        elif r < a + b:
            v += step

        elif r < a + b + c:
            u += step

        else:
            u += step
            v += step

        step >>= 1

    return u, v


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scale",
        type=int,
        required=True,
        help="Number of vertices = 2^scale"
    )

    parser.add_argument(
        "--edge-factor",
        type=int,
        default=8,
        help="Edges per vertex"
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    if args.scale < 2:
        raise ValueError(
            "Scale must be at least 2."
        )

    if args.edge_factor <= 0:
        raise ValueError(
            "Edge factor must be positive."
        )

    random.seed(args.seed)

    num_vertices = 1 << args.scale

    target_edges = (
        num_vertices * args.edge_factor
    )

    # Common Graph500-style RMAT probabilities.
    a = 0.57
    b = 0.19
    c = 0.19
    d = 0.05

    edges = set()

    while len(edges) < target_edges:

        u, v = generate_rmat_edge(
            args.scale,
            a,
            b,
            c,
            d
        )

        # Ignore self-loops.
        if u == v:
            continue

        # Directed graph.
        edge = (u, v)

        # Avoid duplicate arcs.
        if edge in edges:
            continue

        edges.add(edge)

    # Sort so output is deterministic/readable.
    edges = sorted(edges)

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "c Generated RMAT graph\n"
        )

        file.write(
            f"c Scale: {args.scale}\n"
        )

        file.write(
            f"c Edge factor: {args.edge_factor}\n"
        )

        file.write(
            f"c Seed: {args.seed}\n"
        )

        file.write(
            f"p sp {num_vertices} {len(edges)}\n"
        )

        for u, v in edges:

            weight = random.randint(
                1,
                100
            )

            # DIMACS IDs start at 1.
            file.write(
                f"a {u + 1} {v + 1} {weight}\n"
            )

    print(
        f"Generated RMAT scale {args.scale}"
    )

    print(
        f"Vertices: {num_vertices}"
    )

    print(
        f"Edges: {len(edges)}"
    )

    print(
        f"Edge factor: {args.edge_factor}"
    )

    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()
