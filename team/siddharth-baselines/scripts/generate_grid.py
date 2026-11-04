#!/usr/bin/env python3

import argparse
import random


def vertex_id(row, col, width):
    return row * width + col + 1


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--size",
        type=int,
        required=True
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


    if args.size < 2:
        raise ValueError(
            "Grid size must be at least 2."
        )


    random.seed(args.seed)

    n = args.size

    edges = []


    for row in range(n):

        for col in range(n):

            u = vertex_id(
                row,
                col,
                n
            )


            # Horizontal neighbour.
            if col + 1 < n:

                v = vertex_id(
                    row,
                    col + 1,
                    n
                )

                weight1 = random.randint(
                    1,
                    100
                )

                weight2 = random.randint(
                    1,
                    100
                )

                edges.append(
                    (u, v, weight1)
                )

                edges.append(
                    (v, u, weight2)
                )


            # Vertical neighbour.
            if row + 1 < n:

                v = vertex_id(
                    row + 1,
                    col,
                    n
                )

                weight1 = random.randint(
                    1,
                    100
                )

                weight2 = random.randint(
                    1,
                    100
                )

                edges.append(
                    (u, v, weight1)
                )

                edges.append(
                    (v, u, weight2)
                )


    vertices = n * n


    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "c Generated 2D grid graph\n"
        )

        file.write(
            f"c Grid size: {n} x {n}\n"
        )

        file.write(
            f"c Seed: {args.seed}\n"
        )

        file.write(
            f"p sp {vertices} {len(edges)}\n"
        )


        for u, v, weight in edges:

            file.write(
                f"a {u} {v} {weight}\n"
            )


    print(
        f"Generated {n}x{n} grid"
    )

    print(
        f"Vertices: {vertices}"
    )

    print(
        f"Edges: {len(edges)}"
    )

    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()
