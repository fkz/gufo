#!/usr/bin/env python3
"""Summarize an ordered Qwen expert trace without loading it into memory."""
import argparse
import collections
import csv


def summarize(path):
    counts = collections.defaultdict(collections.Counter)
    with open(path, newline="") as source:
        reader = csv.DictReader(source)
        columns = [f"expert_{i}" for i in range(10)]
        required = {"routing_id", "component", "layer", "token_row", *columns}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("expected an ordered expert trace CSV")
        for row in reader:
            group = (row["component"], int(row["layer"]))
            experts = [int(row[key]) for key in columns]
            if len(set(experts)) != 10 or any(e < 0 or e >= 512 for e in experts):
                raise ValueError(f"invalid expert IDs at CSV line {reader.line_num}")
            counts[group].update(experts)
    print("component,layer,selections,unique_experts,top64_pct,top128_pct,top256_pct")
    for (component, layer), histogram in sorted(counts.items()):
        total = histogram.total()
        ordered = [n for _, n in histogram.most_common()]
        coverage = [100 * sum(ordered[:n]) / total for n in (64, 128, 256)]
        print(f"{component},{layer},{total},{len(histogram)}," +
              ",".join(f"{value:.2f}" for value in coverage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    args = parser.parse_args()
    summarize(args.trace)
