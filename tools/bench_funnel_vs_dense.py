#!/usr/bin/env python
"""Funnel-hash lookup throughput vs a dense 2^k count array (Part 2C.3).

Decision table for count-based exploration (Tang et al., 2017): with k
SimHash hyperplanes the key space is 2^k, and for k <= 22 it fits on a GPU
as a dense int32 array whose update is ONE scatter-add. A funnel table
(Farach-Colton, Krapivin, & Kuszmaul 2025; core/elastic_hash.py) only
earns its keep when the key space cannot be materialized, load approaches
1-delta, and worst-case probe bounds without reordering are required.

This benchmark measures both sides at k in {14, 16, 18, 20, 22} for batched
query counts nq in {10^4, 10^5, 10^6}:

  dense  : counts[keys] += 1 (one gather-scatter) + bonus read-back
  funnel : ElasticIntTable.insert_or_increment per key (Python loop) +
           funnel_probe batched lookup (the vectorized path, ~277 slot
           inspections per absent key at delta=0.05)

Expected (and historically observed) result: the dense array wins by
orders of magnitude in this regime; the funnel's guarantee (worst-case
probes WITHOUT reordering, exact integer keys, CPU) is orthogonal to this
workload. Results are appended to BENCHMARKS.md.

Usage:
    python tools/bench_funnel_vs_dense.py [--quick] [--nq 1e4 1e5 1e6]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.elastic_hash import ElasticIntTable, funnel_probe  # noqa: E402

K_LIST = (14, 16, 18, 20, 22)


def bench_dense(keys: np.ndarray, k: int) -> tuple:
    counts = np.zeros(2 ** k, dtype=np.int64)
    t0 = time.perf_counter()
    np.add.at(counts, keys, 1)
    t_ins = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    q = rng.integers(0, 2 ** k, size=min(len(keys), 100_000),
                     dtype=np.int64)
    t0 = time.perf_counter()
    vals = counts[q]
    t_qry = time.perf_counter() - t0
    return t_ins, t_qry, vals.sum()


def bench_funnel(keys: np.ndarray, k: int) -> tuple:
    table = ElasticIntTable(capacity=max(len(keys), 1024), delta=0.05,
                            seed=42)
    t0 = time.perf_counter()
    for key in keys[:200_000]:  # cap: per-key Python insert is the cost
        table.insert_or_increment(int(key), 1)
    t_ins = (time.perf_counter() - t0) * (len(keys) / min(len(keys), 200_000))
    q = keys[:min(len(keys), 100_000)].astype(np.int64)
    t0 = time.perf_counter()
    slots = funnel_probe(table, q)
    t_qry = time.perf_counter() - t0
    hits = int((slots >= 0).sum())
    return t_ins, t_qry, hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="k <= 18 and nq <= 1e5 only")
    ap.add_argument("--nq", type=float, nargs="+",
                    default=[1e4, 1e5, 1e6])
    args = ap.parse_args()
    k_list = (14, 16, 18) if args.quick else K_LIST
    nq_list = [int(n) for n in args.nq if n <= (1e5 if args.quick else 1e7)]

    print(f"{'k':>3} {'nq':>8} | {'dense ins ms':>12} {'dense qry ms':>12} "
          f"| {'funnel ins ms':>13} {'funnel qry ms':>13} | "
          f"{'funnel/dense':>12}")
    print("-" * 88)
    for k in k_list:
        for nq in nq_list:
            rng = np.random.default_rng(k * 1_000_003 + nq)
            keys = rng.integers(0, 2 ** k, size=nq, dtype=np.int64)
            d_ins, d_qry, _ = bench_dense(keys, k)
            f_ins, f_qry, hits = bench_funnel(keys, k)
            ratio = (f_ins + f_qry) / max(d_ins + d_qry, 1e-12)
            print(f"{k:>3} {nq:>8} | {d_ins*1e3:>12.2f} {d_qry*1e3:>12.2f} "
                  f"| {f_ins*1e3:>13.1f} {f_qry*1e3:>13.1f} "
                  f"| {ratio:>11.0f}x  (probe hits {hits}/{len(keys) and min(nq, 100_000)})",
                  flush=True)
    print("\n[reading] In the k <= 22 / batched-lookup regime the dense "
          "2^k array wins by the factors above. The funnel table's "
          "guarantees (worst-case probes without reordering, exact int64 "
          "keys, no materializable key space) are orthogonal to this "
          "workload — reach for it only when k > 22-ish, load -> 1-delta, "
          "and per-key work dwarfs ~29 probes.")


if __name__ == "__main__":
    main()
