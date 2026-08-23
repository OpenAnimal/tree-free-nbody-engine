"""
Standardized variant benchmark protocol.

Every domain benchmark reports the same variant axes where applicable:

  standard    — the naive/dense/direct reference implementation
  +elastichash— the same computation restricted through the elastic-hash
                CellIndex neighborhood (near-field exact, far-field skipped
                or cluster-approximated)
  +fmm        — the core adaptive FMM / flat FMM engines (only where the 2D log
                kernel applies; otherwise this column is omitted with reason)
  +quantized  — quantized/bit-packed variant (where the module has one)

`VariantBenchmark` measures latency per variant and, when given an accuracy
callable, reports the repo-standard cross-validation error next to it, so
speed is never shown without the accuracy it costs.
"""

import json
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from core.validation import cross_validate


class VariantBenchmark:
    """Collects (variant name, work callable) pairs, times and validates them."""

    def __init__(self, title: str):
        self.title = title
        self._variants: List[Dict] = []

    def add(self, name: str, fn: Callable[[], np.ndarray],
            accuracy_vs: Optional[str] = None, repeats: int = 3,
            note: str = "") -> "VariantBenchmark":
        """
        Registers a variant. `fn` must return the computed result array.
        `accuracy_vs` names another registered variant to use as the exact
        reference for the cross-validation column (typically 'standard').
        """
        self._variants.append({"name": name, "fn": fn, "accuracy_vs": accuracy_vs,
                               "repeats": max(1, repeats), "note": note})
        return self

    def run(self, print_table: bool = True) -> List[Dict]:
        results = []
        outputs: Dict[str, np.ndarray] = {}
        for v in self._variants:
            # warmup
            try:
                outputs[v["name"]] = v["fn"]()
            except Exception as e:  # noqa: BLE001 - report, don't crash the table
                results.append({"variant": v["name"], "error": f"{type(e).__name__}: {e}",
                                "note": v["note"]})
                continue
            best = float("inf")
            for _ in range(v["repeats"]):
                t0 = time.perf_counter()
                outputs[v["name"]] = v["fn"]()
                best = min(best, time.perf_counter() - t0)
            row = {"variant": v["name"], "time_ms": best * 1000.0, "note": v["note"]}
            if v["accuracy_vs"] is not None and v["accuracy_vs"] in outputs:
                ref = outputs[v["accuracy_vs"]]
                try:
                    acc = cross_validate(outputs[v["name"]], ref, name=v["name"])
                    row["rel_l2"] = acc["rel_l2"]
                    row["mean_abs"] = acc["mean_abs"]
                except Exception as e:  # noqa: BLE001
                    row["rel_l2"] = float("nan")
                    row["accuracy_error"] = str(e)
            results.append(row)

        if print_table:
            self.print_table(results)
        return results

    @staticmethod
    def print_table(results: List[Dict]) -> None:
        print(f"\n{'Variant':<22} {'Time (ms)':>10} {'rel L2 vs ref':>14}  Note")
        print("-" * 78)
        base_time = next((r["time_ms"] for r in results if "time_ms" in r), None)
        for r in results:
            if "error" in r:
                print(f"{r['variant']:<22} {'ERROR':>10} {'-':>14}  {r['error']}")
                continue
            speedup = f" ({base_time / r['time_ms']:.1f}x)" if base_time and r is not results[0] else ""
            rel = f"{r['rel_l2']:.3e}" if "rel_l2" in r and np.isfinite(r.get("rel_l2", np.nan)) else "-"
            print(f"{r['variant']:<22} {r['time_ms']:>9.2f}{speedup:>7} {rel:>14}  {r.get('note', '')}")

    def save_json(self, path: str, results: Optional[List[Dict]] = None) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"title": self.title, "results": results or self.run(print_table=False)},
                      f, indent=2, default=float)
