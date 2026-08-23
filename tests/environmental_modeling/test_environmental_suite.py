"""Round-7 Workstream G: test suite for all four environmental challenges.

Run:  python -X utf8 -m tests.environmental_modeling.test_environmental_suite
"""
from __future__ import annotations
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from environmental_modeling.groundwater_plume import (
    test_groundwater_plume,
    test_groundwater_pure_diffusion_analytic,
    test_groundwater_advected_analytic,
)
from environmental_modeling.airborne_exposure import (
    test_airborne_exposure,
    test_well_mixed_anchor,
    test_wall_symmetry,
)
from environmental_modeling.electrolyte_screening import (
    test_electrolyte_screening,
    test_debye_tail_fit,
    test_stillinger_lovett_moment,
)
from environmental_modeling.radiotherapy_dose import (
    test_radiotherapy_dose,
    test_superposition_dose_erf_anchor,
    test_superposition_dose_linearity,
    test_superposition_dose_convergence,
)


def main():
    print("=" * 70)
    print("Workstream G: Environmental & Health Physics Modeling Suite")
    print("=" * 70)
    results = []
    for name, fn in [
        ("T-G3 groundwater plume", test_groundwater_plume),
        ("T-G3 groundwater pure-diffusion analytic", test_groundwater_pure_diffusion_analytic),
        ("T-G3 groundwater advected analytic", test_groundwater_advected_analytic),
        ("T-G4 airborne exposure", test_airborne_exposure),
        ("T-G4 well-mixed anchor", test_well_mixed_anchor),
        ("T-G4 wall-symmetry", test_wall_symmetry),
        ("T-G2 electrolyte screening", test_electrolyte_screening),
        ("T-G2 Debye tail fit (net-charged)", test_debye_tail_fit),
        ("T-G2 Stillinger-Lovett second-moment", test_stillinger_lovett_moment),
        ("T-G1 radiotherapy dose", test_radiotherapy_dose),
        ("T-G1 superposition dose erf anchor", test_superposition_dose_erf_anchor),
        ("T-G1 superposition dose linearity", test_superposition_dose_linearity),
        ("T-G1 superposition dose convergence", test_superposition_dose_convergence),
    ]:
        print(f"\n--- {name} ---")
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append((name, False))

    print("\n" + "=" * 70)
    n_pass = sum(1 for _, ok in results if ok)
    n_total = len(results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{n_pass}/{n_total} challenges passed")
    if n_pass < n_total:
        raise SystemExit(1)
    print("\nAll environmental modeling challenges PASS")


if __name__ == "__main__":
    main()
