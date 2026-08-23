"""Pytest root conftest: ensure the repo root is importable.

Test modules live under ``tests/<package>/`` and import source via
``from core.x import y`` etc. With ``tests/__init__.py`` present pytest's
prepend import mode already inserts the repo root, but this conftest makes
the same guarantee for standalone execution (``python -m tests.core.test_x``)
and for any tooling that imports test modules directly.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
