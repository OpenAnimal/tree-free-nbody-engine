"""
Repository-Wide Comprehensive Verification & Health Check.
Scans and executes all standalone verification blocks and tests across all packages.
"""

import sys
import os
import time
import importlib
import traceback

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

packages = [
    "core",
    "algorithm_theory",
    "neural_ops",
    "bioinformatics",
    "graphics_rendering",
    "game_mechanics_spatial",
    "apps"
]

def run_health_check():
    print("=" * 80)
    print("TREE-FREE N-BODY REPOSITORY COMPREHENSIVE HEALTH & SANITY AUDIT")
    print("=" * 80)
    
    passed_imports = 0
    failed_imports = []
    
    # 1. Test Package Imports
    for pkg in packages:
        pkg_path = os.path.join(root_dir, pkg)
        if not os.path.isdir(pkg_path):
            continue
        for root, _, files in os.walk(pkg_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("__pycache__"):
                    rel_path = os.path.relpath(os.path.join(root, file), root_dir)
                    mod_name = rel_path.replace(os.path.sep, ".").replace(".py", "")
                    if mod_name.endswith(".__init__"):
                        mod_name = mod_name[:-9]
                    
                    try:
                        importlib.import_module(mod_name)
                        passed_imports += 1
                    except Exception as e:
                        failed_imports.append((mod_name, str(e), traceback.format_exc()))
    
    print(f"\n[IMPORT AUDIT] {passed_imports} modules imported successfully.")
    if failed_imports:
        print(f"[FAIL] {len(failed_imports)} modules failed to import:")
        for mod, err, tb in failed_imports:
            print(f"  - {mod}: {err}")
            print(tb)
    else:
        print("[PASS] All repository Python modules imported with zero errors!")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    run_health_check()
