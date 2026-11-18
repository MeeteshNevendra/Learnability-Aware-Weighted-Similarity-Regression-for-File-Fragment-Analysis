"""Minimal test runner (no pytest needed):  python -I tests/run_tests.py"""
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
failed = passed = 0
for fn in sorted(os.listdir(HERE)):
    if not (fn.startswith('test_') and fn.endswith('.py')):
        continue
    spec = importlib.util.spec_from_file_location(fn[:-3], os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in sorted(dir(mod)):
        if name.startswith('test_'):
            try:
                getattr(mod, name)()
                passed += 1
                print(f'PASS {fn}::{name}')
            except Exception:  # noqa: BLE001
                failed += 1
                print(f'FAIL {fn}::{name}')
                traceback.print_exc()
print(f'{passed} passed, {failed} failed')
sys.exit(1 if failed else 0)
