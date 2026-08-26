#!/usr/bin/env python3
"""One-off: generate a single dose-response variant near target_errors,
without touching generate_variants.py or its TARGETS list.

Not part of upstream agent-interp-envs; local addition for shortcut_forensics
Phase 2 easy-control (SPEC.md n_type_errors_easy). Usage:
    python _gen_easy_variant.py 20
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_variants import (  # noqa: E402
    BASE_DIR,
    build_unit_registry,
    build_typing_order,
    write_variant,
    get_mypy_error_count,
    SRC_UNTYPED,
)


def main(target: int) -> None:
    units = build_unit_registry()
    order = build_typing_order(units)
    actual_258 = get_mypy_error_count(SRC_UNTYPED)
    print(f"src_258/: {actual_258} errors; {len(order)} typeable units")

    cache = {0: actual_258}
    tmp = BASE_DIR / "_tmp_easy_variant"

    def errors_for_n(n: int) -> int:
        if n in cache:
            return cache[n]
        names = {u.name for u in order[:n]}
        write_variant(tmp, names, units)
        c = get_mypy_error_count(tmp)
        cache[n] = c
        return c

    lo, hi = 0, len(order)
    best_n, best_diff = 0, abs(actual_258 - target)
    while lo <= hi:
        mid = (lo + hi) // 2
        errs = errors_for_n(mid)
        diff = abs(errs - target)
        if diff < best_diff:
            best_diff, best_n = diff, mid
        if errs > target:
            lo = mid + 1
        elif errs < target:
            hi = mid - 1
        else:
            break
    for n in (best_n - 1, best_n, best_n + 1):
        if 0 <= n <= len(order):
            errs = errors_for_n(n)
            if abs(errs - target) < best_diff:
                best_diff, best_n = abs(errs - target), n

    actual = errors_for_n(best_n)
    out_dir = BASE_DIR / f"src_{actual}"
    names = {u.name for u in order[:best_n]}
    write_variant(out_dir, names, units)
    verify = get_mypy_error_count(out_dir)
    print(f"-> src_{actual}/ : {verify} errors (target {target}, typed {best_n} units)")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
