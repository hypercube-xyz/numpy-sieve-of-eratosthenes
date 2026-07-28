import argparse
import platform
from collections.abc import Callable
from statistics import median
from timeit import repeat

import numpy as np

from eratosthenes import count_primes, eratosthenes

Operation = Callable[[int], int | np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the NumPy Sieve of Eratosthenes."
    )
    parser.add_argument(
        "limits",
        nargs="*",
        type=int,
        default=[1_000_000, 10_000_000],
    )
    parser.add_argument(
        "--operation",
        choices=("find", "count", "both"),
        default="both",
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if any(limit < 0 for limit in args.limits):
        parser.error("limits must be greater than or equal to 0")
    if args.repeats < 1:
        parser.error("--repeats must be greater than 0")
    return args


def _measure(function: Operation, limit: int, repeats: int) -> tuple[float, int, int]:
    result = function(limit)
    count = int(result if isinstance(result, int) else result.size)
    result_bytes = 0 if isinstance(result, int) else result.nbytes
    del result

    duration = median(repeat(lambda: function(limit), repeat=repeats, number=1))
    return duration, count, result_bytes


def main() -> int:
    args = _parse_args()
    functions: dict[str, Operation] = {
        "find": eratosthenes,
        "count": count_primes,
    }
    operations = (
        tuple(functions) if args.operation == "both" else (args.operation,)
    )

    runtime = (
        f"{platform.python_implementation()} {platform.python_version()} | "
        f"NumPy {np.__version__} | {platform.machine()}"
    )
    print(runtime)
    print(
        f"{'Limit':>14}  {'Operation':>9}  {'Median':>10}  "
        f"{'Primes':>11}  {'Mask MiB':>10}  {'Result MiB':>10}"
    )

    failed = False
    for limit in args.limits:
        for operation in operations:
            try:
                duration, count, result_bytes = _measure(
                    functions[operation], limit, args.repeats
                )
            except (MemoryError, ValueError) as error:
                print(f"{limit:>14,}  {operation:>9}  error: {error}")
                failed = True
                continue

            mask_bytes = 0 if limit < 2 else (limit + 1) // 2
            print(
                f"{limit:>14,}  {operation:>9}  {duration:>9.6f}s  "
                f"{count:>11,}  {mask_bytes / 1024**2:>10.2f}  "
                f"{result_bytes / 1024**2:>10.2f}"
            )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
