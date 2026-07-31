import argparse
import platform
import sys
from collections.abc import Callable
from math import isqrt
from statistics import median
from struct import calcsize
from time import perf_counter

import numpy as np

import eratosthenes as direct
import segmented

Operation = Callable[[int], int | list[int] | np.ndarray]


def _python_mask(limit: int) -> list[bool]:
    mask = [True] * ((limit + 1) // 2)

    for prime in range(3, isqrt(limit) + 1, 2):
        if mask[prime // 2]:
            for index in range(prime * prime // 2, len(mask), prime):
                mask[index] = False

    return mask


def _python_eratosthenes(limit: int) -> list[int]:
    limit = direct._validate_limit(limit)
    if limit < 2:
        return []

    mask = _python_mask(limit)
    return [2] + [2 * index + 1 for index in range(1, len(mask)) if mask[index]]


def _python_count_primes(limit: int) -> int:
    limit = direct._validate_limit(limit)
    if limit < 2:
        return 0
    return sum(_python_mask(limit))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the NumPy Sieve of Eratosthenes."
    )
    parser.add_argument(
        "limits",
        nargs="*",
        type=int,
        default=[1_000_000, 10_000_000],
        help="inclusive limits to benchmark",
    )
    parser.add_argument(
        "--operation",
        choices=("find", "count", "both"),
        default="both",
    )
    parser.add_argument(
        "--algorithm",
        choices=("python", "direct", "segmented", "numpy", "all"),
        default="numpy",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="executions per case (default: 5)",
    )
    args = parser.parse_args()

    if any(limit < 0 for limit in args.limits):
        parser.error("limits must be greater than or equal to 0")
    if args.repeats < 1:
        parser.error("--repeats must be greater than 0")
    return args


def _measure(function: Operation, limit: int, repeats: int) -> tuple[float, int, int]:
    durations = []
    count = result_bytes = 0

    for execution in range(repeats):
        started = perf_counter()
        result = function(limit)
        durations.append(perf_counter() - started)

        if execution == 0:
            if isinstance(result, int):
                count = result
            elif isinstance(result, np.ndarray):
                count = result.size
                result_bytes = result.nbytes
            else:
                count = len(result)
                result_bytes = sys.getsizeof(result) + sum(map(sys.getsizeof, result))
        del result

    return median(durations), count, result_bytes


def main() -> int:
    args = _parse_args()
    implementations: dict[str, dict[str, Operation]] = {
        "python": {
            "find": _python_eratosthenes,
            "count": _python_count_primes,
        },
        "direct": {
            "find": direct.eratosthenes,
            "count": direct.count_primes,
        },
        "segmented": {
            "find": segmented.eratosthenes,
            "count": segmented.count_primes,
        },
    }
    operation_names = (
        ("find", "count") if args.operation == "both" else (args.operation,)
    )
    if args.algorithm == "all":
        algorithm_names = tuple(implementations)
    elif args.algorithm == "numpy":
        algorithm_names = ("direct", "segmented")
    else:
        algorithm_names = (args.algorithm,)

    runtime = (
        f"{platform.python_implementation()} {platform.python_version()} | "
        f"NumPy {np.__version__} | {platform.machine()}"
    )
    print(runtime)
    print(
        f"{'Limit':>14}  {'Algorithm':>15}  {'Operation':>9}  {'Median time':>12}  "
        f"{'Primes':>11}  {'Mask storage MiB':>16}  {'Result MiB':>10}"
    )

    failed = False
    for limit in args.limits:
        for algorithm in algorithm_names:
            label = "Python baseline" if algorithm == "python" else f"NumPy {algorithm}"
            for operation in operation_names:
                try:
                    duration, count, result_bytes = _measure(
                        implementations[algorithm][operation],
                        limit,
                        args.repeats,
                    )
                except (MemoryError, ValueError) as error:
                    print(f"{limit:>14,}  {label:>15}  {operation:>9}  error: {error}")
                    failed = True
                    continue

                if limit < 2:
                    mask_bytes = 0
                elif algorithm == "python":
                    mask_bytes = sys.getsizeof([]) + ((limit + 1) // 2) * calcsize("P")
                elif algorithm == "direct":
                    mask_bytes = (limit + 1) // 2
                else:
                    mask_bytes = min(segmented._SEGMENT_SIZE, (limit - 1) // 2)
                print(
                    f"{limit:>14,}  {label:>15}  {operation:>9}  "
                    f"{duration:>11.6g}s  {count:>11,}  "
                    f"{mask_bytes / 1024**2:>16.2f}  "
                    f"{result_bytes / 1024**2:>10.2f}"
                )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
