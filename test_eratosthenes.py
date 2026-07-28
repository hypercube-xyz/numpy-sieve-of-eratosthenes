import subprocess
import sys
import unittest
from math import isqrt
from pathlib import Path

import numpy as np

from eratosthenes import count_primes, eratosthenes


def reference_primes(limit: int) -> list[int]:
    return [
        value
        for value in range(2, limit + 1)
        if all(value % divisor for divisor in range(2, isqrt(value) + 1))
    ]


class SieveTests(unittest.TestCase):
    def test_matches_reference_for_small_limits(self) -> None:
        for limit in range(501):
            with self.subTest(limit=limit):
                expected = reference_primes(limit)
                actual = eratosthenes(limit)
                np.testing.assert_array_equal(
                    actual, np.asarray(expected, dtype=np.int64), strict=True
                )
                self.assertEqual(count_primes(limit), len(expected))

    def test_known_prime_count(self) -> None:
        self.assertEqual(count_primes(1_000_000), 78_498)

    def test_result_contract(self) -> None:
        result = eratosthenes(100)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.dtype, np.dtype(np.int64))
        self.assertIsInstance(count_primes(100), int)

    def test_accepts_numpy_integers(self) -> None:
        for value in (np.int32(30), np.int64(30), np.uint32(30)):
            with self.subTest(value=value):
                np.testing.assert_array_equal(
                    eratosthenes(value),
                    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
                    strict=True,
                )
                self.assertEqual(count_primes(value), 10)

    def test_rejects_invalid_types(self) -> None:
        for function in (eratosthenes, count_primes):
            for value in (True, np.bool_(True), 1.5, "10", None, [10]):
                with self.subTest(function=function.__name__, value=value):
                    with self.assertRaises(TypeError):
                        function(value)  # type: ignore[arg-type]

    def test_rejects_out_of_range_values(self) -> None:
        too_large = int(np.iinfo(np.intp).max) + 1
        for function in (eratosthenes, count_primes):
            for value in (-1, np.int64(-1), too_large, np.uint64(2**64 - 1)):
                with self.subTest(function=function.__name__, value=value):
                    with self.assertRaises(ValueError):
                        function(value)

    def test_benchmark_fails_when_a_case_fails(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("benchmark.py")),
                str(int(np.iinfo(np.intp).max) + 1),
                "--repeats",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("error: limit is too large for this platform", completed.stdout)


if __name__ == "__main__":
    unittest.main()
