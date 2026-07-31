import subprocess
import sys
import unittest
from math import isqrt
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import segmented
from benchmark import _measure, _python_count_primes, _python_eratosthenes
from eratosthenes import count_primes, eratosthenes

_IMPLEMENTATIONS = (
    (eratosthenes, count_primes),
    (segmented.eratosthenes, segmented.count_primes),
)


def reference_primes(limit: int) -> list[int]:
    return [
        value
        for value in range(2, limit + 1)
        if all(value % divisor for divisor in range(2, isqrt(value) + 1))
    ]


class SieveTests(unittest.TestCase):
    def test_matches_reference_for_small_limits(self) -> None:
        for limit in range(501):
            expected = np.asarray(reference_primes(limit), dtype=np.int64)
            for find, count in _IMPLEMENTATIONS:
                with self.subTest(algorithm=find.__module__, limit=limit):
                    np.testing.assert_array_equal(find(limit), expected, strict=True)
                    self.assertEqual(count(limit), expected.size)
            self.assertEqual(_python_eratosthenes(limit), expected.tolist())
            self.assertEqual(_python_count_primes(limit), expected.size)

    def test_known_prime_count(self) -> None:
        for _, count in _IMPLEMENTATIONS:
            self.assertEqual(count(1_000_000), 78_498)

    def test_matches_direct_sieve_across_small_segments(self) -> None:
        with patch.object(segmented, "_SEGMENT_SIZE", 7):
            for limit in range(501):
                with self.subTest(limit=limit):
                    expected = eratosthenes(limit)
                    np.testing.assert_array_equal(
                        segmented.eratosthenes(limit),
                        expected,
                        strict=True,
                    )
                    self.assertEqual(segmented.count_primes(limit), expected.size)

    def test_reuses_one_segment_buffer(self) -> None:
        with patch.object(segmented, "_SEGMENT_SIZE", 7):
            segments = segmented._segments(20)
            _, first = next(segments)
            _, final = next(segments)
        self.assertEqual((first.size, final.size), (7, 2))
        self.assertTrue(np.shares_memory(first, final))

    def test_result_contract(self) -> None:
        for find, count in _IMPLEMENTATIONS:
            result = find(100)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.ndim, 1)
            self.assertEqual(result.dtype, np.dtype(np.int64))
            self.assertIsInstance(count(100), int)

    def test_accepts_numpy_integers(self) -> None:
        for find, count in _IMPLEMENTATIONS:
            for value in (np.int32(30), np.int64(30), np.uint32(30)):
                with self.subTest(function=find.__module__, value=value):
                    np.testing.assert_array_equal(
                        find(value),
                        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
                        strict=True,
                    )
                    self.assertEqual(count(value), 10)

    def test_rejects_invalid_types(self) -> None:
        for find, count in _IMPLEMENTATIONS:
            for function in (find, count):
                for value in (True, np.bool_(True), 1.5, "10", None, [10]):
                    with (
                        self.subTest(function=function.__module__, value=value),
                        self.assertRaises(TypeError),
                    ):
                        function(value)  # type: ignore[arg-type]

    def test_rejects_out_of_range_values(self) -> None:
        too_large = int(np.iinfo(np.intp).max) + 1
        for find, count in _IMPLEMENTATIONS:
            for function in (find, count):
                for value in (-1, np.int64(-1), too_large, np.uint64(2**64 - 1)):
                    with (
                        self.subTest(function=function.__module__, value=value),
                        self.assertRaises(ValueError),
                    ):
                        function(value)

    def test_benchmark_repeats_are_executions(self) -> None:
        operation = Mock(return_value=7)
        with patch("benchmark.perf_counter", side_effect=range(6)):
            result = _measure(operation, 7, 3)
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(result, (1, 7, 0))

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
