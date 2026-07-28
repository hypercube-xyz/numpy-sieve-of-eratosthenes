from collections.abc import Iterator
from math import isqrt

import numpy as np

from eratosthenes import _validate_limit, eratosthenes as _direct_sieve

__all__ = ["count_primes", "eratosthenes"]

_SEGMENT_SIZE = 8 << 20


def _segments(limit: int) -> Iterator[tuple[int, np.ndarray]]:
    base_primes = _direct_sieve(isqrt(limit))[1:].tolist()

    for low in range(3, limit + 1, 2 * _SEGMENT_SIZE):
        size = min(_SEGMENT_SIZE, (limit - low) // 2 + 1)
        high = low + 2 * size - 2
        mask = np.ones(size, dtype=np.bool_)

        for prime in base_primes:
            square = prime * prime
            if square > high:
                break

            start = max(square, low + (-low % prime))
            if start % 2 == 0:
                start += prime
            mask[(start - low) // 2 :: prime] = False

        yield low, mask


def eratosthenes(limit: int | np.integer) -> np.ndarray:
    """Return primes up to and including limit using a segmented sieve."""
    limit = _validate_limit(limit)
    if limit < 2:
        return np.empty(0, dtype=np.int64)

    chunks = [np.array([2], dtype=np.int64)]
    for low, mask in _segments(limit):
        chunk = np.flatnonzero(mask).astype(np.int64, copy=False)
        chunk *= 2
        chunk += low
        chunks.append(chunk)
    return np.concatenate(chunks)


def count_primes(limit: int | np.integer) -> int:
    """Count primes up to and including limit using a segmented sieve."""
    limit = _validate_limit(limit)
    if limit < 2:
        return 0
    return 1 + sum(int(np.count_nonzero(mask)) for _, mask in _segments(limit))
