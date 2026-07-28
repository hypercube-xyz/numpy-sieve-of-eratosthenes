from math import isqrt

import numpy as np

__all__ = ["count_primes", "eratosthenes"]

_MAX_LIMIT = int(np.iinfo(np.intp).max)


def _validate_limit(limit: int | np.integer) -> int:
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)):
        raise TypeError("limit must be an integer")

    limit = int(limit)
    if limit < 0:
        raise ValueError("limit must be greater than or equal to 0")
    if limit > _MAX_LIMIT:
        raise ValueError("limit is too large for this platform")
    return limit


def _prime_mask(limit: int) -> np.ndarray:
    """Return a mask for limit >= 2; slot 0 is 2 and slot i >= 1 is 2 * i + 1."""
    mask = np.ones((limit + 1) // 2, dtype=np.bool_)

    for prime in range(3, isqrt(limit) + 1, 2):
        if mask[prime // 2]:
            mask[prime * prime // 2 :: prime] = False

    return mask


def eratosthenes(limit: int | np.integer) -> np.ndarray:
    """Return every prime up to and including limit as a NumPy int64 array."""
    limit = _validate_limit(limit)
    if limit < 2:
        return np.empty(0, dtype=np.int64)

    primes = np.flatnonzero(_prime_mask(limit)).astype(np.int64, copy=False)
    primes *= 2
    primes += 1
    primes[0] = 2
    return primes


def count_primes(limit: int | np.integer) -> int:
    """Return the number of primes up to and including limit."""
    limit = _validate_limit(limit)
    if limit < 2:
        return 0
    return int(np.count_nonzero(_prime_mask(limit)))
