"""Modular reduction helpers ported from PQClean's reduce.c."""

from __future__ import annotations

from .params import Q, QINV


def montgomery_reduce(a: int) -> int:
    """
    Match PQClean's `PQCLEAN_MLDSA44_CLEAN_montgomery_reduce`.

    For finite field element ``a`` with ``-2^{31}Q <= a <= Q*2^{31}``,
    compute ``r ≡ a * 2^{-32} (mod Q)`` such that ``-Q < r < Q``.
    """

    t = (a * QINV) & 0xFFFFFFFF
    t = (a - t * Q) >> 32
    return int(t)


def reduce32(a: int) -> int:
    """
    Match PQClean's `PQCLEAN_MLDSA44_CLEAN_reduce32`.

    For finite field element ``a`` with ``a <= 2^{31} - 2^{22} - 1``,
    compute ``r ≡ a (mod Q)`` such that ``-6283008 <= r <= 6283008``.
    """

    t = (a + (1 << 22)) >> 23
    return int(a - t * Q)


def caddq(a: int) -> int:
    """
    Match PQClean's `PQCLEAN_MLDSA44_CLEAN_caddq`.

    Conditionally add ``Q`` when ``a`` is negative so the return value is in
    ``[0, Q)``.
    """

    return int(a + ((a >> 31) & Q))


def freeze(a: int) -> int:
    """
    Match PQClean's `PQCLEAN_MLDSA44_CLEAN_freeze`.

    Produce the standard representative ``a mod^+ Q`` by combining
    ``reduce32`` and ``caddq``.
    """

    return caddq(reduce32(a))
