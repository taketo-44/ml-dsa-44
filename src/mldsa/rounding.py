"""Rounding helpers translated from PQClean's rounding.c."""

from __future__ import annotations

from .params import D, GAMMA2, Q


def power2round(a: int) -> tuple[int, int]:
    """
    Port of `PQCLEAN_MLDSA44_CLEAN_power2round`.

    For a standard representative ``a`` compute ``a0`` and ``a1`` such that
    ``a mod^+ Q = a1 * 2^D + a0`` with ``-2^{D-1} < a0 <= 2^{D-1}``.
    """

    a1 = (a + (1 << (D - 1)) - 1) >> D
    a0 = a - (a1 << D)
    return int(a0), int(a1)


def decompose(a: int) -> tuple[int, int]:
    """
    Port of `PQCLEAN_MLDSA44_CLEAN_decompose`.

    Split ``a`` into ``a0`` and ``a1`` such that
    ``a mod^+ Q = a1 * ALPHA + a0`` while keeping ``a0`` centered.
    """

    a1 = (a + 127) >> 7
    a1 = (a1 * 11275 + (1 << 23)) >> 24
    a1 ^= ((43 - a1) >> 31) & a1

    a0 = a - a1 * 2 * GAMMA2
    a0 -= (((Q - 1) // 2 - a0) >> 31) & Q
    return int(a0), int(a1)


def make_hint(a0: int, a1: int) -> int:
    """
    Port of `PQCLEAN_MLDSA44_CLEAN_make_hint`.

    Return ``1`` when the low bits overflow into the high bits, ``0`` otherwise.
    """

    if a0 > GAMMA2 or a0 < -GAMMA2 or (a0 == -GAMMA2 and a1 != 0):
        return 1
    return 0


def use_hint(a: int, hint: int) -> int:
    """
    Port of `PQCLEAN_MLDSA44_CLEAN_use_hint`.

    Correct the high bits of ``a`` according to ``hint`` and return the high part.
    """

    a0, a1 = decompose(a)
    if hint == 0:
        return a1

    if a0 > 0:
        return 0 if a1 == 43 else a1 + 1
    return 43 if a1 == 0 else a1 - 1
