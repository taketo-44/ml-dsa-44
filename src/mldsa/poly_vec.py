"""
Polynomial vector helpers translated from PQClean's `polyvec.c`.

The functions operate on simple Python lists:

* A polynomial is represented as ``List[int]`` of length ``N``.
* A ``polyvecl`` is a ``List`` of ``L`` polynomials.
* A ``polyveck`` is a ``List`` of ``K`` polynomials.
* The matrix ``A`` used in ML-DSA is a ``List`` of ``K`` ``polyvecl`` rows.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .params import CRHBYTES, K, L, POLYW1_PACKEDBYTES, SEEDBYTES
from .poly import (
    poly_add,
    poly_caddq,
    poly_chknorm,
    poly_decompose,
    poly_invntt_tomont,
    poly_make_hint,
    poly_ntt,
    poly_pointwise_montgomery,
    poly_power2round,
    poly_reduce,
    poly_shiftl,
    poly_sub,
    poly_uniform,
    poly_uniform_eta,
    poly_uniform_gamma1,
    poly_use_hint,
    polyw1_pack,
)

Poly = List[int]
PolyVec = List[Poly]
Matrix = List[PolyVec]


def _clone(poly: Sequence[int]) -> Poly:
    return list(poly)


def _require_length(vec: Sequence[Sequence[int]], expected: int, name: str) -> None:
    if len(vec) != expected:
        raise ValueError(f"{name} must contain exactly {expected} polynomials")


def polyvec_matrix_expand(rho: bytes) -> List[PolyVec]:
    """
    Port of `PQCLEAN_MLDSA44_CLEAN_polyvec_matrix_expand`.

    Returns the KxL matrix A whose entries are produced via `poly_uniform`.
    """

    if len(rho) != SEEDBYTES:
        raise ValueError("rho must be SEEDBYTES bytes")

    matrix: List[PolyVec] = []
    for i in range(K):
        row: PolyVec = []
        for j in range(L):
            row.append(poly_uniform(rho, (i << 8) + j))
        matrix.append(row)
    return matrix


def polyvec_matrix_pointwise_montgomery(mat: Sequence[PolyVec], v: PolyVec) -> PolyVec:
    """
    Multiply matrix ``mat`` with vector ``v`` in the NTT domain and accumulate
    the products as defined by PQClean's `polyvec_matrix_pointwise_montgomery`.
    """

    _require_length(mat, K, "matrix rows")
    _require_length(v, L, "vector v")
    return [polyvecl_pointwise_acc_montgomery(row, v) for row in mat]


def polyvecl_uniform_eta(seed: bytes, nonce: int) -> PolyVec:
    """Sample a polyvecl with coefficients in ``[-ETA, ETA]``."""

    if len(seed) != CRHBYTES:
        raise ValueError("seed must be CRHBYTES bytes")
    vec = []
    for i in range(L):
        vec.append(poly_uniform_eta(seed, (nonce + i) & 0xFFFF))
    return vec


def polyvecl_uniform_gamma1(seed: bytes, nonce: int) -> PolyVec:
    """Sample a polyvecl with coefficients in ``[-(GAMMA1 - 1), GAMMA1]``."""

    if len(seed) != CRHBYTES:
        raise ValueError("seed must be CRHBYTES bytes")
    vec = []
    for i in range(L):
        vec.append(poly_uniform_gamma1(seed, (nonce * L + i) & 0xFFFF))
    return vec


def polyvecl_reduce(vec: PolyVec) -> PolyVec:
    """Reduce every polynomial in the vector via `poly_reduce`."""

    _require_length(vec, L, "polyvecl")
    return [poly_reduce(_clone(poly)) for poly in vec]


def polyvecl_add(u: PolyVec, v: PolyVec) -> PolyVec:
    """Add two length-L polynomial vectors without modular reduction."""

    _require_length(u, L, "u")
    _require_length(v, L, "v")
    return [poly_add(a, b) for a, b in zip(u, v)]


def polyvecl_ntt(vec: PolyVec) -> PolyVec:
    """Apply the in-place NTT to every polynomial in the vector."""

    _require_length(vec, L, "polyvecl")
    return [poly_ntt(_clone(poly)) for poly in vec]


def polyvecl_invntt_tomont(vec: PolyVec) -> PolyVec:
    """Apply the inverse NTT followed by multiplication with ``2^{32}``."""

    _require_length(vec, L, "polyvecl")
    return [poly_invntt_tomont(_clone(poly)) for poly in vec]


def polyvec_invntt_tomont(vec: PolyVec) -> PolyVec:
    """
    Backwards-compatible alias for `polyvecl_invntt_tomont` (the PQClean name).
    """

    return polyvecl_invntt_tomont(vec)


def polyvecl_pointwise_poly_montgomery(a: Poly, vec: PolyVec) -> PolyVec:
    """Pointwise multiply ``a`` with each entry of ``vec`` in the NTT domain."""

    _require_length(vec, L, "polyvecl")
    return [poly_pointwise_montgomery(a, poly) for poly in vec]


def polyvecl_pointwise_acc_montgomery(u: PolyVec, v: PolyVec) -> Poly:
    """
    Accumulate the pointwise product of two polyvecl instances (NTT domain).
    """

    _require_length(u, L, "u")
    _require_length(v, L, "v")
    acc = poly_pointwise_montgomery(u[0], v[0])
    for i in range(1, L):
        acc = poly_add(acc, poly_pointwise_montgomery(u[i], v[i]))
    return acc


def polyvecl_chknorm(vec: PolyVec, bound: int) -> bool:
    """Return True iff every polynomial's infinity norm is < bound."""

    _require_length(vec, L, "polyvecl")
    return all(poly_chknorm(poly, bound) for poly in vec)


def polyveck_uniform_eta(seed: bytes, nonce: int) -> PolyVec:
    """Sample a polyveck with coefficients in ``[-ETA, ETA]``."""

    if len(seed) != CRHBYTES:
        raise ValueError("seed must be CRHBYTES bytes")
    vec = []
    for i in range(K):
        vec.append(poly_uniform_eta(seed, (nonce + i) & 0xFFFF))
    return vec


def polyveck_reduce(vec: PolyVec) -> PolyVec:
    """Reduce each polynomial of a length-K vector via `poly_reduce`."""

    _require_length(vec, K, "polyveck")
    return [poly_reduce(_clone(poly)) for poly in vec]


def polyveck_caddq(vec: PolyVec) -> PolyVec:
    """Conditionally add ``Q`` to each polynomial coefficient."""

    _require_length(vec, K, "polyveck")
    return [poly_caddq(_clone(poly)) for poly in vec]


def polyveck_add(u: PolyVec, v: PolyVec) -> PolyVec:
    """Add two polyveck instances without modular reduction."""

    _require_length(u, K, "u")
    _require_length(v, K, "v")
    return [poly_add(a, b) for a, b in zip(u, v)]


def polyveck_sub(u: PolyVec, v: PolyVec) -> PolyVec:
    """Subtract the second polyveck from the first."""

    _require_length(u, K, "u")
    _require_length(v, K, "v")
    return [poly_sub(a, b) for a, b in zip(u, v)]


def polyveck_shiftl(vec: PolyVec) -> PolyVec:
    """Multiply every polynomial in the vector by ``2^D``."""

    _require_length(vec, K, "polyveck")
    return [poly_shiftl(_clone(poly)) for poly in vec]


def polyveck_ntt(vec: PolyVec) -> PolyVec:
    """Apply the forward NTT to each polynomial of the vector."""

    _require_length(vec, K, "polyveck")
    return [poly_ntt(_clone(poly)) for poly in vec]


def polyveck_invntt_tomont(vec: PolyVec) -> PolyVec:
    """Inverse NTT and scaling by ``2^{32}`` for each polynomial."""

    _require_length(vec, K, "polyveck")
    return [poly_invntt_tomont(_clone(poly)) for poly in vec]


def polyveck_pointwise_poly_montgomery(a: Poly, vec: PolyVec) -> PolyVec:
    """Multiply each polynomial of ``vec`` with ``a`` in the NTT domain."""

    _require_length(vec, K, "polyveck")
    return [poly_pointwise_montgomery(a, poly) for poly in vec]


def polyveck_pointwise_acc_montgomery(u: PolyVec, v: PolyVec) -> Poly:
    """Accumulate pointwise products of two length-K vectors (NTT domain)."""

    _require_length(u, K, "u")
    _require_length(v, K, "v")
    acc = poly_pointwise_montgomery(u[0], v[0])
    for i in range(1, K):
        acc = poly_add(acc, poly_pointwise_montgomery(u[i], v[i]))
    return acc


def polyveck_chknorm(vec: PolyVec, bound: int) -> bool:
    """Return True iff all polynomials respect the supplied norm bound."""

    _require_length(vec, K, "polyveck")
    return all(poly_chknorm(poly, bound) for poly in vec)


def polyveck_power2round(vec: PolyVec) -> Tuple[PolyVec, PolyVec]:
    """
    Split every polynomial into low/high parts as in `poly_power2round`.

    Returns ``(high_vec, low_vec)`` mirroring PQClean's argument order.
    """

    _require_length(vec, K, "polyveck")
    highs: PolyVec = []
    lows: PolyVec = []
    for poly in vec:
        low, high = poly_power2round(poly)
        highs.append(high)
        lows.append(low)
    return highs, lows


def polyveck_decompose(vec: PolyVec) -> Tuple[PolyVec, PolyVec]:
    """Apply `poly_decompose` to each polynomial in the vector."""

    _require_length(vec, K, "polyveck")
    highs: PolyVec = []
    lows: PolyVec = []
    for poly in vec:
        low, high = poly_decompose(poly)
        highs.append(high)
        lows.append(low)
    return highs, lows


def polyveck_make_hint(v0: PolyVec, v1: PolyVec) -> Tuple[PolyVec, int]:
    """
    Compute the hint vectors for ``v0``/``v1``.

    Returns the list of hint polynomials along with their total Hamming weight.
    """

    _require_length(v0, K, "v0")
    _require_length(v1, K, "v1")
    hints: PolyVec = []
    weight = 0
    for low, high in zip(v0, v1):
        hint_poly, count = poly_make_hint(low, high)
        hints.append(hint_poly)
        weight += count
    return hints, weight


def polyveck_use_hint(vec: PolyVec, hints: PolyVec) -> PolyVec:
    """Correct the high bits of every polynomial using the provided hints."""

    _require_length(vec, K, "vector")
    _require_length(hints, K, "hints")
    return [poly_use_hint(poly, hint) for poly, hint in zip(vec, hints)]


def polyveck_pack_w1(vec: PolyVec) -> bytes:
    """Pack the `w1` vector as defined by PQClean."""

    _require_length(vec, K, "polyveck")
    out = bytearray(K * POLYW1_PACKEDBYTES)
    for i, poly in enumerate(vec):
        start = i * POLYW1_PACKEDBYTES
        out[start : start + POLYW1_PACKEDBYTES] = polyw1_pack(poly)
    return bytes(out)
