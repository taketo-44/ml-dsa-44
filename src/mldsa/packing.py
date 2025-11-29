"""
Serialization helpers translated from PQClean's `packing.c`.

All helpers work on the lightweight Python representations used throughout the
package: a polynomial is a ``List[int]`` of length ``N`` and polynomial vectors
are simple lists of polynomials with length ``L`` (polyvecl) or ``K`` (polyveck).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .params import (
    CRYPTO_BYTES,
    CRYPTO_PUBLICKEYBYTES,
    CRYPTO_SECRETKEYBYTES,
    CTILDEBYTES,
    K,
    L,
    N,
    OMEGA,
    POLYETA_PACKEDBYTES,
    POLYT0_PACKEDBYTES,
    POLYT1_PACKEDBYTES,
    POLYZ_PACKEDBYTES,
    SEEDBYTES,
    TRBYTES,
)
from .poly import (
    polyeta_pack,
    polyeta_unpack,
    polyt0_pack,
    polyt0_unpack,
    polyt1_pack,
    polyt1_unpack,
    polyz_pack,
    polyz_unpack,
)

Poly = List[int]
PolyVec = List[Poly]


def _expect_length(buf: Sequence[Poly], length: int, name: str) -> None:
    if len(buf) != length:
        raise ValueError(f"{name} must contain exactly {length} polynomials")


def pack_pk(rho: bytes, t1: Sequence[Poly]) -> bytes:
    """
    Bit-pack the public key ``pk = (rho, t1)``.
    """

    if len(rho) != SEEDBYTES:
        raise ValueError("rho must be SEEDBYTES bytes long")
    _expect_length(t1, K, "t1")

    out = bytearray(CRYPTO_PUBLICKEYBYTES)
    out[:SEEDBYTES] = rho

    offset = SEEDBYTES
    for i, poly in enumerate(t1):
        start = offset + i * POLYT1_PACKEDBYTES
        out[start : start + POLYT1_PACKEDBYTES] = polyt1_pack(poly)

    return bytes(out)


def unpack_pk(pk: bytes) -> Tuple[bytes, PolyVec]:
    """
    Inverse of :func:`pack_pk`. Returns ``(rho, t1)``.
    """

    if len(pk) != CRYPTO_PUBLICKEYBYTES:
        raise ValueError("packed public key has invalid length")

    rho = bytes(pk[:SEEDBYTES])
    t1: PolyVec = []
    offset = SEEDBYTES
    for i in range(K):
        start = offset + i * POLYT1_PACKEDBYTES
        t1.append(polyt1_unpack(pk[start : start + POLYT1_PACKEDBYTES]))

    return rho, t1


def pack_sk(
    rho: bytes,
    key: bytes,
    tr: bytes,
    t0: Sequence[Poly],
    s1: Sequence[Poly],
    s2: Sequence[Poly],
) -> bytes:
    """
    Bit-pack the secret key ``sk = (rho, key, tr, s1, s2, t0)`` as done by PQClean.
    """

    if len(rho) != SEEDBYTES or len(key) != SEEDBYTES:
        raise ValueError("rho and key must be SEEDBYTES bytes long")
    if len(tr) != TRBYTES:
        raise ValueError("tr must be TRBYTES bytes long")
    _expect_length(s1, L, "s1")
    _expect_length(s2, K, "s2")
    _expect_length(t0, K, "t0")

    out = bytearray(CRYPTO_SECRETKEYBYTES)
    offset = 0

    out[offset : offset + SEEDBYTES] = rho
    offset += SEEDBYTES

    out[offset : offset + SEEDBYTES] = key
    offset += SEEDBYTES

    out[offset : offset + TRBYTES] = tr
    offset += TRBYTES

    for poly in s1:
        out[offset : offset + POLYETA_PACKEDBYTES] = polyeta_pack(poly)
        offset += POLYETA_PACKEDBYTES

    for poly in s2:
        out[offset : offset + POLYETA_PACKEDBYTES] = polyeta_pack(poly)
        offset += POLYETA_PACKEDBYTES

    for poly in t0:
        out[offset : offset + POLYT0_PACKEDBYTES] = polyt0_pack(poly)
        offset += POLYT0_PACKEDBYTES

    return bytes(out)


def unpack_sk(sk: bytes) -> Tuple[bytes, bytes, bytes, PolyVec, PolyVec, PolyVec]:
    """
    Inverse of :func:`pack_sk`. Returns ``(rho, key, tr, t0, s1, s2)``.
    """

    if len(sk) != CRYPTO_SECRETKEYBYTES:
        raise ValueError("packed secret key has invalid length")

    offset = 0
    rho = bytes(sk[offset : offset + SEEDBYTES])
    offset += SEEDBYTES

    key = bytes(sk[offset : offset + SEEDBYTES])
    offset += SEEDBYTES

    tr = bytes(sk[offset : offset + TRBYTES])
    offset += TRBYTES

    s1: PolyVec = []
    for _ in range(L):
        s1.append(polyeta_unpack(sk[offset : offset + POLYETA_PACKEDBYTES]))
        offset += POLYETA_PACKEDBYTES

    s2: PolyVec = []
    for _ in range(K):
        s2.append(polyeta_unpack(sk[offset : offset + POLYETA_PACKEDBYTES]))
        offset += POLYETA_PACKEDBYTES

    t0: PolyVec = []
    for _ in range(K):
        t0.append(polyt0_unpack(sk[offset : offset + POLYT0_PACKEDBYTES]))
        offset += POLYT0_PACKEDBYTES

    return rho, key, tr, t0, s1, s2


def pack_sig(c: bytes, z: Sequence[Poly], h: Sequence[Poly]) -> bytes:
    """
    Bit-pack the signature ``sig = (c, z, h)``.
    """

    if len(c) != CTILDEBYTES:
        raise ValueError("challenge hash must be CTILDEBYTES bytes")
    _expect_length(z, L, "z")
    _expect_length(h, K, "h")

    out = bytearray(CRYPTO_BYTES)
    out[:CTILDEBYTES] = c

    offset = CTILDEBYTES
    for poly in z:
        out[offset : offset + POLYZ_PACKEDBYTES] = polyz_pack(poly)
        offset += POLYZ_PACKEDBYTES

    hint_indices_start = offset
    hint_counters_start = hint_indices_start + OMEGA

    # Zero-initialize the hint section
    out[hint_indices_start : hint_counters_start + K] = b"\x00" * (OMEGA + K)

    k = 0
    for i, poly in enumerate(h):
        for idx, coeff in enumerate(poly):
            if coeff != 0:
                if idx >= N:
                    raise ValueError("hint coefficient index exceeds polynomial degree")
                if k >= OMEGA:
                    raise ValueError("hint weight exceeds OMEGA")
                out[hint_indices_start + k] = idx & 0xFF
                k += 1
        out[hint_counters_start + i] = k

    return bytes(out)


def unpack_sig(sig: bytes) -> Tuple[bytes, PolyVec, PolyVec]:
    """
    Inverse of :func:`pack_sig`.

    Raises ``ValueError`` if the encoding is malformed (matching PQClean's
    return value semantics where ``1`` indicated failure).
    """

    if len(sig) != CRYPTO_BYTES:
        raise ValueError("packed signature has invalid length")

    offset = 0
    c = bytes(sig[offset : offset + CTILDEBYTES])
    offset += CTILDEBYTES

    z: PolyVec = []
    for _ in range(L):
        z.append(polyz_unpack(sig[offset : offset + POLYZ_PACKEDBYTES]))
        offset += POLYZ_PACKEDBYTES

    indices = sig[offset : offset + OMEGA]
    counters = sig[offset + OMEGA : offset + OMEGA + K]

    h: PolyVec = [[0] * N for _ in range(K)]
    k = 0
    for i in range(K):
        bound = counters[i]
        if bound < k or bound > OMEGA:
            raise ValueError("hint counters are out of range")

        for j in range(k, bound):
            idx = indices[j]
            if idx >= N:
                raise ValueError("hint index exceeds polynomial degree")
            if j > k and idx <= indices[j - 1]:
                raise ValueError("hint indices must be strictly increasing")
            h[i][idx] = 1

        k = bound

    if any(indices[j] for j in range(k, OMEGA)):
        raise ValueError("unused hint index slots must be zero")

    return c, z, h
