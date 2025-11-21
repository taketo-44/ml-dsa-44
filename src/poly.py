"""
Polynomial helpers translated from PQClean's `crypto_sign/ml-dsa-44/clean/poly.c`.

The routines work on plain Python lists of integers where index ``i`` holds the
coefficient of ``x^i``.  The implementations mirror the reference C code as
closely as possible so that other modules (packing, signing, …) can rely on the
same bit-level behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import shake_128, shake_256
from typing import Iterable, List, Sequence, Tuple

from .params import (
    CTILDEBYTES,
    CRHBYTES,
    D,
    ETA,
    GAMMA1,
    N,
    POLYETA_PACKEDBYTES,
    POLYT0_PACKEDBYTES,
    POLYT1_PACKEDBYTES,
    POLYZ_PACKEDBYTES,
    POLYW1_PACKEDBYTES,
    Q,
    SEEDBYTES,
    SHAKE128_RATE,
    SHAKE256_RATE,
    TAU,
)
from .reduce import caddq, montgomery_reduce, reduce32
from .rounding import decompose as coeff_decompose
from .rounding import make_hint as coeff_make_hint
from .rounding import power2round as coeff_power2round
from .rounding import use_hint as coeff_use_hint

try:
    from .ntt import invntt_tomont as _invntt_tomont
    from .ntt import ntt as _ntt
except Exception:  # pragma: no cover - optional dependency
    _ntt = None
    _invntt_tomont = None


POLY_UNIFORM_NBLOCKS = (768 + SHAKE128_RATE - 1) // SHAKE128_RATE
POLY_UNIFORM_ETA_NBLOCKS = (136 + SHAKE256_RATE - 1) // SHAKE256_RATE
POLY_UNIFORM_GAMMA1_NBLOCKS = (POLYZ_PACKEDBYTES + SHAKE256_RATE - 1) // SHAKE256_RATE


@dataclass
class _ShakeStream:
    """Small helper that emulates the incremental SHAKE API used by PQClean."""

    constructor: callable
    seed_material: bytes
    pos: int = 0

    def squeeze(self, length: int) -> bytes:
        sponge = self.constructor()
        sponge.update(self.seed_material)
        data = sponge.digest(self.pos + length)
        chunk = data[-length:]
        self.pos += length
        return chunk


def _shake128_stream(seed: bytes, nonce: int) -> _ShakeStream:
    if len(seed) != SEEDBYTES:
        raise ValueError("seed must be SEEDBYTES long")
    nonce_bytes = nonce.to_bytes(2, "little")
    return _ShakeStream(shake_128, seed + nonce_bytes)


def _shake256_stream(seed: bytes, nonce: int | None = None) -> _ShakeStream:
    if nonce is None:
        seed_material = seed
    else:
        seed_material = seed + nonce.to_bytes(2, "little")
    return _ShakeStream(shake_256, seed_material)


def poly_reduce(a: List[int]) -> List[int]:
    """In-place reduction to representatives in ``[-6283008, 6283008]``."""

    for i, coeff in enumerate(a):
        a[i] = reduce32(coeff)
    return a


def poly_caddq(a: List[int]) -> List[int]:
    """In-place conditional addition of ``Q`` when coefficients are negative."""

    for i, coeff in enumerate(a):
        a[i] = caddq(coeff)
    return a


def poly_add(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """Add two polynomials without reducing the result modulo ``Q``."""

    if len(a) != len(b):
        raise ValueError("polynomials must have the same length")
    return [x + y for x, y in zip(a, b)]


def poly_sub(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """Subtract polynomials without reducing the result modulo ``Q``."""

    if len(a) != len(b):
        raise ValueError("polynomials must have the same length")
    return [x - y for x, y in zip(a, b)]


def poly_shiftl(a: List[int]) -> List[int]:
    """Multiply polynomial ``a`` by ``2^D`` in place."""

    shift = 1 << D
    for i, coeff in enumerate(a):
        a[i] = coeff * shift
    return a


def poly_ntt(a: List[int]) -> List[int]:
    """Forward NTT as defined in PQClean (delegates to :mod:`src.ntt`)."""

    if _ntt is None:
        raise NotImplementedError("NTT backend is not available")
    _ntt(a)
    return a


def poly_invntt_tomont(a: List[int]) -> List[int]:
    """Inverse NTT followed by multiplication with ``2^{32}``."""

    if _invntt_tomont is None:
        raise NotImplementedError("invNTT backend is not available")
    _invntt_tomont(a)
    return a


def poly_pointwise_montgomery(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """Pointwise multiply polys already in the NTT domain."""

    if len(a) != len(b):
        raise ValueError("polynomials must have the same length")
    return [montgomery_reduce(int(x) * int(y)) for x, y in zip(a, b)]


def poly_power2round(a: Sequence[int]) -> Tuple[List[int], List[int]]:
    """
    Split coefficients ``c`` into ``c0`` and ``c1`` such that
    ``c mod Q = c1 * 2^D + c0`` with ``c0`` centered.
    """

    c0, c1 = [], []
    for coeff in a:
        low, high = coeff_power2round(coeff)
        c0.append(low)
        c1.append(high)
    return c0, c1


def poly_decompose(a: Sequence[int]) -> Tuple[List[int], List[int]]:
    """
    Split coefficients into high and low bits with respect to ``ALPHA`` just as
    `PQCLEAN_MLDSA44_CLEAN_poly_decompose` does.
    """

    c0, c1 = [], []
    for coeff in a:
        low, high = coeff_decompose(coeff)
        c0.append(low)
        c1.append(high)
    return c0, c1


def poly_make_hint(a0: Sequence[int], a1: Sequence[int]) -> Tuple[List[int], int]:
    """
    Compute the hint polynomial indicating overflow of the low bits.

    Returns both the hint coefficients and their Hamming weight to match the C
    reference routine which returns the weight in addition to populating a
    polynomial.
    """

    if len(a0) != len(a1):
        raise ValueError("a0 and a1 must have the same length")

    hints: List[int] = []
    count = 0
    for low, high in zip(a0, a1):
        hint = coeff_make_hint(low, high)
        hints.append(hint)
        count += hint
    return hints, count


def poly_use_hint(a: Sequence[int], h: Sequence[int]) -> List[int]:
    """Correct the high bits of ``a`` using hint polynomial ``h``."""

    if len(a) != len(h):
        raise ValueError("polynomial and hint must have the same length")
    return [coeff_use_hint(coeff, hint) for coeff, hint in zip(a, h)]


def poly_chknorm(a: Sequence[int], bound: int) -> bool:
    """
    Return ``True`` when the infinity norm of ``a`` is strictly less than
    ``bound`` (which must satisfy ``bound <= (Q - 1) / 8``), ``False`` otherwise.
    """

    if bound > (Q - 1) // 8:
        return False

    for coeff in a:
        t = coeff >> 31
        t = coeff - (t & 2 * coeff)
        if t >= bound:
            return False
    return True


def rej_uniform(buf: bytes, length: int) -> List[int]:
    """
    Exposed helper mirroring PQClean's static `rej_uniform`.

    The returned list contains up to ``length`` coefficients sampled uniformly
    from ``[0, Q)`` by consuming groups of three bytes from ``buf``.
    """

    coeffs = [0] * length
    filled = _rej_uniform_into(coeffs, 0, buf, len(buf), length)
    return coeffs[:filled]


def poly_uniform(seed: bytes, nonce: int) -> List[int]:
    """
    Sample a polynomial with coefficients uniformly distributed in ``[0, Q)``.
    """

    stream = _shake128_stream(seed, nonce)
    buflen = POLY_UNIFORM_NBLOCKS * SHAKE128_RATE
    buf = bytearray(stream.squeeze(buflen))
    coeffs = [0] * N
    ctr = _rej_uniform_into(coeffs, 0, buf, buflen, N)

    while ctr < N:
        off = buflen % 3
        buf[:off] = buf[buflen - off:buflen]
        buf[off:off + SHAKE128_RATE] = stream.squeeze(SHAKE128_RATE)
        buflen = SHAKE128_RATE + off
        ctr += _rej_uniform_into(coeffs, ctr, buf, buflen, N - ctr)

    return coeffs


def rej_eta(buf: bytes, length: int) -> List[int]:
    """
    Exposed helper mirroring PQClean's static `rej_eta`.

    Decode coefficients uniformly distributed in ``[-ETA, ETA]`` from packed
    nibbles stored inside ``buf``.
    """

    coeffs = [0] * length
    filled = _rej_eta_into(coeffs, 0, buf, len(buf), length)
    return coeffs[:filled]


def poly_uniform_eta(seed: bytes, nonce: int) -> List[int]:
    """
    Sample a polynomial with coefficients uniformly distributed in ``[-ETA, ETA]``.
    """

    if len(seed) != CRHBYTES:
        raise ValueError("seed must be CRHBYTES long")

    stream = _shake256_stream(seed, nonce)
    buflen = POLY_UNIFORM_ETA_NBLOCKS * SHAKE256_RATE
    buf = bytearray(stream.squeeze(buflen))
    coeffs = [0] * N
    ctr = _rej_eta_into(coeffs, 0, buf, buflen, N)

    while ctr < N:
        buf[:] = stream.squeeze(SHAKE256_RATE)
        ctr += _rej_eta_into(coeffs, ctr, buf, SHAKE256_RATE, N - ctr)

    return coeffs


def poly_uniform_gamma1(seed: bytes, nonce: int) -> List[int]:
    """
    Sample coefficients in ``[-(GAMMA1 - 1), GAMMA1]`` by unpacking shake output.
    """

    if len(seed) != CRHBYTES:
        raise ValueError("seed must be CRHBYTES long")

    stream = _shake256_stream(seed, nonce)
    buf = stream.squeeze(POLY_UNIFORM_GAMMA1_NBLOCKS * SHAKE256_RATE)
    return polyz_unpack(buf[:POLYZ_PACKEDBYTES])


def poly_challenge(seed: bytes) -> List[int]:
    """
    Implementation of the ``challenge`` hash used by Dilithium/ML-DSA.

    The ``seed`` must be ``CTILDEBYTES`` long and is assumed to be the output of
    the hashing procedure defined in the specification.
    """

    if len(seed) != CTILDEBYTES:
        raise ValueError("seed must be CTILDEBYTES long")

    stream = _shake256_stream(seed)
    buf = bytearray(stream.squeeze(SHAKE256_RATE))
    signs = 0
    for i in range(8):
        signs |= buf[i] << (8 * i)
    pos = 8

    coeffs = [0] * N
    for i in range(N - TAU, N):
        while True:
            if pos >= SHAKE256_RATE:
                buf = bytearray(stream.squeeze(SHAKE256_RATE))
                pos = 0
            b = buf[pos]
            pos += 1
            if b <= i:
                break
        coeffs[i] = coeffs[b]
        coeffs[b] = 1 - 2 * (signs & 1)
        signs >>= 1

    return coeffs


def polyeta_pack(a: Sequence[int]) -> bytes:
    """Bit-pack coefficients from ``[-ETA, ETA]``."""

    out = bytearray(POLYETA_PACKEDBYTES)
    for i in range(N // 8):
        t = [ETA - a[8 * i + j] for j in range(8)]

        out[3 * i + 0] = ((t[0]) | (t[1] << 3) | (t[2] << 6)) & 0xFF
        out[3 * i + 1] = ((t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | (t[5] << 7)) & 0xFF
        out[3 * i + 2] = ((t[5] >> 1) | (t[6] << 2) | (t[7] << 5)) & 0xFF

    return bytes(out)


def polyeta_unpack(data: bytes) -> List[int]:
    """Inverse of :func:`polyeta_pack`."""

    if len(data) != POLYETA_PACKEDBYTES:
        raise ValueError("packed polyeta must be POLYETA_PACKEDBYTES long")

    coeffs = [0] * N
    for i in range(N // 8):
        coeffs[8 * i + 0] = ETA - ((data[3 * i + 0] >> 0) & 7)
        coeffs[8 * i + 1] = ETA - ((data[3 * i + 0] >> 3) & 7)
        coeffs[8 * i + 2] = ETA - (((data[3 * i + 0] >> 6) | (data[3 * i + 1] << 2)) & 7)
        coeffs[8 * i + 3] = ETA - ((data[3 * i + 1] >> 1) & 7)
        coeffs[8 * i + 4] = ETA - ((data[3 * i + 1] >> 4) & 7)
        coeffs[8 * i + 5] = ETA - (((data[3 * i + 1] >> 7) | (data[3 * i + 2] << 1)) & 7)
        coeffs[8 * i + 6] = ETA - ((data[3 * i + 2] >> 2) & 7)
        coeffs[8 * i + 7] = ETA - ((data[3 * i + 2] >> 5) & 7)

    return coeffs


def polyt1_pack(a: Sequence[int]) -> bytes:
    """Bit-pack ``t1`` coefficients (10-bit each)."""

    out = bytearray(POLYT1_PACKEDBYTES)
    for i in range(N // 4):
        out[5 * i + 0] = ((a[4 * i + 0] >> 0)) & 0xFF
        out[5 * i + 1] = ((a[4 * i + 0] >> 8) | (a[4 * i + 1] << 2)) & 0xFF
        out[5 * i + 2] = ((a[4 * i + 1] >> 6) | (a[4 * i + 2] << 4)) & 0xFF
        out[5 * i + 3] = ((a[4 * i + 2] >> 4) | (a[4 * i + 3] << 6)) & 0xFF
        out[5 * i + 4] = ((a[4 * i + 3] >> 2)) & 0xFF

    return bytes(out)


def polyt1_unpack(data: bytes) -> List[int]:
    """Inverse of :func:`polyt1_pack`."""

    if len(data) != POLYT1_PACKEDBYTES:
        raise ValueError("packed t1 must be POLYT1_PACKEDBYTES long")

    coeffs = [0] * N
    for i in range(N // 4):
        coeffs[4 * i + 0] = ((data[5 * i + 0] >> 0) | (data[5 * i + 1] << 8)) & 0x3FF
        coeffs[4 * i + 1] = ((data[5 * i + 1] >> 2) | (data[5 * i + 2] << 6)) & 0x3FF
        coeffs[4 * i + 2] = ((data[5 * i + 2] >> 4) | (data[5 * i + 3] << 4)) & 0x3FF
        coeffs[4 * i + 3] = ((data[5 * i + 3] >> 6) | (data[5 * i + 4] << 2)) & 0x3FF

    return coeffs


def polyt0_pack(a: Sequence[int]) -> bytes:
    """Bit-pack ``t0`` coefficients (signed ``D``-bit values)."""

    out = bytearray(POLYT0_PACKEDBYTES)
    for i in range(N // 8):
        t = [(1 << (D - 1)) - a[8 * i + j] for j in range(8)]

        out[13 * i + 0] = t[0] & 0xFF
        out[13 * i + 1] = ((t[0] >> 8) | (t[1] << 5)) & 0xFF
        out[13 * i + 2] = (t[1] >> 3) & 0xFF
        out[13 * i + 3] = ((t[1] >> 11) | (t[2] << 2)) & 0xFF
        out[13 * i + 4] = ((t[2] >> 6) | (t[3] << 7)) & 0xFF
        out[13 * i + 5] = (t[3] >> 1) & 0xFF
        out[13 * i + 6] = ((t[3] >> 9) | (t[4] << 4)) & 0xFF
        out[13 * i + 7] = (t[4] >> 4) & 0xFF
        out[13 * i + 8] = ((t[4] >> 12) | (t[5] << 1)) & 0xFF
        out[13 * i + 9] = ((t[5] >> 7) | (t[6] << 6)) & 0xFF
        out[13 * i + 10] = (t[6] >> 2) & 0xFF
        out[13 * i + 11] = ((t[6] >> 10) | (t[7] << 3)) & 0xFF
        out[13 * i + 12] = (t[7] >> 5) & 0xFF

    return bytes(out)


def polyt0_unpack(data: bytes) -> List[int]:
    """Inverse of :func:`polyt0_pack`."""

    if len(data) != POLYT0_PACKEDBYTES:
        raise ValueError("packed t0 must be POLYT0_PACKEDBYTES long")

    coeffs = [0] * N
    for i in range(N // 8):
        coeffs[8 * i + 0] = (data[13 * i + 0] | (data[13 * i + 1] << 8)) & 0x1FFF
        coeffs[8 * i + 1] = ((data[13 * i + 1] >> 5) | (data[13 * i + 2] << 3) | (data[13 * i + 3] << 11)) & 0x1FFF
        coeffs[8 * i + 2] = ((data[13 * i + 3] >> 2) | (data[13 * i + 4] << 6)) & 0x1FFF
        coeffs[8 * i + 3] = ((data[13 * i + 4] >> 7) | (data[13 * i + 5] << 1) | (data[13 * i + 6] << 9)) & 0x1FFF
        coeffs[8 * i + 4] = ((data[13 * i + 6] >> 4) | (data[13 * i + 7] << 4) | (data[13 * i + 8] << 12)) & 0x1FFF
        coeffs[8 * i + 5] = ((data[13 * i + 8] >> 1) | (data[13 * i + 9] << 7)) & 0x1FFF
        coeffs[8 * i + 6] = ((data[13 * i + 9] >> 6) | (data[13 * i + 10] << 2) | (data[13 * i + 11] << 10)) & 0x1FFF
        coeffs[8 * i + 7] = ((data[13 * i + 11] >> 3) | (data[13 * i + 12] << 5)) & 0x1FFF

        for j in range(8):
            coeffs[8 * i + j] = (1 << (D - 1)) - coeffs[8 * i + j]

    return coeffs


def polyz_pack(a: Sequence[int]) -> bytes:
    """Bit-pack coefficients with range ``[-(GAMMA1 - 1), GAMMA1]``."""

    out = bytearray(POLYZ_PACKEDBYTES)
    for i in range(N // 4):
        t0 = GAMMA1 - a[4 * i + 0]
        t1 = GAMMA1 - a[4 * i + 1]
        t2 = GAMMA1 - a[4 * i + 2]
        t3 = GAMMA1 - a[4 * i + 3]

        out[9 * i + 0] = t0 & 0xFF
        out[9 * i + 1] = (t0 >> 8) & 0xFF
        out[9 * i + 2] = ((t0 >> 16) | (t1 << 2)) & 0xFF
        out[9 * i + 3] = (t1 >> 6) & 0xFF
        out[9 * i + 4] = ((t1 >> 14) | (t2 << 4)) & 0xFF
        out[9 * i + 5] = (t2 >> 4) & 0xFF
        out[9 * i + 6] = ((t2 >> 12) | (t3 << 6)) & 0xFF
        out[9 * i + 7] = (t3 >> 2) & 0xFF
        out[9 * i + 8] = (t3 >> 10) & 0xFF

    return bytes(out)


def polyz_unpack(data: bytes) -> List[int]:
    """Inverse of :func:`polyz_pack`."""

    if len(data) != POLYZ_PACKEDBYTES:
        raise ValueError("packed z must be POLYZ_PACKEDBYTES long")

    coeffs = [0] * N
    for i in range(N // 4):
        coeffs[4 * i + 0] = GAMMA1 - ((data[9 * i + 0] | (data[9 * i + 1] << 8) | (data[9 * i + 2] << 16)) & 0x3FFFF)
        coeffs[4 * i + 1] = GAMMA1 - (((data[9 * i + 2] >> 2) | (data[9 * i + 3] << 6) | (data[9 * i + 4] << 14)) & 0x3FFFF)
        coeffs[4 * i + 2] = GAMMA1 - (((data[9 * i + 4] >> 4) | (data[9 * i + 5] << 4) | (data[9 * i + 6] << 12)) & 0x3FFFF)
        coeffs[4 * i + 3] = GAMMA1 - (((data[9 * i + 6] >> 6) | (data[9 * i + 7] << 2) | (data[9 * i + 8] << 10)) & 0x3FFFF)

    return coeffs


def polyw1_pack(a: Sequence[int]) -> bytes:
    """Bit-pack ``w1`` coefficients with representatives in ``[0, 43]``."""

    out = bytearray(POLYW1_PACKEDBYTES)
    for i in range(N // 4):
        out[3 * i + 0] = ((a[4 * i + 0]) | (a[4 * i + 1] << 6)) & 0xFF
        out[3 * i + 1] = (((a[4 * i + 1] >> 2)) | (a[4 * i + 2] << 4)) & 0xFF
        out[3 * i + 2] = (((a[4 * i + 2] >> 4)) | (a[4 * i + 3] << 2)) & 0xFF

    return bytes(out)


def _rej_uniform_into(
    out: List[int],
    start: int,
    buf: Sequence[int],
    buflen: int,
    want: int,
) -> int:
    ctr = 0
    pos = 0
    while ctr < want and pos + 3 <= buflen:
        t = buf[pos] | (buf[pos + 1] << 8) | (buf[pos + 2] << 16)
        pos += 3
        t &= 0x7FFFFF
        if t < Q:
            out[start + ctr] = t
            ctr += 1
    return ctr


def _rej_eta_into(
    out: List[int],
    start: int,
    buf: Sequence[int],
    buflen: int,
    want: int,
) -> int:
    ctr = 0
    pos = 0
    while ctr < want and pos < buflen:
        t0 = buf[pos] & 0x0F
        t1 = buf[pos] >> 4
        pos += 1

        if t0 < 15:
            t0 = t0 - (205 * t0 >> 10) * 5
            out[start + ctr] = 2 - t0
            ctr += 1
            if ctr == want:
                break
        if t1 < 15 and ctr < want:
            t1 = t1 - (205 * t1 >> 10) * 5
            out[start + ctr] = 2 - t1
            ctr += 1
    return ctr
