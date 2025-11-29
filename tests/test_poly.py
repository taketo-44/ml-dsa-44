import itertools

import pytest

from mldsa import params
from mldsa.poly import (
    poly_add,
    poly_caddq,
    poly_challenge,
    poly_chknorm,
    poly_decompose,
    poly_make_hint,
    poly_power2round,
    poly_reduce,
    poly_shiftl,
    poly_sub,
    poly_uniform,
    poly_uniform_eta,
    poly_uniform_gamma1,
    poly_use_hint,
    polyeta_pack,
    polyeta_unpack,
    polyz_pack,
    polyz_unpack,
    polyt0_pack,
    polyt0_unpack,
    polyt1_pack,
    polyt1_unpack,
    poly_pointwise_montgomery,
)
from mldsa.reduce import montgomery_reduce, reduce32


def _eta_poly():
    return [(i % (2 * params.ETA + 1)) - params.ETA for i in range(params.N)]


def _t1_poly():
    return [i % (1 << 10) for i in range(params.N)]


def _t0_poly():
    base = (1 << (params.D - 1)) - 1
    return [((i % (2 * base)) - base) for i in range(params.N)]


def _z_poly():
    span = 2 * params.GAMMA1
    return [params.GAMMA1 - (i % span) for i in range(params.N)]


def test_poly_reduce_matches_scalar_reduce32():
    coeffs = [params.Q + 1, -params.Q - 2, 5]
    reduced = poly_reduce(coeffs.copy())
    assert reduced == [reduce32(c) for c in coeffs]


def test_poly_caddq_normalises_negative_coefficients():
    coeffs = [-1, 0, 5]
    assert poly_caddq(coeffs.copy())[0] == params.Q - 1


def test_poly_add_length_mismatch_raises():
    with pytest.raises(ValueError):
        poly_add([1], [1, 2])


def test_poly_sub_length_mismatch_raises():
    with pytest.raises(ValueError):
        poly_sub([1, 2], [1])


def test_poly_shiftl_multiplies_by_2d():
    coeffs = [1, -2, 3]
    expected = [c * (1 << params.D) for c in coeffs]
    assert poly_shiftl(coeffs.copy()) == expected


def test_poly_pointwise_montgomery_matches_scalar_product():
    a = [1, params.Q - 1, 2, 3]
    b = [2, 3, 4, 5]
    result = poly_pointwise_montgomery(a, b)
    expected = [montgomery_reduce(x * y) for x, y in zip(a, b)]
    assert result == expected


def test_poly_power2round_reconstructs_coefficients():
    coeffs = [(i << (params.D - 1)) + i for i in range(16)]
    low, high = poly_power2round(coeffs)
    reconstructed = [(h << params.D) + l for l, h in zip(low, high)]
    assert reconstructed == coeffs


def test_poly_decompose_preserves_mod_q_value():
    coeffs = [(i * params.GAMMA2 + 7) % params.Q for i in range(16)]
    low, high = poly_decompose(coeffs)
    for original, l, h in zip(coeffs, low, high):
        assert (h * 2 * params.GAMMA2 + l - original) % params.Q == 0


def test_poly_make_and_use_hint_work_together():
    coeffs = [params.GAMMA2 + 5, -params.GAMMA2, 0, 7]
    low, high = poly_decompose(coeffs)
    hints, weight = poly_make_hint(low, high)
    assert weight == sum(hints)
    corrected = poly_use_hint(coeffs, hints)
    for hint, c_high, corr in zip(hints, high, corrected):
        if hint == 0:
            assert corr == c_high


def test_poly_chknorm_detects_large_coefficients():
    small = [0] * params.N
    assert poly_chknorm(small, 100)

    large = small.copy()
    large[0] = 1000
    assert not poly_chknorm(large, 500)


def test_poly_uniform_is_deterministic_and_bounded():
    seed = bytes(range(params.SEEDBYTES))
    first = poly_uniform(seed, 0)
    second = poly_uniform(seed, 0)
    third = poly_uniform(seed, 1)

    assert first == second
    assert first != third
    assert all(0 <= coeff < params.Q for coeff in first)


def test_poly_uniform_eta_range():
    seed = bytes([1] * params.CRHBYTES)
    poly = poly_uniform_eta(seed, 0)
    assert len(poly) == params.N
    assert all(-params.ETA <= coeff <= params.ETA for coeff in poly)


def test_poly_uniform_gamma1_range():
    seed = bytes([2] * params.CRHBYTES)
    poly = poly_uniform_gamma1(seed, 0)
    assert len(poly) == params.N
    assert all(-(params.GAMMA1 - 1) <= c <= params.GAMMA1 for c in poly)


def test_poly_challenge_sets_tau_coefficients():
    seed = bytes([3] * params.CTILDEBYTES)
    challenge = poly_challenge(seed)
    non_zero = [c for c in challenge if c != 0]
    assert len(non_zero) == params.TAU
    assert set(non_zero).issubset({1, -1})


def test_polyeta_pack_roundtrip():
    poly = _eta_poly()
    packed = polyeta_pack(poly)
    unpacked = polyeta_unpack(packed)
    assert unpacked == poly


def test_polyt1_pack_roundtrip():
    poly = _t1_poly()
    packed = polyt1_pack(poly)
    unpacked = polyt1_unpack(packed)
    assert unpacked == [c & 0x3FF for c in poly]


def test_polyt0_pack_roundtrip():
    poly = _t0_poly()
    packed = polyt0_pack(poly)
    unpacked = polyt0_unpack(packed)
    assert unpacked == poly


def test_polyz_pack_roundtrip():
    poly = _z_poly()
    packed = polyz_pack(poly)
    unpacked = polyz_unpack(packed)
    assert unpacked == poly
