import pytest

from src import params
from src.reduce import caddq, freeze, montgomery_reduce, reduce32
from src.rounding import decompose, make_hint, power2round, use_hint


def test_montgomery_reduce_identity():
    assert montgomery_reduce(1 << 32) == 1
    assert montgomery_reduce((params.Q - 1) << 32) == params.Q - 1


def test_reduce32_brings_value_into_range():
    value = params.Q * 3 + 12345
    reduced = reduce32(value)

    assert -params.Q < reduced < params.Q
    assert (value - reduced) % params.Q == 0


def test_caddq_makes_negative_coefficient_positive():
    assert caddq(-1) == params.Q - 1
    assert caddq(5) == 5


def test_freeze_returns_standard_representative():
    value = -2 * params.Q + 17
    frozen = freeze(value)

    assert 0 <= frozen < params.Q
    assert (value - frozen) % params.Q == 0


def test_power2round_reconstruction():
    sample = (1 << (params.D + 1)) + 123
    low, high = power2round(sample)

    assert -2 ** (params.D - 1) < low <= 2 ** (params.D - 1)
    assert (high << params.D) + low == sample


def test_decompose_and_use_hint_round_trip():
    sample = params.GAMMA2 * 5 + 7
    low, high = decompose(sample)
    hint = make_hint(low, high)

    assert -params.GAMMA2 <= low <= params.GAMMA2
    recovered = use_hint(sample, hint)
    if hint == 0:
        assert recovered == high
    else:
        assert recovered != high


def test_make_hint_detects_out_of_bound_low_bits():
    assert make_hint(params.GAMMA2 + 1, 0) == 1
    assert make_hint(0, 0) == 0
