import pytest

from mldsa import params
from mldsa.poly import poly_add, poly_pointwise_montgomery
from mldsa.poly_vec import (
    polyvec_matrix_expand,
    polyvec_matrix_pointwise_montgomery,
    polyvecl_add,
    polyvecl_chknorm,
    polyvecl_pointwise_acc_montgomery,
    polyvecl_pointwise_poly_montgomery,
    polyvecl_reduce,
    polyvecl_uniform_eta,
    polyvecl_uniform_gamma1,
    polyveck_add,
    polyveck_caddq,
    polyveck_chknorm,
    polyveck_decompose,
    polyveck_make_hint,
    polyveck_pointwise_acc_montgomery,
    polyveck_pointwise_poly_montgomery,
    polyveck_power2round,
    polyveck_reduce,
    polyveck_shiftl,
    polyveck_sub,
    polyveck_uniform_eta,
    polyveck_use_hint,
    polyveck_pack_w1,
)


def _const_poly(value: int):
    return [value] * params.N


def test_polyvec_matrix_expand_deterministic():
    rho = bytes(range(params.SEEDBYTES))
    mat1 = polyvec_matrix_expand(rho)
    mat2 = polyvec_matrix_expand(rho)

    assert mat1 == mat2
    assert len(mat1) == params.K
    assert all(len(row) == params.L for row in mat1)
    assert all(len(poly) == params.N for row in mat1 for poly in row)


def test_polyvec_matrix_pointwise_montgomery_matches_manual_accumulation():
    mat = [
        [_const_poly(i + j + 1) for j in range(params.L)]
        for i in range(params.K)
    ]
    vector = [_const_poly(j + 1) for j in range(params.L)]

    result = polyvec_matrix_pointwise_montgomery(mat, vector)

    expected = []
    for row in mat:
        acc = poly_pointwise_montgomery(row[0], vector[0])
        for j in range(1, params.L):
            acc = poly_add(acc, poly_pointwise_montgomery(row[j], vector[j]))
        expected.append(acc)

    assert result == expected


def test_polyvecl_uniform_eta_range():
    seed = bytes([4] * params.CRHBYTES)
    vec = polyvecl_uniform_eta(seed, 0)
    assert len(vec) == params.L
    assert all(len(poly) == params.N for poly in vec)
    for coeff in vec[0]:
        assert -params.ETA <= coeff <= params.ETA


def test_polyvecl_uniform_gamma1_range():
    seed = bytes([5] * params.CRHBYTES)
    vec = polyvecl_uniform_gamma1(seed, 0)
    assert len(vec) == params.L
    for coeff in vec[0]:
        assert -(params.GAMMA1 - 1) <= coeff <= params.GAMMA1


def test_polyvecl_reduce_enforces_length():
    vec = [_const_poly(params.Q + 1) for _ in range(params.L)]
    reduced = polyvecl_reduce(vec)
    assert all(all(0 <= coeff < params.Q for coeff in poly) for poly in reduced)

    with pytest.raises(ValueError):
        polyvecl_reduce(vec[:-1])


def test_polyvecl_add_and_pointwise_helpers():
    u = [_const_poly(i) for i in range(params.L)]
    v = [_const_poly(i + 1) for i in range(params.L)]

    added = polyvecl_add(u, v)
    assert added[0][0] == u[0][0] + v[0][0]

    multiplied = polyvecl_pointwise_poly_montgomery(_const_poly(1), v)
    assert multiplied == [poly_pointwise_montgomery(_const_poly(1), poly) for poly in v]

    acc = polyvecl_pointwise_acc_montgomery(u, v)
    expected = poly_pointwise_montgomery(u[0], v[0])
    for i in range(1, params.L):
        expected = poly_add(expected, poly_pointwise_montgomery(u[i], v[i]))
    assert acc == expected


def test_polyvecl_and_polyveck_chknorm():
    vec_l = [_const_poly(0) for _ in range(params.L)]
    vec_k = [_const_poly(0) for _ in range(params.K)]
    assert polyvecl_chknorm(vec_l, 10)
    assert polyveck_chknorm(vec_k, 10)


def test_polyveck_uniform_eta_and_reduce_family():
    seed = bytes([6] * params.CRHBYTES)
    vec = polyveck_uniform_eta(seed, 0)
    assert len(vec) == params.K
    assert all(len(poly) == params.N for poly in vec)

    reduced = polyveck_reduce(vec)
    cadded = polyveck_caddq([[-1] * params.N for _ in range(params.K)])
    assert all(poly[0] == params.Q - 1 for poly in cadded)

    shifted = polyveck_shiftl([_const_poly(1) for _ in range(params.K)])
    assert shifted[0][0] == (1 << params.D)


def test_polyveck_add_sub_pointwise():
    u = [_const_poly(i + 1) for i in range(params.K)]
    v = [_const_poly(1) for _ in range(params.K)]

    assert polyveck_add(u, v)[0][0] == u[0][0] + 1
    assert polyveck_sub(u, v)[0][0] == u[0][0] - 1

    multiplied = polyveck_pointwise_poly_montgomery(_const_poly(2), v)
    assert multiplied == [poly_pointwise_montgomery(_const_poly(2), poly) for poly in v]

    acc = polyveck_pointwise_acc_montgomery(u, v)
    expected = poly_pointwise_montgomery(u[0], v[0])
    for i in range(1, params.K):
        expected = poly_add(expected, poly_pointwise_montgomery(u[i], v[i]))
    assert acc == expected


def test_polyveck_power2round_and_decompose_roundtrip():
    vec = [list(range(params.N)) for _ in range(params.K)]

    highs, lows = polyveck_power2round(vec)
    for poly, high, low in zip(vec, highs, lows):
        reconstructed = [(h << params.D) + l for h, l in zip(high, low)]
        assert reconstructed == poly

    high_dec, low_dec = polyveck_decompose(vec)
    for original, high, low in zip(vec, high_dec, low_dec):
        assert all((h * 2 * params.GAMMA2 + l - o) % params.Q == 0 for h, l, o in zip(high, low, original))


def test_polyveck_make_and_use_hint():
    v0 = [list(range(params.N)) for _ in range(params.K)]
    v1 = [list(range(params.N)) for _ in range(params.K)]
    hints, weight = polyveck_make_hint(v0, v1)
    assert weight == sum(sum(hint) for hint in hints)

    corrected = polyveck_use_hint(v1, hints)
    assert len(corrected) == params.K


def test_polyveck_pack_w1_length_and_validation():
    vec = [[(i + j) % 16 for j in range(params.N)] for i in range(params.K)]
    packed = polyveck_pack_w1(vec)
    assert len(packed) == params.K * params.POLYW1_PACKEDBYTES

    with pytest.raises(ValueError):
        polyveck_pack_w1(vec[:-1])
