import pytest

from src import params
from src.packing import pack_pk, pack_sig, pack_sk, unpack_pk, unpack_sig, unpack_sk


def _eta_poly():
    return [(i % (2 * params.ETA + 1)) - params.ETA for i in range(params.N)]


def _t1_poly():
    return [i % (1 << 10) for i in range(params.N)]


def _t0_poly():
    limit = 1 << (params.D - 1)
    return [((i % (2 * limit)) - limit) for i in range(params.N)]


def _z_poly():
    span = 2 * params.GAMMA1
    return [params.GAMMA1 - (i % span) for i in range(params.N)]


def _hint_vector():
    vec = [[0] * params.N for _ in range(params.K)]
    for i, poly in enumerate(vec):
        for j in range(i + 1):
            poly[(i * 3 + j) % params.N] = 1
    return vec


def test_pack_pk_roundtrip():
    rho = bytes(range(params.SEEDBYTES))
    t1 = [_t1_poly() for _ in range(params.K)]
    packed = pack_pk(rho, t1)
    unpacked_rho, unpacked_t1 = unpack_pk(packed)

    assert unpacked_rho == rho
    assert unpacked_t1 == [[c & 0x3FF for c in poly] for poly in t1]


def test_pack_sk_roundtrip():
    rho = bytes([1] * params.SEEDBYTES)
    key = bytes([2] * params.SEEDBYTES)
    tr = bytes([3] * params.TRBYTES)
    t0 = [_t0_poly() for _ in range(params.K)]
    s1 = [_eta_poly() for _ in range(params.L)]
    s2 = [_eta_poly() for _ in range(params.K)]

    packed = pack_sk(rho, key, tr, t0, s1, s2)
    unpacked_rho, unpacked_key, unpacked_tr, unpacked_t0, unpacked_s1, unpacked_s2 = unpack_sk(packed)

    assert unpacked_rho == rho
    assert unpacked_key == key
    assert unpacked_tr == tr
    assert unpacked_t0 == t0
    assert unpacked_s1 == s1
    assert unpacked_s2 == s2


def test_pack_sig_roundtrip():
    c = bytes([4] * params.CTILDEBYTES)
    z = [_z_poly() for _ in range(params.L)]
    h = _hint_vector()

    packed = pack_sig(c, z, h)
    unpacked_c, unpacked_z, unpacked_h = unpack_sig(packed)

    assert unpacked_c == c
    assert unpacked_z == z
    assert unpacked_h == h


def test_unpack_sig_detects_invalid_ordering():
    c = bytes([5] * params.CTILDEBYTES)
    z = [_z_poly() for _ in range(params.L)]
    h = _hint_vector()
    packed = bytearray(pack_sig(c, z, h))

    # Corrupt the first hint counter so it is smaller than the number of indices
    hint_indices_offset = params.CTILDEBYTES + params.L * params.POLYZ_PACKEDBYTES
    counters_offset = hint_indices_offset + params.OMEGA
    packed[counters_offset] = 0

    with pytest.raises(ValueError):
        unpack_sig(bytes(packed))
