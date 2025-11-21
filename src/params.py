"""
Parameter definitions for the ML-DSA-44 (Dilithium-II equivalent) parameter set.

Values are taken verbatim from PQClean's `params.h` so that modules translated
from the reference C implementation can share a single source of truth.
"""

SEEDBYTES = 32
CRHBYTES = 64
TRBYTES = 64
RNDBYTES = 32

N = 256
Q = 8380417
D = 13
ROOT_OF_UNITY = 1753

K = 4
L = 4
ETA = 2
TAU = 39
BETA = 78
GAMMA1 = 1 << 17
GAMMA2 = (Q - 1) // 88
OMEGA = 80
CTILDEBYTES = 32

POLYT1_PACKEDBYTES = 320
POLYT0_PACKEDBYTES = 416
POLYVECH_PACKEDBYTES = OMEGA + K
POLYZ_PACKEDBYTES = 576
POLYW1_PACKEDBYTES = 192
POLYETA_PACKEDBYTES = 96

# Combined byte lengths used by the serialized ML-DSA artifacts.
CRYPTO_PUBLICKEYBYTES = SEEDBYTES + K * POLYT1_PACKEDBYTES
CRYPTO_SECRETKEYBYTES = (
    2 * SEEDBYTES
    + TRBYTES
    + L * POLYETA_PACKEDBYTES
    + K * POLYETA_PACKEDBYTES
    + K * POLYT0_PACKEDBYTES
)
CRYPTO_BYTES = CTILDEBYTES + L * POLYZ_PACKEDBYTES + (OMEGA + K)

# SHAKE sponge rates used by the reference sampling routines.
SHAKE128_RATE = 168
SHAKE256_RATE = 136

# Montgomery reduction helper constants from reduce.h.
MONT = -4186625  # 2^32 % Q
QINV = 58728449  # q^(-1) mod 2^32

# Frequently used helper macros in the reference code.
ALPHA = 2 * GAMMA2
