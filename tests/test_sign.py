import pytest

from mldsa.sign import sign_context, sign_signature, sign_keypair, sign_open, sign_verify


def test_sign_signature_produces_string_signature():
    private_key = "test_private_key"
    signature = sign_signature(private_key)

    assert isinstance(signature, str), "sign_signature must return the signature as a string."
    assert signature, "sign_signature should not return an empty signature."
