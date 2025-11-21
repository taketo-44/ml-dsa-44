import pytest

from sign import sign_message, sign_repair, sign_signature


def test_sign_repair_returns_key_pair():
    key_pair = sign_repair()

    assert isinstance(key_pair, tuple), "Expected sign_repair to return a key pair tuple."
    assert len(key_pair) == 2, "sign_repair should return exactly two values (private, public)."

    private_key, public_key = key_pair
    assert isinstance(private_key, str) and private_key, "Private key should be a non-empty string."
    assert isinstance(public_key, str) and public_key, "Public key should be a non-empty string."


def test_sign_signature_produces_string_signature():
    private_key = "test_private_key"
    signature = sign_signature(private_key)

    assert isinstance(signature, str), "sign_signature must return the signature as a string."
    assert signature, "sign_signature should not return an empty signature."


def test_sign_message_returns_signature():
    private_key = "test_private_key"
    message = "test message"

    signature = sign_message(private_key, message)

    assert isinstance(signature, str), "sign_message must return the signature as a string."
    assert signature, "sign_message should return a non-empty signature."
