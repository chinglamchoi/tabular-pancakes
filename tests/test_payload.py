"""Payload channel: encode/decode round trip and end-to-end recovery."""
import numpy as np

from pancakemark import (
    keygen, sample_latents, encode_payload, decode_payload, bit_error_rate,
)
from pancakemark.detect import direction_stats

RNG = np.random.default_rng(20)


def test_gray_roundtrip_exhaustive():
    from pancakemark.payload import _gray, _gray_inverse
    for b in (1, 2, 3, 4):
        x = np.arange(1 << b)
        assert np.array_equal(_gray_inverse(_gray(x)), x)


def test_encode_decode_roundtrip_no_channel():
    k, b = 8, 3
    bits = RNG.integers(0, 2, size=k * b)
    delta = encode_payload(bits, k, b)
    out = decode_payload(delta, b)
    assert bit_error_rate(bits, out) == 0.0


def test_end_to_end_payload_through_sampler():
    k, b = 6, 2
    bits = RNG.integers(0, 2, size=k * b)
    delta = encode_payload(bits, k, b)
    key = keygen(n=64, k=k, gamma=2.0, beta=0.05, delta=delta, seed=21)
    z = sample_latents(4_000, key, RNG)
    delta_hat = direction_stats(z, key)["delta_hat"]
    out = decode_payload(delta_hat, b)
    assert bit_error_rate(bits, out) == 0.0


def test_gray_neighbor_property():
    """Adjacent constellation points differ in exactly one bit."""
    from pancakemark.payload import _gray
    for b in (2, 3, 4):
        g = _gray(np.arange(1 << b))
        for i in range(len(g)):
            diff = bin(int(g[i]) ^ int(g[(i + 1) % len(g)])).count("1")
            assert diff == 1
