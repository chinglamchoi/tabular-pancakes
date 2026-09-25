"""Multi-bit payloads over the inhomogeneous-CLWE phase channel.

b bits per direction over the constellation {0, 1, ..., 2^b - 1} / 2^b,
Gray-coded so a one-step phase error flips a single bit. Decoding is
nearest-constellation-point on the circular metric. Outer error-correcting
codes are deliberately out of scope for P0 (documented in the paper as an
orthogonal layer); with k directions the raw capacity is k*b bits.
"""
from __future__ import annotations

import numpy as np


def _gray(x: np.ndarray) -> np.ndarray:
    return x ^ (x >> 1)


def _gray_inverse(g: np.ndarray) -> np.ndarray:
    """binary = gray ^ (gray>>1) ^ (gray>>2) ^ ... (canonical prefix-XOR)."""
    x = g.copy()
    mask = g >> 1
    while mask.any():
        x = x ^ mask
        mask = mask >> 1
    return x


def encode_payload(bits: np.ndarray, k: int, bits_per_direction: int) -> np.ndarray:
    """bits (length k*b, 0/1) -> phases delta in [0,1)^k."""
    b = bits_per_direction
    bits = np.asarray(bits, dtype=int).reshape(k, b)
    symbols = np.zeros(k, dtype=int)
    for i in range(b):
        symbols = (symbols << 1) | bits[:, i]
    return _gray(symbols).astype(float) / (1 << b)


def decode_payload(delta_hat: np.ndarray, bits_per_direction: int) -> np.ndarray:
    """Estimated phases -> bits (length k*b), circular nearest neighbor."""
    b = bits_per_direction
    M = 1 << b
    gray_symbols = np.mod(np.rint(np.asarray(delta_hat) * M).astype(int), M)
    symbols = _gray_inverse(gray_symbols)
    bits = np.zeros((symbols.size, b), dtype=int)
    for i in range(b):
        bits[:, b - 1 - i] = (symbols >> i) & 1
    return bits.reshape(-1)


def bit_error_rate(bits_true: np.ndarray, bits_dec: np.ndarray) -> float:
    bits_true = np.asarray(bits_true, dtype=int)
    bits_dec = np.asarray(bits_dec, dtype=int)
    return float(np.mean(bits_true != bits_dec))
