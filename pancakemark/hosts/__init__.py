"""Generator hosts: the interface between pancake latents and released tables.

A Host wraps a generator G: R^n -> X with (approximate) inverse E ~= G^{-1}.
P1 provides analytically controlled hosts (exact linear/nonlinear flows,
noisy encoders, quantized outputs) so the sigma_inv measurement protocol and
end-to-end detection are validated against known ground truth *before* any
learned model enters. TabSyn/TabDDPM integration points live in tabsyn.py
(cluster-side; requires torch + trained checkpoints).
"""
from .base import Host
from .synthetic import (
    LinearHost,
    TanhFlowHost,
    NoisyEncoderHost,
    QuantizedHost,
)

__all__ = [
    "Host", "LinearHost", "TanhFlowHost", "NoisyEncoderHost", "QuantizedHost",
]
