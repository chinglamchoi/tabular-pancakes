"""Host interface.

generate : latents (m, n) -> rows (m, d)   (the released table, numeric block)
invert   : rows (m, d) -> latent estimates (m, n)

Contract notes.
- generate must be a *fixed measurable map applied per row*; then marked
  outputs are the pushforward of the marked prior and Theorem-1-style
  undetectability transfers by post-processing. Hosts must not mix
  information across rows (row-exchangeability of detection relies on it).
- invert is the verifier's tool only; it plays no role in undetectability.
- Inversion error is whatever it is; the measurement protocol
  (pancakemark.inversion) characterizes it empirically rather than
  assuming a form. Closed-form power predictions additionally model it as
  additive Gaussian on-axis noise -- exact for NoisyEncoderHost, an
  approximation for real encoders -- and every report therefore carries
  the assumption-free empirical mu alongside the prediction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Host(ABC):
    name: str = "host"

    @property
    @abstractmethod
    def latent_dim(self) -> int: ...

    @abstractmethod
    def generate(self, z: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def invert(self, x: np.ndarray) -> np.ndarray: ...
