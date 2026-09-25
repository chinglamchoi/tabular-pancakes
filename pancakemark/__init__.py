"""Tabular Pancakes: hCLWE ("Gaussian pancake") watermarking for tabular generators.

P0 scope: exact pancake sampling, Fourier detection, certified nulls,
phase payloads, and closed-form theory -- all model-free (latent space only).

Mathematical conventions
------------------------
We use the *standard-normal* convention throughout: the ambient latent is
z ~ N(0, I_n), and the marked marginal along a secret unit direction w has
(unnormalized) density

    p(t)  ∝  exp(-t^2/2) * sum_{m in Z} exp(-(m + delta - gamma*t)^2 / (2 beta^2)),

i.e. a comb of layers under the Gaussian envelope, with phase delta in [0,1).
BRST21 / CLUE-Mark use the rho_s(x) = exp(-pi |x/s|^2) convention; the two
families coincide up to the rescaling (gamma, beta) -> (gamma, beta)/sqrt(2*pi).
All hardness statements are convention-independent (they hold for the family).

Exact mixture form (derived by completing the square; see sampler.py):
with gamma' = sqrt(gamma^2 + beta^2),

    M ~ P(M=m) ∝ exp(-(m+delta)^2 / (2 gamma'^2))          (discrete Gaussian)
    t | M=m ~ N( (m+delta) * gamma / gamma'^2 ,  beta^2/gamma'^2 ).

Detection frequency: f = gamma'^2 / gamma, so that
    f * t = (m + delta) + (beta * gamma'/gamma) * eps,  eps ~ N(0,1),
giving phase jitter std beta*gamma'/gamma (in cycle units) exactly.
"""

from .keys import Key, keygen
from .sampler import sample_pancake_1d, sample_latents, pancake_density_1d
from .detect import phase_angles, row_scores, direction_stats, global_stat, detect
from .null import hoeffding_pvalue, mc_key_null_pvalue, null_bias_estimate
from .payload import encode_payload, decode_payload, bit_error_rate
from . import theory

__version__ = "0.1.0"
