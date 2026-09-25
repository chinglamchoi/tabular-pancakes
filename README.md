# Tabular Pancakes

hCLWE ("Gaussian pancake") watermarking for tabular generative models: a
training-free latent prior swap that is computationally indistinguishable from
Gaussian without the key, and detectable, attributable, and row-level
localizable with it. This package contains the full implementation, the
experiment drivers behind every table and figure in the paper, and the result
files they produced. No watermark keys ship with the code: every run generates
fresh keys.

## Installation

```bash
pip install -e ".[test,eval]"     # numpy/scipy/matplotlib + pytest + scikit-learn
pytest                            # model-free test suite, ~1 minute, CPU-only
```

PyTorch and the upstream TabSyn/TabWak pipeline are needed only for the
real-generator experiments (see "Real-generator experiments" below).

## Quick start

```python
import numpy as np
from pancakemark import keygen
from pancakemark.sampler import sample_latents
from pancakemark.detect import detect
from pancakemark.null import mc_key_null_pvalue

key = keygen(n=64, k=4, gamma=2.0, beta=0.05, seed=0)   # secret; per release
z_marked   = sample_latents(6000, key, marked=True)      # pancake prior
z_unmarked = sample_latents(6000, key, marked=False)     # N(0, I)

print(detect(z_marked, key))                 # large S, certified p-value
print(mc_key_null_pvalue(z_unmarked, key))   # ~uniform p on unmarked data
```

For marking released tables directly (the data-space variant, no model
access), see `pancakemark/datamark.py` (`TableCodec` + `DataSpaceMark` +
`detect_data_space`); for row-level localization of mixed tables, see
`pancakemark/localize.py`.

## Package layout

| Module | Contents |
|---|---|
| `pancakemark/sampler.py` | exact hCLWE sampling via the discrete-Gaussian mixture (paper App. C) |
| `pancakemark/detect.py` | phase-aware Fourier detector at the exact score frequency |
| `pancakemark/null.py` | certified p-values: exact MC key rerandomization + Hoeffding bound |
| `pancakemark/localize.py` | conformal row p-values, BH flagging, debiased fraction estimate |
| `pancakemark/datamark.py`, `tabular.py` | data-space variant: published codec + QIM-style marking |
| `pancakemark/categorical.py` | keyed Gumbel-max categorical channel |
| `pancakemark/payload.py`, `keys.py` | phase-coded payload; Haar frames, HMAC-SHA256 PRF, JSON keys |
| `pancakemark/inversion.py`, `hosts/` | sampler inversion and host adapters (flow, TabSyn) |
| `pancakemark/attacks.py`, `baselines.py`, `forensics.py` | attack suite; Gaussian Shading / TabWak baselines; Benford/GRIM/terminal-digit tests |
| `pancakemark/theory.py` | closed forms overlaid against empirics throughout |

## Reproducing the paper

Model-free experiments run locally on CPU (`python experiments/<name>.py`,
outputs to `results/<name>/`); the shipped `results/` folder contains the CSVs
behind every reported number.

| Paper item | Driver |
|---|---|
| Sampler/detector validation, null calibration, power, payload (Sec. 5.1, App. C) | `experiments/p0_*.py` |
| Conditioning wall: exact-inversion decomposition (Sec. 3.3) | `run_gonogo.py`, `experiments/p1b_tabsyn_grid.py`, `p1_hosts_sigma_inv.py` |
| False-accusation audit, fidelity C2ST (Sec. 5.2, App. D) | `experiments/p2_fpr_audit.py`, `p2_fidelity.py`, `p2b_tabsyn_fidelity.py` |
| Robustness / degradation law (Prop. 4.3, Sec. 5.1) | `experiments/p3_robustness.py`, `p3b_tabsyn_robustness.py` |
| Forensic-test parity (Sec. 1, Sec. 5) | `experiments/p4_forensics.py` |
| Localization (Fig. 2, Sec. 5.3) | `experiments/p5_localization.py`, `p13_datamark_localize.py` |
| Removal frontier and sqrt(n/k) law (Thm. 4.4, App. D) | `experiments/p6_removal.py`, `p6b_k_ablation.py` |
| Categorical channel | `experiments/p7_categorical.py` |
| Concrete security: covariance attack (App. F) | `experiments/p8_covariance_attack.py` |
| Payload under attack | `experiments/p9_payload_attacks.py` |
| Laundering / D3 boundary (App. E) | `experiments/p10_radioactivity_toy.py` |
| Baseline comparison (Table 1) | `experiments/p11_baselines.py` |
| Dataset matrix + dial (Table 2, App. D) | `experiments/p12_datamark.py`, `p12b_dial.py` |
| Generator matrix (Table 3) | `experiments/p14_models_matrix.py` |

## Real-generator experiments

The TabSyn-backed experiments train generators through the public
TabSyn/TabWak pipeline. Setup:

1. Clone https://github.com/chaoyitud/TabWak next to this repository and apply
   the compatibility notes in `third_party/FIXES_TABWAK.md` (dataset config
   templates are provided in `third_party/tabwak_data_Info/`), then train the
   VAE and diffusion checkpoints per dataset through that pipeline.
2. Run `run_gonogo.py` (the conditioning-wall measurement) and the
   TabSyn-backed drivers (`experiments/p1b_*.py`, `p2b_*.py`, `p3b_*.py`,
   `p11`--`p14`) with `--host tabsyn`; each accepts `--host flow` for a
   checkpoint-free local validation run.

Generator training dominates the compute (about 30 GPU-hours for a full
replication); all verification suites are CPU-light.

## Conventions

Standard-normal convention throughout: the pancake marginal is
`p(t) ∝ exp(−t²/2)·Σ_m exp(−(m+δ−γt)²/(2β²))`. Bruna et al. use
`ρ_s(x)=exp(−π|x/s|²)`; the families coincide under `(γ,β) → (γ,β)/√(2π)`,
and hardness statements are convention-independent.
