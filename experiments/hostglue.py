"""Shared glue for TabSyn-backed experiments.

get_host(name, ...) returns a (host, meta) pair where host satisfies the
Host contract and meta carries what the experiments need:
  - "flow": the local TanhFlowHost (numeric array in/out) -- lets every
    TabSyn-backed script be validated end-to-end without torch/checkpoints.
  - "tabsyn": TabSynHost; generate() returns a DataFrame, so we adapt:
    numeric_view(table) -> float array of the numeric columns (for table
    attacks and C2ST), reassemble(table, X_num) -> table with numeric
    columns replaced (attacked) -- categoricals pass through untouched.
"""
from __future__ import annotations

import numpy as np


class DataFrameNumericAdapter:
    """Numeric-block view over a mixed-type DataFrame host."""

    def __init__(self, host):
        self.host = host
        self.name = host.name
        self._num_cols = None

    @property
    def latent_dim(self):
        return self.host.latent_dim

    def generate(self, z):
        return self.host.generate(z)

    def invert(self, table):
        return self.host.invert(table)

    def numeric_view(self, table) -> np.ndarray:
        num = table.select_dtypes("number")
        self._num_cols = list(num.columns)
        return num.to_numpy(dtype=float)

    def reassemble(self, table, X_num: np.ndarray):
        out = table.copy()
        out[self._num_cols] = X_num
        return out


class ArrayHostAdapter:
    """Same interface over an array host (numeric_view = identity)."""

    def __init__(self, host):
        self.host = host
        self.name = host.name

    @property
    def latent_dim(self):
        return self.host.latent_dim

    def generate(self, z):
        return self.host.generate(z)

    def invert(self, table):
        return self.host.invert(table)

    def numeric_view(self, table) -> np.ndarray:
        return np.asarray(table, dtype=float)

    def reassemble(self, table, X_num: np.ndarray):
        return X_num


def get_host(name: str, n: int = 64, tabwak_root: str = "TabWak",
             dataname: str = "adult", device: str = "cuda:0", steps: int = 50,
             refine_iters: int = 0, fp_iters: int = 0):
    if name == "flow":
        from pancakemark.hosts import TanhFlowHost
        return ArrayHostAdapter(TanhFlowHost(n, layers=3, seed=42))
    if name == "tabsyn":
        from pancakemark.hosts.tabsyn import TabSynHost
        return DataFrameNumericAdapter(
            TabSynHost(tabwak_root, dataname, device=device, steps=steps,
                       refine_iters=refine_iters, fp_iters=fp_iters))
    raise ValueError(f"unknown host {name!r}")


def add_host_args(ap):
    ap.add_argument("--host", default="flow", choices=["flow", "tabsyn"])
    ap.add_argument("--tabwak-root", default="TabWak")
    ap.add_argument("--dataname", default="adult")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--refine-iters", type=int, default=0,
                    help="decoder-matching refinement iterations for the "
                         "tabsyn verifier (0 = plain encoder pass)")
    ap.add_argument("--fp-iters", type=int, default=0,
                    help="fixed-point iterations for exact reverse-ODE "
                         "inversion (0 = TabWak's one-shot gen_reverse)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    return ap


def effective_seeds(args) -> int:
    """--quick caps replication at 2 (fast validation runs)."""
    return min(args.seeds, 2) if args.quick else args.seeds


def run_tag(args) -> str:
    """Result-dir tag: 'flow', or 'tabsyn_<dataname>' so datasets never
    overwrite each other's CSVs."""
    return args.host if args.host != "tabsyn" else f"tabsyn_{args.dataname}"
