"""Shared experiment helpers: output paths, CSV saving, plotting defaults."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(os.environ.get("PANCAKE_RESULTS", Path(__file__).parent.parent / "results"))

mpl.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
})


def outdir(name: str) -> Path:
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def save_fig(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def aggregate(rows: list[dict], group_keys: list[str],
              value_keys: list[str]) -> list[dict]:
    """Group rows and append mean/std/nseeds per value key."""
    import numpy as _np
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    out = []
    for gk, rs in groups.items():
        rec = dict(zip(group_keys, gk))
        rec["nseeds"] = len(rs)
        for v in value_keys:
            vals = _np.array([r[v] for r in rs], dtype=float)
            rec[f"{v}_mean"] = float(vals.mean())
            rec[f"{v}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(rec)
    return out
