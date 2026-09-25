"""The conditioning wall: exact-inversion decomposition on TabSyn--Adult.

Measures why phase-precision latent verification fails on a diffusion--VAE
release map (paper Sec. 3.3), in three parts:

  A0: per-step preimage check at steps=50 -- solve each step of the forward
      map in isolation from the true (x_t, y_t) pair, reporting forward
      residuals, preimage errors, condition numbers, and any
      non-injectivity.
  A1: sigma_ode with the damped fixed point + gradient polish at
      steps = 50 / 200 / 1000 (generation and inversion on the same grid,
      as deployed by the provider) -- the step-count-independence result.
  B:  decoder-matching refinement sweep at the best grid.

Run:  python -u run_gonogo.py 2>&1 | tee results/gonogo_adult.txt
"""
import json

import numpy as np

from pancakemark import keygen
from pancakemark.inversion import inversion_report
from pancakemark.hosts.tabsyn import TabSynHost

host = TabSynHost("TabWak", "adult", device="cuda:0", steps=50)
torch = host.torch
n = host.latent_dim
dev = host.device
noise_fn = host.model.noise_fn
sched = host.scheduler
print(f"latent_dim = {n}")


def consts(t, dt):
    acp = sched.alphas_cumprod
    prev = t - dt
    a_t = float(acp[t])
    a_p = float(acp[prev]) if prev >= 0 else float(sched.final_alpha_cumprod)
    return a_t ** 0.5, a_p ** 0.5, (1 - a_t) ** 0.5, (1 - a_p) ** 0.5


def solve_step(y, t, dt, fp_iters=60, w=0.3, grad_iters=40):
    s_t, s_p, b_t, b_p = consts(t, dt)
    coef = (s_t / s_p) * b_p - b_t
    tt = torch.full((y.shape[0],), t, dtype=torch.long, device=dev)
    xk = y
    with torch.no_grad():
        for _ in range(fp_iters):
            eps = noise_fn(xk, tt)
            xk = (1 - w) * xk + w * ((s_t / s_p) * y - coef * eps)
    host.fp_grad_iters, host.fp_grad_lr = grad_iters, 0.02
    xk = host._grad_polish(xk, y, tt, s_t, s_p, b_t, b_p)
    with torch.no_grad():
        eps = noise_fn(xk, tt)
        fwd = (s_p / s_t) * (xk - b_t * eps) + b_p * eps
        res = float((fwd - y).pow(2).mean().sqrt())
    return xk, res


print("\n== A0: per-step invertibility at steps=50 (isolated true pairs) ==")
steps = 50
sched.set_timesteps(steps)
ts = [int(t) for t in sched.timesteps]
dt = 1000 // steps
m = 500
x = torch.from_numpy(np.random.default_rng(0)
                     .standard_normal((m, n)).astype(np.float32)).to(dev)
traj = []
with torch.no_grad():
    for t in ts:
        tt = torch.full((m,), t, dtype=torch.long, device=dev)
        y = sched.step(noise_fn(x, tt), t, x)
        traj.append((t, x, y))
        x = y
print(f"{'t':>5} {'fwd_residual':>13} {'dist_to_true':>13}"
      "   (dist >> residual == FOLD: wrong preimage, info destroyed)")
fold = []
for t, x_true, y in traj:
    xk, res = solve_step(y, t, dt)
    dist = float((xk - x_true).pow(2).mean().sqrt())
    is_fold = dist > 100 * max(res, 1e-8) and dist > 1e-2
    print(f"{t:>5} {res:>13.3e} {dist:>13.3e}" + ("   <-- FOLD" if is_fold else ""))
    if is_fold:
        fold.append(t)
print(f"folding steps at dt=20: {fold if fold else 'none detected'}")

print("\n== A1: sigma_ode, damped fp + grad polish, finer grids ==")
z_np = np.random.default_rng(0).standard_normal((1000, n)).astype(np.float32)
host.fp_iters, host.fp_damping = 40, 0.3
host.fp_grad_iters, host.fp_grad_lr = 30, 0.02
best_steps, best_sig = None, np.inf
for steps_ in (50, 200, 1000):
    host.steps = steps_
    x_emb = host.generate_embedding(z_np)
    r = host.invert_embedding(x_emb) - z_np
    sig = float(np.std(r))
    worst = sorted(host.fp_step_residuals, key=lambda p: -p[1])[:3]
    print(f"steps={steps_:<5d} sigma_ode={sig:.6f}   worst: "
          + ", ".join(f"t={t}:{v:.1e}" for t, v in worst))
    if sig < best_sig:
        best_steps, best_sig = steps_, sig

key = keygen(n, k=4, gamma=2.0, beta=0.05, seed=0)
print(f"\n== B: full round trip at steps={best_steps} "
      f"(sigma_ode={best_sig:.4f}), refinement sweep ==")
host.steps = best_steps
reports = {}
for iters in (1000, 3000):
    host.refine_iters = iters
    rep = inversion_report(host, key, num_rows=1000,
                           rng=np.random.default_rng(1))
    reports[iters] = rep
    print(f"refine_iters={iters:<5d} "
          f"sigma_onaxis={rep['sigma_onaxis']:.4f}  "
          f"mu_emp={rep['mu_emp']:.4f}  "
          f"rows@1e-6={rep['rows_needed_alpha1e6_power0.9']}")
best_iters = max(reports, key=lambda i: reports[i]["mu_emp"])
print(f"\nbest config: steps={best_steps}, fp40 w=0.3 + grad30, "
      f"refine_iters={best_iters}")
print(json.dumps(reports[best_iters], indent=2, default=str))

print("""
--------------------------------------------------------------------
READING THE OUTPUT
  A0 non-injective steps + A1 sigma_ode small at finer grids
      -> the residual is a discretization artifact; latent
         verification is feasible at the provider's grid.
  A1 sigma_ode ~constant across steps = 50/200/1000
      -> the residual is step-count-independent: the release map
         itself is ill-conditioned (the result reported in the paper).
  sigma_ode small but mu_emp tiny after refinement (B)
      -> the VAE table round trip, not the ODE, dominates the noise
         budget; embedding must move to data space.
--------------------------------------------------------------------""")
