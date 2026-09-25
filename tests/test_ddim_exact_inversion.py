"""Algebra check for TabSynHost._invert_embedding_exact (numpy mirror).

Mirrors TabWak's DDIMScheduler.generate loop (eta=0, no clipping,
final_alpha_cumprod=1) with a synthetic Lipschitz eps-net, then inverts it
with the exact fixed-point update used by the host:

    x = (s_t/s_p) * y - ((s_t/s_p) * b_p - b_t) * eps(x, t)

Assertions: the fixed point recovers the input latents to float precision,
while the one-shot estimate (fp=1, i.e. TabWak's gen_reverse up to its
skipped final step) leaves a visibly larger residual.
"""
import numpy as np

T_TRAIN = 1000


def _schedule():
    betas = np.linspace(1e-4, 0.02, T_TRAIN)
    acp = np.cumprod(1.0 - betas)
    return acp


def _eps_net(rng, n):
    W1 = rng.standard_normal((n + 1, 2 * n)) / np.sqrt(n + 1)
    W2 = rng.standard_normal((2 * n, n)) / np.sqrt(2 * n)

    def eps(x, t):
        h = np.tanh(np.c_[x, np.full((x.shape[0], 1), t / T_TRAIN)] @ W1)
        return h @ W2
    return eps


def _forward(z, eps, acp, steps):
    ts = np.arange(0, T_TRAIN, T_TRAIN // steps)[::-1]
    dt = T_TRAIN // steps
    x = z.copy()
    for t in ts:
        prev = t - dt
        a_t = acp[t]
        a_p = acp[prev] if prev >= 0 else 1.0
        e = eps(x, t)
        x0_pred = (x - np.sqrt(1 - a_t) * e) / np.sqrt(a_t)
        x = np.sqrt(a_p) * x0_pred + np.sqrt(1 - a_p) * e
    return x


def _invert(y, eps, acp, steps, fp_iters, w=1.0):
    ts = np.arange(0, T_TRAIN, T_TRAIN // steps)[::-1]
    dt = T_TRAIN // steps
    x = y.copy()
    for t in ts[::-1]:
        prev = t - dt
        a_t = acp[t]
        a_p = acp[prev] if prev >= 0 else 1.0
        s_t, s_p = np.sqrt(a_t), np.sqrt(a_p)
        b_t, b_p = np.sqrt(1 - a_t), np.sqrt(1 - a_p)
        coef = (s_t / s_p) * b_p - b_t
        yy = x
        xk = yy
        for _ in range(fp_iters):
            xk = (1 - w) * xk + w * ((s_t / s_p) * yy - coef * eps(xk, t))
        x = xk
    return x


def test_fixed_point_inversion_is_exact():
    rng = np.random.default_rng(0)
    n, m, steps = 16, 200, 50
    eps = _eps_net(rng, n)
    acp = _schedule()
    z = rng.standard_normal((m, n))
    y = _forward(z, eps, acp, steps)

    err_fp = np.std(_invert(y, eps, acp, steps, fp_iters=25) - z)
    err_oneshot = np.std(_invert(y, eps, acp, steps, fp_iters=1) - z)
    assert err_fp < 1e-8, f"exact inversion not exact: {err_fp}"
    assert err_oneshot > 10 * max(err_fp, 1e-12), (
        f"one-shot unexpectedly exact ({err_oneshot}); test net too tame")


def test_single_step_update_inverts_their_step():
    """One forward step, algebraic identity (no iteration needed when eps
    is evaluated at the true preimage)."""
    rng = np.random.default_rng(1)
    n = 8
    eps = _eps_net(rng, n)
    acp = _schedule()
    t, dt = 980, 20
    a_t, a_p = acp[t], acp[t - dt]
    x = rng.standard_normal((5, n))
    e = eps(x, t)
    y = (np.sqrt(a_p) * (x - np.sqrt(1 - a_t) * e) / np.sqrt(a_t)
         + np.sqrt(1 - a_p) * e)
    s_t, s_p = np.sqrt(a_t), np.sqrt(a_p)
    coef = (s_t / s_p) * np.sqrt(1 - a_p) - np.sqrt(1 - a_t)
    x_rec = (s_t / s_p) * y - coef * e
    assert np.allclose(x_rec, x, atol=1e-12)


def test_damping_rescues_stiff_steps():
    """A stiff eps (fixed-point Jacobian eigenvalue -1.5 on every step, as
    on the low-t cosine steps of the adult model) makes the plain fixed
    point diverge; Krasnoselskii-Mann damping converges. Mirrors the
    fp_damping remedy in TabSynHost._invert_embedding_exact."""
    rng = np.random.default_rng(2)
    n, m, steps = 8, 100, 50
    acp = _schedule()
    ts = np.arange(0, T_TRAIN, T_TRAIN // steps)
    dt = T_TRAIN // steps
    # per-step linear eps with FP eigenvalue lambda = -coef*c = -1.5
    c_by_t = {}
    for t in ts:
        prev = t - dt
        a_t, a_p = acp[t], (acp[prev] if prev >= 0 else 1.0)
        coef = np.sqrt(a_t / a_p) * np.sqrt(1 - a_p) - np.sqrt(1 - a_t)
        c_by_t[int(t)] = 1.5 / coef if abs(coef) > 1e-12 else 0.0

    def eps(x, t):
        return c_by_t[int(t)] * x

    z = rng.standard_normal((m, n))
    y = _forward(z, eps, acp, steps)
    err_plain = np.std(_invert(y, eps, acp, steps, fp_iters=40, w=1.0) - z)
    err_damped = np.std(_invert(y, eps, acp, steps, fp_iters=40, w=0.4) - z)
    assert not np.isfinite(err_plain) or err_plain > 1.0
    assert err_damped < 1e-6, f"damped iteration did not converge: {err_damped}"
