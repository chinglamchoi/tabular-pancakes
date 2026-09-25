"""TabSyn host wired to the TabWak reference pipeline (cluster-side).

Verified against https://github.com/chaoyitud/TabWak (commit of Sep 2025):
  generation  DDIMScheduler(num_train_timesteps=1000).generate(
                  model.noise_fn, latents, num_inference_steps=steps, eta=0.0)
              with latents ~ N(0, I_{in_dim}) and released embedding
              x = x_next * 2 + train_z.mean(0)                [watermark/sample.py]
  decoding    split_num_cat_target(x, info, num_inverse, cat_inverse)
              -> recover_data -> DataFrame                    [tabsyn/latent_utils.py]
  inversion   table -> process_data/preprocess_syn -> encoder latent
              -> (lat - mean)/2 -> DDIMScheduler.gen_reverse  [watermark/detection.py]

Our watermark replaces `latents` with the pancake prior; everything else in
the pipeline is untouched, so undetectability transfers by post-processing.

Requirements (cluster): torch, the TabWak checkout, its data/<dataname>/
directory (download_dataset.py + process_dataset.py) and trained VAE +
diffusion checkpoints under tabsyn/vae/ckpt/<dataname>/ and
tabsyn/ckpt/<dataname>/model.pt. This module imports torch lazily; the
rest of pancakemark stays numpy-only.

First run (inversion feasibility check):

    host = TabSynHost("/path/to/TabWak", "adult", device="cuda:0", steps=50)
    key  = keygen(n=host.latent_dim, k=4, gamma=2.0, beta=0.05, seed=0)
    print(inversion_report(host, key, num_rows=2000))
    # -> read sigma_onaxis / mu_emp; also compare invert_embedding (ODE-only
    #    error) vs invert (full VAE+preprocessing round trip) to see where
    #    the noise budget goes.
"""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

from .base import Host


class TabSynHost(Host):
    name = "tabsyn"

    def __init__(
        self,
        tabwak_root: str,
        dataname: str,
        device: str = "cuda:0",
        steps: int = 50,
        batch_size: int = 8192,
        workdir: str | None = None,
        refine_iters: int = 0,
        refine_lr: float = 0.1,
        fp_iters: int = 0,
        fp_damping: float = 0.5,
        fp_grad_iters: int = 0,
        fp_grad_lr: float = 0.02,
    ):
        import torch  # lazy; cluster-side dependency

        self.torch = torch
        root = os.path.abspath(tabwak_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from tabsyn.model import MLPDiffusion, DDIMModel, DDIMScheduler
        from tabsyn import latent_utils
        from tabsyn import process_syn_dataset

        self._lu = latent_utils
        self._psd = process_syn_dataset
        self.dataname = dataname
        self.device = device
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        # refine_iters > 0 switches the verifier from the plain VAE-encoder
        # pass to decoder-matching refinement (TabWak's detection path:
        # initialize at the encoder posterior mean, then optimize the latent
        # against the frozen decoder's reconstruction loss). The encoder
        # pass alone leaves a large latent residual (measured on Adult:
        # sigma_onaxis ~ 1.2); refinement closes the VAE leg of it.
        self.refine_iters = int(refine_iters)
        self.refine_lr = float(refine_lr)
        # fp_iters > 0 replaces TabWak's one-shot gen_reverse with per-step
        # inversion of their discrete forward map. TabWak's first-order
        # reverse leaves a step-count-INDEPENDENT residual (adult:
        # sigma_ode ~ 0.78 at 50/200/1000 steps -- upstream error is
        # amplified exponentially along the expanding reverse trajectory).
        # The plain fixed point x <- (s_t/s_p) y - coef*eps(x,t) oscillates
        # on the low-t cosine-schedule steps (|coef * d eps/dx| >= 1 there),
        # so the iteration is DAMPED (Krasnoselskii-Mann, fp_damping in
        # (0,1]); fp_grad_iters > 0 additionally polishes each step by Adam
        # on the forward residual ||step_t(x) - y||^2, which converges
        # regardless of contraction. Per-step convergence is recorded in
        # self.fp_step_residuals after each call. Verifier-side only:
        # generation is byte-identical to TabWak's.
        self.fp_iters = int(fp_iters)
        self.fp_damping = float(fp_damping)
        self.fp_grad_iters = int(fp_grad_iters)
        self.fp_grad_lr = float(fp_grad_lr)
        self.fp_step_residuals: list = []
        self.workdir = workdir or tempfile.mkdtemp(prefix="pancake_tabsyn_")
        os.makedirs(self.workdir, exist_ok=True)

        args = SimpleNamespace(dataname=dataname, device=device)
        (train_z, _, self.dataset_dir, ckpt_path, self.info,
         self.num_inverse, self.cat_inverse, self.d_num) = \
            latent_utils.get_input_generate(args, get_d_num=True)
        self.in_dim = train_z.shape[1]
        self.mean = train_z.mean(0).to(device)

        denoise_fn = MLPDiffusion(self.in_dim, 1024).to(device)
        self.model = DDIMModel(denoise_fn).to(device)
        state = torch.load(os.path.join(ckpt_path, "model.pt"),
                           map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():   # frozen: verifier never trains it
            p.requires_grad_(False)
        self.scheduler = DDIMScheduler(num_train_timesteps=1000)

    # ------------------------------------------------------------------
    @property
    def latent_dim(self) -> int:
        return self.in_dim

    def _to_batches(self, arr: np.ndarray):
        for i in range(0, arr.shape[0], self.batch_size):
            yield arr[i:i + self.batch_size]

    # ------------------------------------------------------------------
    def generate_embedding(self, z: np.ndarray) -> np.ndarray:
        """z (m, in_dim) -> released latent embedding x = ODE(z)*2 + mean."""
        torch = self.torch
        outs = []
        for zb in self._to_batches(np.asarray(z, dtype=np.float32)):
            latents = torch.from_numpy(zb).to(self.device)
            x = self.scheduler.generate(self.model.noise_fn, latents,
                                        num_inference_steps=self.steps,
                                        eta=0.0, device=self.device)
            outs.append((x * 2 + self.mean).detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def decode_embedding(self, x_emb: np.ndarray):
        """Embedding -> released table (pandas DataFrame, original schema)."""
        syn_num, syn_cat, syn_target = self._lu.split_num_cat_target(
            x_emb.astype(np.float32), self.info, self.num_inverse,
            self.cat_inverse, self.device)
        df = self._lu.recover_data(syn_num, syn_cat, syn_target, self.info)
        idx_name_mapping = {int(k): v
                            for k, v in self.info["idx_name_mapping"].items()}
        return df.rename(columns=idx_name_mapping)

    def generate(self, z: np.ndarray):
        return self.decode_embedding(self.generate_embedding(z))

    # ------------------------------------------------------------------
    def invert_embedding(self, x_emb: np.ndarray) -> np.ndarray:
        """Embedding -> z_hat (no VAE error). fp_iters=0: TabWak's one-shot
        gen_reverse; fp_iters>0: exact fixed-point inversion of the same
        discrete forward map."""
        if self.fp_iters > 0 or self.fp_grad_iters > 0:
            return self._invert_embedding_exact(x_emb)
        torch = self.torch
        outs = []
        for xb in self._to_batches(np.asarray(x_emb, dtype=np.float32)):
            lat = torch.from_numpy(xb).to(self.device)
            lat = (lat - self.mean) / 2
            z = self.scheduler.gen_reverse(self.model.noise_fn, lat,
                                           num_inference_steps=self.steps,
                                           eta=0.0, device=self.device)
            outs.append(z.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def _invert_embedding_exact(self, x_emb: np.ndarray) -> np.ndarray:
        """Invert scheduler.generate exactly, step by step.

        Their forward step at timestep t (eta=0, no clipping):
            y = (s_p/s_t) * (x - b_t * eps(x,t)) + b_p * eps(x,t)
        with s_* = sqrt(alpha_cumprod), b_* = sqrt(1 - alpha_cumprod),
        prev = t - dt (alpha_prev = final_alpha_cumprod = 1 when prev < 0).
        Solving for x given y and eps:
            x = (s_t/s_p) * y - ((s_t/s_p) * b_p - b_t) * eps(x, t)
        which we iterate to the fixed point (eps is implicit in x). The
        iteration starts at x0 = y; its FIRST pass equals TabWak's one-shot
        estimate, further passes converge to the exact preimage. All steps
        of the forward grid are inverted, including the final t=0 /
        final_alpha step that gen_reverse skips."""
        torch = self.torch
        sched = self.scheduler
        sched.set_timesteps(self.steps)
        ts = [int(t) for t in sched.timesteps]           # descending
        dt = sched.num_train_timesteps // sched.num_inference_steps
        acp = sched.alphas_cumprod
        final = float(sched.final_alpha_cumprod)
        w = self.fp_damping
        outs = []
        self.fp_step_residuals = []
        with torch.no_grad():
            for xb in self._to_batches(np.asarray(x_emb, dtype=np.float32)):
                x = torch.from_numpy(xb).to(self.device)
                x = (x - self.mean) / 2
                N = x.shape[0]
                for t in ts[::-1]:                        # invert last step first
                    prev = t - dt
                    a_t = float(acp[t])
                    a_p = float(acp[prev]) if prev >= 0 else final
                    s_t, s_p = a_t ** 0.5, a_p ** 0.5
                    b_t, b_p = (1 - a_t) ** 0.5, (1 - a_p) ** 0.5
                    coef = (s_t / s_p) * b_p - b_t
                    tt = torch.full((N,), t, dtype=torch.long,
                                    device=self.device)
                    y = x
                    xk = y
                    for _ in range(self.fp_iters):        # damped fixed point
                        eps = self.model.noise_fn(xk, tt)
                        xk = (1 - w) * xk + w * ((s_t / s_p) * y - coef * eps)
                    if self.fp_grad_iters > 0:            # unconditional polish
                        xk = self._grad_polish(xk, y, tt, s_t, s_p, b_t, b_p)
                    x = xk
                    # convergence diagnostic: forward-map residual at this step
                    eps = self.model.noise_fn(x, tt)
                    fwd = (s_p / s_t) * (x - b_t * eps) + b_p * eps
                    self.fp_step_residuals.append(
                        (t, float((fwd - y).pow(2).mean().sqrt())))
                outs.append(x.cpu().numpy())
        return np.concatenate(outs, axis=0)

    def _grad_polish(self, x0, y, tt, s_t, s_p, b_t, b_p):
        """Adam on the per-step forward residual ||step_t(x) - y||^2.
        Converges regardless of fixed-point contraction; model params are
        frozen so gradients flow only to x."""
        torch = self.torch
        with torch.enable_grad():
            x = x0.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=self.fp_grad_lr)
            for _ in range(self.fp_grad_iters):
                opt.zero_grad()
                eps = self.model.noise_fn(x, tt)
                fwd = (s_p / s_t) * (x - b_t * eps) + b_p * eps
                loss = (fwd - y).pow(2).sum(dim=1).mean()
                loss.backward()
                opt.step()
        return x.detach()

    def encode_table(self, df) -> np.ndarray:
        """Released table -> latent embedding via frozen preprocessing + VAE
        encoder (the verifier's canonicalization path, as in TabWak's
        detection.py)."""
        torch = self.torch
        csv_path = os.path.join(self.workdir, "suspect.csv")
        # process_data reads with pd.read_csv(..., header=info['header']).
        # For adult (and most TabWak datasets) header is None: the file must
        # be HEADERLESS and positional, in the original schema order --
        # otherwise the header row is parsed as data ('age' -> float crash).
        # (TabWak's own sample.py->detection.py writes a named header here
        # and would hit the same crash; we match the reader, not their bug.)
        cols = self.info.get("column_names") or []
        if cols and all(c in df.columns for c in cols):
            df = df[cols]                      # enforce positional order
        with_header = self.info.get("header", None) is not None
        df.to_csv(csv_path, index=False, header=with_header)
        self._psd.process_data(name=self.dataname, data_path=csv_path,
                               save_dir=self.workdir, k="")
        X_num, X_cat = self._psd.preprocess_syn(self.workdir,
                                                self.info["task_type"], k="")
        if self.refine_iters > 0:
            return self._decoder_refine(X_num, X_cat)
        with torch.no_grad():
            lat = self._lu.get_encoder_latent(X_num, X_cat, self.info,
                                              self.device)
        return lat.detach().cpu().numpy()

    def _decoder_refine(self, X_num, X_cat) -> np.ndarray:
        """Decoder-matching latent inversion (TabWak detection.py's
        get_decoder_latent, reimplemented without wandb/deprecated kwargs):
        start at the encoder posterior mean, optimize the latent tokens to
        minimize the frozen decoder's reconstruction loss (num MSE + mean
        categorical CE), with a hinge keeping tokens within mu_z +- 3 std_z."""
        torch = self.torch
        dev = self.device
        vae = self.info["model"]
        dec = self.info["pre_decoder"]
        # Module.to() is IN-PLACE and these modules are shared with TabWak's
        # latent_utils (split_num_cat_target feeds pre_decoder CPU tensors),
        # so remember where they live and restore on exit.
        vae_dev0 = next(vae.parameters()).device
        dec_dev0 = next(dec.parameters()).device
        vae = vae.to(dev).eval()
        dec = dec.to(dev).eval()
        outs = []
        Xn_all = np.asarray(X_num, dtype=np.float32)
        Xc_all = np.asarray(X_cat)
        try:
            for i in range(0, Xn_all.shape[0], self.batch_size):
                Xn = torch.tensor(Xn_all[i:i + self.batch_size],
                                  dtype=torch.float32, device=dev)
                Xc = torch.tensor(Xc_all[i:i + self.batch_size],
                                  dtype=torch.long, device=dev)
                with torch.no_grad():
                    _, mu_z, logvar_z = vae.VAE(Xn, Xc)
                    mu_z = mu_z[:, 1:, :].detach()
                    std_z = torch.exp(0.5 * logvar_z[:, 1:, :].detach())
                lat = mu_z.clone().requires_grad_(True)
                opt = torch.optim.AdamW([lat], lr=self.refine_lr)
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=100)
                ce = torch.nn.CrossEntropyLoss()
                for _ in range(self.refine_iters):
                    opt.zero_grad()
                    rec_num, rec_cat = dec(lat)[:2]
                    mse = (Xn - rec_num).pow(2).mean()
                    ce_sum, nc = 0.0, 0
                    for j, head in enumerate(rec_cat):
                        if head is not None:
                            ce_sum = ce_sum + ce(head, Xc[:, j])
                            nc += 1
                    hinge = (torch.relu(lat - (mu_z + 3 * std_z)).mean()
                             + torch.relu((mu_z - 3 * std_z) - lat).mean())
                    loss = mse + (ce_sum / nc if nc else 0.0) + hinge
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    opt.step()
                    sched.step(loss.item())
                B, T, D = lat.shape
                outs.append(lat.detach().view(B, T * D).cpu().numpy())
        finally:
            vae.to(vae_dev0)
            dec.to(dec_dev0)
        return np.concatenate(outs, axis=0)

    def invert(self, table) -> np.ndarray:
        """Full verifier path: table -> encoder latent -> reverse ODE."""
        return self.invert_embedding(self.encode_table(table))
