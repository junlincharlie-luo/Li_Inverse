#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse kinetics for Li concentration movies (reaction-limited, Butler–Volmer).
CPU-friendly for macOS. Works well for small grids like 114×30×30.

USAGE:
  python inverse_li_inversion_mac.py --frames frames.npy --times times.npy --mask mask.npy \
      --outdir results_mac --iters 150

Expected shapes:
  frames.npy: [T, H, W] float in [0,1]
  times.npy:  [T] (monotone increasing; units consistent, e.g., hours)
  mask.npy:   [H, W] boolean (True = inside particle). If omitted → all True.

Outputs:
  results_mac/
    mu_curve.png, j0_curve.png, ln_k.png
    mu_coeff.npy, j_coeff.npy, ln_k.npy
    history.npy
"""

import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --------------------------
# Small helpers
# --------------------------
def legendre_on_01(c: torch.Tensor, N: int) -> torch.Tensor:
    x = 2*c - 1
    Ps = [torch.ones_like(x)]
    if N > 1:
        Ps.append(x.clone())
    for n in range(2, N):
        Pn = ((2*n-1)*x*Ps[-1] - (n-1)*Ps[-2]) / n
        Ps.append(Pn)
    return torch.stack(Ps, dim=-1)  # (..., N)

def laplacian5(u: torch.Tensor, hx: float, hy: float, mask: torch.Tensor) -> torch.Tensor:
    # Handle both 2D [H,W] and 3D [B,H,W] tensors
    if u.ndim == 2:
        u = u.unsqueeze(0)  # Add batch dimension [1,H,W]
        squeeze_output = True
    else:
        squeeze_output = False

    up = torch.nn.functional.pad(u, (1,1,1,1), mode='replicate')
    Lx = (up[:, 1:-1,2:] - 2*up[:, 1:-1,1:-1] + up[:, 1:-1,:-2]) / (hx*hx)
    Ly = (up[:, 2:,1:-1] - 2*up[:, 1:-1,1:-1] + up[:, :-2,1:-1]) / (hy*hy)
    L = Lx + Ly

    if squeeze_output:
        L = L.squeeze(0)  # Remove batch dimension

    return torch.where(mask, L, torch.zeros_like(L))

def pchip_avg_rate(times: np.ndarray, avg_c: np.ndarray):
    """Monotone cubic (Fritsch–Carlson) PCHIP derivative approximation."""
    t = times.astype(float)
    y = avg_c.astype(float)
    n = len(t)
    h = np.diff(t)
    m = np.diff(y)/h
    d = np.zeros_like(y)
    d[0] = m[0]; d[-1] = m[-1]
    for i in range(1, n-1):
        if m[i-1]*m[i] <= 0:
            d[i] = 0.0
        else:
            w1 = 2*h[i] + h[i-1]
            w2 = h[i] + 2*h[i-1]
            d[i] = (w1 + w2) / (w1/m[i-1] + w2/m[i])
    t_dense = np.linspace(t[0], t[-1], 2001)
    dcdt_dense = np.zeros_like(t_dense)
    for k, tk in enumerate(t_dense):
        i = min(max(np.searchsorted(t, tk)-1, 0), n-2)
        s = (tk - t[i]) / h[i]
        dy_ds = (y[i]*(6*s*s - 6*s)
                 + y[i+1]*(-6*s*s + 6*s)
                 + d[i]*h[i]*(3*s*s - 4*s + 1)
                 + d[i+1]*h[i]*(3*s*s - 2*s))
        dcdt_dense[k] = dy_ds / h[i]
    return t_dense, dcdt_dense

def interp_linear(x, y, xq: torch.Tensor) -> torch.Tensor:
    # piecewise linear interpolation (numpy arrays x,y; xq torch scalar/tensor)
    xv = xq.detach().cpu().numpy()
    idx = np.clip(np.searchsorted(x, xv), 1, len(x)-1)
    x0, x1 = x[idx-1], x[idx]
    y0, y1 = y[idx-1], y[idx]
    w = (xv - x0)/(x1 - x0)
    out = (1-w)*y0 + w*y1
    # Ensure out is always an array, even for scalar inputs
    out = np.atleast_1d(out)
    return torch.from_numpy(out).to(xq.device, xq.dtype).squeeze()

# --------------------------
# Constitutive laws (learned)
# --------------------------
class Constitutive(nn.Module):
    def __init__(self, N_mu=5, N_j=5, alpha=0.5, kBT=1.0, Bmin=0.19e9, kappa=5.02e-10):
        super().__init__()
        self.N_mu = N_mu; self.N_j = N_j
        self.a = nn.Parameter(torch.zeros(N_mu))
        self.b = nn.Parameter(torch.zeros(N_j))
        self.alpha = alpha
        self.kBT = kBT
        self.kappa = kappa
        self.Bmin = Bmin
        # miscibility endpoints used for neutral shift
        self.register_buffer('c1', torch.tensor(0.0126))
        self.register_buffer('c2', torch.tensor(1.0-0.0126))

    def mu_h(self, c: torch.Tensor) -> torch.Tensor:
        P = legendre_on_01(c, self.N_mu)
        mu_raw = torch.log(torch.clamp(c,1e-6,1-1e-6)) - torch.log(torch.clamp(1-c,1e-6,1-1e-6)) \
                 + (P * self.a).sum(dim=-1)
        with torch.no_grad():
            P1 = legendre_on_01(self.c1, self.N_mu); P2 = legendre_on_01(self.c2, self.N_mu)
        mu_c1 = torch.log(self.c1) - torch.log(1-self.c1) + (P1*self.a).sum()
        mu_c2 = torch.log(self.c2) - torch.log(1-self.c2) + (P2*self.a).sum()
        shift = 0.5*(mu_c1 + mu_c2)
        return mu_raw - shift

    def j0(self, c: torch.Tensor) -> torch.Tensor:
        P = legendre_on_01(c, self.N_j)
        return torch.clamp(c*(1-c), 0.0, 0.25) * torch.exp((P * self.b).sum(dim=-1))

    def R_BV(self, c: torch.Tensor, mu: torch.Tensor, mu_res: torch.Tensor) -> torch.Tensor:
        eta = (mu - mu_res)/self.kBT
        return self.j0(c) * (torch.exp(-self.alpha*eta) - torch.exp((1.0-self.alpha)*eta))

# --------------------------
# Inverse model
# --------------------------
class InverseKinetics(nn.Module):
    def __init__(self, frames_np: np.ndarray, times_np: np.ndarray, mask_np: np.ndarray,
                 pixel_size: float=50e-9, alpha: float=0.5, Bmin: float=0.19e9,
                 kappa: float=5.02e-10, kBT: float=1.0, N_mu: int=5, N_j: int=5,
                 substeps: int=5, lrk: float=1.0, device: str='cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.frames = torch.from_numpy(frames_np).float().to(self.device)  # [T,H,W]
        self.times_np = np.asarray(times_np).astype(float)
        self.T, self.H, self.W = self.frames.shape
        self.register_buffer('mask', torch.from_numpy(mask_np.astype(bool)).to(self.device))
        self.const = Constitutive(N_mu=N_mu, N_j=N_j, alpha=alpha, kBT=kBT, Bmin=Bmin, kappa=kappa).to(self.device)
        self.psi = nn.Parameter(torch.zeros(self.H, self.W, device=self.device))  # ln k(x,y)
        self.hx = pixel_size; self.hy = pixel_size
        self.substeps = substeps
        self.lrk = lrk

        # average rate (PCHIP) from data
        avg_c = frames_np.reshape(self.T, -1)[:, mask_np.reshape(-1)].mean(axis=1)
        self.t_dense, self.dcdt_dense = pchip_avg_rate(self.times_np, avg_c)

    def forward_simulate(self) -> torch.Tensor:
        c = self.frames[0].clone()
        out = [c]
        kfield = torch.exp(self.psi)
        for k in range(self.T-1):
            t0, t1 = self.times_np[k], self.times_np[k+1]
            dt = (t1 - t0)/self.substeps
            for s in range(self.substeps):
                t_now = torch.tensor(t0 + (s+0.5)*dt, device=self.device)
                rbar = interp_linear(self.t_dense, self.dcdt_dense, t_now)
                cbar = c[self.mask].mean()
                mu_bulk = self.const.mu_h(c)
                lap = laplacian5(c, self.hx, self.hy, self.mask)
                mu = mu_bulk - (self.const.kappa * lap) + (self.const.Bmin)*(c - cbar)

                # solve scalar mu_res by Newton to match mean(R)=rbar
                mu_res = cbar.new_tensor(0.0)
                for _ in range(8):
                    R = kfield * self.const.R_BV(c, mu, mu_res)
                    Rm = R[self.mask]
                    f = Rm.mean() - rbar
                    if torch.abs(f) < 1e-9: break
                    eta = (mu[self.mask] - mu_res)/self.const.kBT
                    j0v = self.const.j0(c[self.mask])
                    dRdeta = j0v * ( -self.const.alpha*torch.exp(-self.const.alpha*eta)
                                     - (1.0-self.const.alpha)*torch.exp((1.0-self.const.alpha)*eta) )
                    dRdmures = -(dRdeta)/self.const.kBT
                    df = dRdmures.mean()
                    mu_res = mu_res - f/torch.clamp(df, 1e-9)

                # explicit Euler
                c = c + dt * kfield * self.const.R_BV(c, mu, mu_res)
                c = torch.clamp(c, 0.0, 1.0)
            out.append(c)
        return torch.stack(out, dim=0)

    def loss(self):
        c_pred = self.forward_simulate()
        mis = ((c_pred[1:] - self.frames[1:])**2)[self.mask].mean()

        reg_mu = (self.const.a**2).mean()
        reg_j  = (self.const.b**2).mean()

        lap_psi = laplacian5(self.psi, self.hx, self.hy, self.mask)
        reg_k_l2  = (self.psi[self.mask]**2).mean()
        reg_k_lap = (lap_psi[self.mask]**2).mean()

        total = mis + 1e-3*reg_mu + 1e-3*reg_j + self.lrk*(1e-3*reg_k_l2 + 5e-4*reg_k_lap)
        terms = dict(mis=float(mis.detach().cpu()),
                     reg_mu=float(reg_mu.detach().cpu()),
                     reg_j=float(reg_j.detach().cpu()),
                     reg_k_l2=float(reg_k_l2.detach().cpu()),
                     reg_k_lap=float(reg_k_lap.detach().cpu()))
        return total, terms

# --------------------------
# Fit wrapper + plotting
# --------------------------
def fit_inverse(frames, times, mask, outdir="results_mac",
                pixel_size=50e-9, N_mu=5, N_j=5, alpha=0.5,
                Bmin=0.19e9, kappa=5.02e-10, kBT=1.0,
                lrk=1.0, iters=150, substeps=5, seed=0):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    model = InverseKinetics(frames, times, mask, pixel_size, alpha, Bmin, kappa,
                            kBT, N_mu, N_j, substeps=substeps, lrk=lrk, device='cpu')

    # LBFGS on CPU works fine for this size
    # Combine all parameters into a single group (LBFGS doesn't support parameter groups in newer PyTorch)
    all_params = list(model.const.parameters()) + [model.psi]
    optimizer = optim.LBFGS(all_params, lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    history = []
    for it in range(iters):
        def closure():
            optimizer.zero_grad()
            L,_ = model.loss()
            L.backward()
            return L
        optimizer.step(closure)
        with torch.no_grad():
            L, terms = model.loss()
        history.append((float(L.detach().cpu()), terms))
        print(f"[{it+1:03d}] loss={history[-1][0]:.6e} mis={terms['mis']:.6e}")

    # export curves/maps
    cgrid = torch.linspace(0.001, 0.999, 500)
    with torch.no_grad():
        mu_curve = model.const.mu_h(cgrid).cpu().numpy()
        j0_curve = model.const.j0(cgrid).cpu().numpy()
        psi_map = model.psi.cpu().numpy()

    np.save(os.path.join(outdir, "mu_coeff.npy"), model.const.a.detach().cpu().numpy())
    np.save(os.path.join(outdir, "j_coeff.npy"), model.const.b.detach().cpu().numpy())
    np.save(os.path.join(outdir, "ln_k.npy"), psi_map)
    np.save(os.path.join(outdir, "history.npy"), np.array(history, dtype=object))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(dict(N_mu=N_mu, N_j=N_j, alpha=alpha, Bmin=Bmin, kappa=kappa,
                       pixel_size=pixel_size, lrk=lrk, iters=iters, substeps=substeps), f, indent=2)

    # plots
    cgrid_np = cgrid.numpy()
    plt.figure(); plt.plot(cgrid_np, mu_curve); plt.xlabel("c"); plt.ylabel(r"$\mu_h(c)$"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mu_curve.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(cgrid_np, j0_curve); plt.xlabel("c"); plt.ylabel(r"$j_0(c)$"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "j0_curve.png"), dpi=200); plt.close()

    plt.figure(); plt.imshow(psi_map, origin="lower"); plt.colorbar(label="ln k")
    plt.title("ln k(x,y)"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ln_k.png"), dpi=200); plt.close()

    return dict(mu_curve=(cgrid_np, mu_curve), j0_curve=(cgrid_np, j0_curve), ln_k=psi_map, history=history)

# --------------------------
# CLI
# --------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=str, required=True, help="path to frames.npy [T,H,W]")
    p.add_argument("--times",  type=str, required=True, help="path to times.npy [T]")
    p.add_argument("--mask",   type=str, default="", help="path to mask.npy [H,W] bool; default = all ones")
    p.add_argument("--outdir", type=str, default="results_mac")
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--substeps", type=int, default=5)
    p.add_argument("--N_mu", type=int, default=5)
    p.add_argument("--N_j", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--Bmin", type=float, default=0.19e9)
    p.add_argument("--kappa", type=float, default=5.02e-10)
    p.add_argument("--kBT", type=float, default=1.0)
    p.add_argument("--pixel_size", type=float, default=50e-9)
    p.add_argument("--lrk", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    frames = np.load(args.frames)  # [T,H,W]
    times  = np.load(args.times)   # [T]
    if args.mask and os.path.exists(args.mask):
        mask = np.load(args.mask).astype(bool)
    else:
        mask = np.ones(frames.shape[1:], dtype=bool)

    assert frames.ndim==3, "frames must be [T,H,W]"
    assert times.ndim==1 and len(times)==frames.shape[0], "times must match T"
    assert mask.shape==(frames.shape[1], frames.shape[2]), "mask must be [H,W]"

    # optional clipping to avoid log(0)
    frames = np.clip(frames, 1e-6, 1-1e-6)

    torch.set_num_threads(min(8, os.cpu_count() or 2))
    fit_inverse(frames, times, mask,
                outdir=args.outdir, pixel_size=args.pixel_size,
                N_mu=args.N_mu, N_j=args.N_j, alpha=args.alpha,
                Bmin=args.Bmin, kappa=args.kappa, kBT=args.kBT,
                lrk=args.lrk, iters=args.iters, substeps=args.substeps, seed=args.seed)

if __name__ == "__main__":
    main()