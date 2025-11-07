# bayes_inverse_lfp.py
# Reconstruct μ_h(c), j0(c), k(x,y), and μ_res(t) from operando concentration movies
# Dependencies: numpy, torch, scipy (for PCHIP), tqdm (optional), matplotlib (optional for quick plots)

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from scipy.interpolate import PchipInterpolator

# -----------------------------
# Utilities
# -----------------------------

def get_device(prefer_gpu=True):
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def legendre_on_01(c, N):
    """
    Evaluate shifted Legendre polynomials P_n(c) on [0,1] for n=0..N.
    Uses the relation with standard Legendre on [-1,1]: x=2c-1.
    Returns tensor [..., N+1].
    """
    x = 2*c - 1
    # Recurrence for Legendre polynomials
    Ps = [torch.ones_like(x)]
    if N >= 1:
        Ps.append(x.clone())
    for n in range(1, N):
        Pn1 = Ps[n]
        Pn0 = Ps[n-1]
        Pn = ((2*n+1)*x*Pn1 - n*Pn0)/(n+1)
        Ps.append(Pn)
    return torch.stack(Ps, dim=-1)  # last axis: degree

def clamp01(x, eps=1e-7):
    return torch.clamp(x, eps, 1. - eps)

def finite_diff_time(c, t):
    """Centered finite difference for dc/dt on the discrete grid, shape [T,X,Y]."""
    cnp = c.cpu().numpy()
    dt = np.diff(t)
    dc = np.empty_like(cnp)
    dc[1:-1] = (cnp[2:] - cnp[:-2]) / (t[2:, None, None] - t[:-2, None, None])
    dc[0] = (cnp[1] - cnp[0]) / (dt[0])
    dc[-1] = (cnp[-1] - cnp[-2]) / (dt[-1])
    return torch.from_numpy(dc).to(c.device)

# -----------------------------
# Parameterizations
# -----------------------------

@dataclass
class ModelDims:
    N_mu: int = 3     # number of Legendre terms for μ_h (excluding P0 in the "excess" part)
    N_J: int = 3      # number of Legendre terms for j0 exp part (including P0)
    k_coarse: Tuple[int,int] = (15, 15)  # coarse grid for ln k (upsampled to X,Y)

@dataclass
class RegWeights:
    w_pix: float = 1.0          # weight for pixel MSE term
    w_rate: float = 1.0         # weight for average-rate penalty
    w_mu_prior: float = 1e-2    # L2 prior on a
    w_J_prior: float = 1e-2     # L2 prior on b
    w_k_grad: float = 1e-1      # smoothness on ln k (∑|∇ψ|^2)
    w_k_mean: float = 1e-2      # center ln k around zero (prevents drift)
    w_bounds: float = 1e2       # penalty to keep c in (0,1) during forward solve

@dataclass
class SolverOpts:
    substeps: int = 5           # substepping per frame interval for ODE stability
    alpha_init: float = 0.5     # Butler-Volmer symmetry factor initial guess
    alpha_bounds: Tuple[float,float] = (0.1, 0.9)
    lr: float = 1e-1
    max_iter: int = 300
    lbfgs_history: int = 100
    use_lbfgs: bool = True
    print_every: int = 25
    seed: int = 0

class InversionModel(torch.nn.Module):
    def __init__(self, X, Y, T, dims: ModelDims, device):
        super().__init__()
        self.X, self.Y, self.T = X, Y, T
        self.N_mu, self.N_J = dims.N_mu, dims.N_J
        self.k_coarse = dims.k_coarse

        # μ_h(c) = logit(c) + sum_{n=1..N_mu} a_n P_n(c)
        self.a = torch.nn.Parameter(torch.zeros(self.N_mu, device=device))

        # j0(c) = c*(1-c) * exp( sum_{n=0..N_J} b_n P_n(c) )
        self.b = torch.nn.Parameter(torch.zeros(self.N_J+1, device=device))

        # alpha (symmetry factor) in (0,1): optimize unconstrained via sigmoid transform
        alpha0 = torch.tensor(0.5, device=device).log() - torch.tensor(0.5, device=device).log()  # 0
        self.alpha_unconstrained = torch.nn.Parameter(torch.tensor(0.0, device=device))

        # ln k on coarse grid, upsampled to X×Y with bilinear interpolation
        kx, ky = self.k_coarse
        self.psi_coarse = torch.nn.Parameter(torch.zeros(1, 1, kx, ky, device=device))

        # μ_res(t): one free variable per time step
        self.mu_res = torch.nn.Parameter(torch.zeros(T, device=device))

    def alpha(self):
        return torch.sigmoid(self.alpha_unconstrained)  # in (0,1)

    def upsample_ln_k(self):
        psi = F.interpolate(self.psi_coarse, size=(self.X, self.Y), mode="bilinear", align_corners=True)
        return psi[0,0]  # [X,Y]

    def mu_h(self, c):
        c = clamp01(c)
        P = legendre_on_01(c, max(self.N_mu, self.N_J)).to(c.device)  # compute up to needed degree once
        # use P[... , 1: N_mu+1] for μ_h "excess"; logit is added
        mu = torch.log(c) - torch.log1p(-c)
        if self.N_mu > 0:
            mu = mu + (P[..., 1:self.N_mu+1] * self.a).sum(dim=-1)
        return mu

    def j0(self, c):
        c = clamp01(c)
        P = legendre_on_01(c, max(self.N_mu, self.N_J)).to(c.device)
        expo = (P[..., :self.N_J+1] * self.b).sum(dim=-1)
        return c*(1. - c)*torch.exp(expo)

    def reaction_rate(self, c, mu_res, ln_k):
        """
        R = k * j0(c) * [exp(-alpha*(μ - μ_res)) - exp((1-alpha)*(μ - μ_res))],
        using nondimensional μ (by kBT). See Eq. (S45) form. α∈(0,1).
        """
        mu = self.mu_h(c)
        j0c = self.j0(c)
        a = self.alpha()
        eta = mu - mu_res  # nondimensional
        term = torch.exp(-a*eta) - torch.exp((1.-a)*eta)
        return torch.exp(ln_k)*j0c*term

# -----------------------------
# Interpolated average rate I(t)
# -----------------------------

def average_c_time_series(c_data: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    # c_data: [T,X,Y], mask: [X,Y] boolean or None
    if mask is None:
        m = np.ones(c_data.shape[1:], dtype=bool)
    else:
        m = mask.astype(bool)
    area = m.sum()
    cbar = (c_data[:, m]).mean(axis=1)
    return cbar, area

def make_pchip_rate(t: np.ndarray, cbar: np.ndarray):
    # Section 6.7 suggests spline or PCHIP; we choose PCHIP to preserve monotonicity.
    # I(t) = d/dt [ cbar_interp(t) ]
    interp = PchipInterpolator(t, cbar)
    def I_time(tsamp: np.ndarray):
        return interp.derivative()(tsamp)
    return I_time

# -----------------------------
# Loss and forward simulator
# -----------------------------

def grad_smoothness(psi):
    # ∑ |∇ψ|^2 with forward differences, Neumann at edges
    dx = psi[1:, :] - psi[:-1, :]
    dy = psi[:, 1:] - psi[:, :-1]
    return (dx**2).mean() + (dy**2).mean()

def forward_simulate(model: InversionModel,
                     c0: torch.Tensor,
                     t: np.ndarray,
                     ln_k: torch.Tensor,
                     I_of_t,   # callable returning average rate at times
                     mask_torch: Optional[torch.Tensor],
                     substeps: int,
                     bounds_penalty: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Explicit RK4 with per-interval constant μ_res (optimized variable), enforced via a penalty
    that matches average R(c, μ_res) to I(t_mid).
    Returns:
      c_pred [T,X,Y], rate_mismatch_penalty scalar
    """
    Tn, X, Y = c0.shape[0], c0.shape[1], c0.shape[2]
    c_list = [c0[0]]
    # mask
    if mask_torch is None:
        wmask = torch.ones((X,Y), dtype=c0.dtype, device=c0.device)
    else:
        wmask = mask_torch.to(c0.dtype)

    rate_pen = c0.new_tensor(0.0)
    for n in range(Tn-1):
        t0, t1 = t[n], t[n+1]
        dt = (t1 - t0)/substeps
        # mid-time for average-rate target
        t_mid = 0.5*(t0 + t1)
        I_target = c0.new_tensor(I_of_t(np.array([t_mid]))[0])

        # evolve from c_list[n] to next state with RK4
        c = c_list[n]
        mu_res_n = model.mu_res[n]  # learnable scalar

        for _ in range(substeps):
            def R(cc):
                return model.reaction_rate(cc, mu_res_n, ln_k) * wmask

            k1 = R(c)
            k2 = R(c + 0.5*dt*k1)
            k3 = R(c + 0.5*dt*k2)
            k4 = R(c + dt*k3)
            c = c + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

            # bounds soft penalty (keep c in (0,1))
            bounds_loss = bounds_penalty * (torch.relu(-c).mean() + torch.relu(c-1.).mean())
            rate_pen = rate_pen + bounds_loss

        c_list.append(c)

        # average-rate mismatch penalty for this interval
        avg_R = (R(c_list[n]).mean())
        avg_loss = (avg_R - I_target)**2
        rate_pen = rate_pen + avg_loss

    c_pred = torch.stack(c_list, dim=0)
    return c_pred, rate_pen

# -----------------------------
# Main inversion routine
# -----------------------------

def invert_kinetics(
    c_data: np.ndarray,           # [T,X,Y], values in [0,1]
    t: np.ndarray,                # [T]
    mask: Optional[np.ndarray] = None,   # [X,Y] boolean
    dims: ModelDims = ModelDims(),
    reg: RegWeights = RegWeights(),
    opts: SolverOpts = SolverOpts(),
    priors: Optional[Dict] = None # e.g., {"c1":0.06,"c2":0.91} to softly enforce mosaic gap if desired
) -> Dict:
    """
    Returns a dict with learned parameters and diagnostics.
    """
    torch.manual_seed(opts.seed)
    device = get_device(prefer_gpu=True)
    Tn, X, Y = c_data.shape
    # tensors
    c_np = np.clip(c_data, 1e-6, 1-1e-6)
    c0_t = torch.from_numpy(c_np).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(mask.astype(bool)).to(device) if mask is not None else None

    # build rate interpolant I(t) from experimental average composition
    cbar, _ = average_c_time_series(c_np, mask)
    I_of_t = make_pchip_rate(t, cbar)  # §6.7
    # model
    model = InversionModel(X, Y, Tn, dims, device).to(device)

    # optimizer
    params = [p for p in model.parameters()]
    if opts.use_lbfgs:
        optimizer = torch.optim.LBFGS(params, lr=opts.lr, max_iter=opts.max_iter,
                                      history_size=opts.lbfgs_history, line_search_fn="strong_wolfe")
    else:
        optimizer = torch.optim.Adam(params, lr=opts.lr)

    # training loop
    iter_count = {"i": 0}
    P_mu = None  # cache basis degree if needed

    def closure():
        optimizer.zero_grad()
        ln_k = model.upsample_ln_k()  # [X,Y]
        c_pred, rate_pen = forward_simulate(model, c0_t, t, ln_k, I_of_t, mask_t, opts.substeps, reg.w_bounds)

        # pixel MSE over frames (exclude initial frame as in S109 normalization)
        if mask_t is None:
            mse_pix = (c_pred[1:] - c0_t[1:]).pow(2).mean()
        else:
            mse_pix = ((c_pred[1:] - c0_t[1:]) * mask_t).pow(2).sum() / mask_t.sum()

        # priors/regularizers (Gaussian priors for a,b; smoothness/mean for ln k)
        reg_mu = (model.a**2).mean()
        reg_J  = (model.b**2).mean()
        reg_k  = grad_smoothness(ln_k) + (ln_k.mean()**2)

        loss = (reg.w_pix * mse_pix
                + reg.w_rate * rate_pen
                + reg.w_mu_prior * reg_mu
                + reg.w_J_prior  * reg_J
                + reg.w_k_grad   * reg_k)

        loss.backward()
        iter_count["i"] += 1
        if (iter_count["i"] % opts.print_every) == 0 or iter_count["i"] == 1:
            with torch.no_grad():
                rmse = torch.sqrt(mse_pix)
                print(f"[{iter_count['i']:4d}] loss={loss.item():.4e}  rmse={rmse.item():.4e}  "
                      f"alpha={model.alpha().item():.3f}")
        return loss

    if opts.use_lbfgs:
        optimizer.step(closure)
    else:
        for i in range(opts.max_iter):
            loss = closure()
            optimizer.step()

    # Final forward pass & pack results
    with torch.no_grad():
        ln_k = model.upsample_ln_k()
        c_pred, rate_pen = forward_simulate(model, c0_t, t, ln_k, I_of_t, mask_t, opts.substeps, reg.w_bounds)
        if mask_t is None:
            rmse = torch.sqrt(((c_pred[1:] - c0_t[1:])**2).mean()).item()
        else:
            rmse = torch.sqrt((((c_pred[1:] - c0_t[1:]) * mask_t)**2).sum()/mask_t.sum()).item()
        alpha = model.alpha().item()

        # Build callable μ_h and j0 on numpy grid
        def mu_h_callable(c_in: np.ndarray):
            ct = torch.from_numpy(np.clip(c_in, 1e-6, 1-1e-6)).to(ln_k.device, dtype=torch.float32)
            with torch.no_grad():
                out = model.mu_h(ct).cpu().numpy()
            return out

        def j0_callable(c_in: np.ndarray):
            ct = torch.from_numpy(np.clip(c_in, 1e-6, 1-1e-6)).to(ln_k.device, dtype=torch.float32)
            with torch.no_grad():
                out = model.j0(ct).cpu().numpy()
            return out

        results = dict(
            rmse=rmse,
            a=model.a.detach().cpu().numpy(),
            b=model.b.detach().cpu().numpy(),
            alpha=alpha,
            ln_k=ln_k.detach().cpu().numpy(),
            k=np.exp(ln_k.detach().cpu().numpy()),
            mu_res=model.mu_res.detach().cpu().numpy(),
            c_pred=c_pred.detach().cpu().numpy(),
            mu_h=mu_h_callable,
            j0=j0_callable,
        )
    return results

# -----------------------------
# Example usage (commented)
# -----------------------------
if __name__ == "__main__":
    # Suppose you have c_data [T,X,Y], t [T], mask [X,Y]
    import json
    T, X, Y = 114, 30, 30
    c_data = np.load("ratio_tensor_smoothed.npy")        # shape [T,X,Y]
    t = np.load("time.npy")                  # shape [T]
    # mask = np.load("mask.npy")                      # shape [X,Y] or None
    # For demo, create dummy arrays:
    # c_data = np.random.rand(T, X, Y)*0.9+0.05
    # t = np.linspace(0, 1000, T)
    mask = np.ones((X,Y), dtype=bool)

    dims = ModelDims(N_mu=3, N_J=3, k_coarse=(15,15))
    reg  = RegWeights(w_pix=1.0, w_rate=10.0, w_mu_prior=1e-3, w_J_prior=1e-3, w_k_grad=1e-1, w_k_mean=1e-2)
    opts = SolverOpts(substeps=5, max_iter=200, use_lbfgs=True, print_every=10)
    out = invert_kinetics(c_data, t, mask, dims, reg, opts)
    print("RMSE:", out["rmse"])
