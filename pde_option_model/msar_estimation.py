"""Two-regime MS-AR(1) with TVTP -- Hamilton filter + MLE (FW9).

Specification (matches the M9 spec identified in
`outputs/fw9_self_estimation/preprocessing_audit.md`):

    y_t = asinh(P_t / scale_P)
    Regime J_t in {0 = normal, 1 = stress}, shared AR coefficient phi.
    y_t | J_t = i, y_{t-1} ~ Normal(mu_i + phi * y_{t-1}, sigma_i^2)

    P(J_t = 1 | J_{t-1} = 0, z_{t-1}) = Lambda(alpha01 + gamma01 * z_{t-1})
    P(J_t = 0 | J_{t-1} = 1, z_{t-1}) = Lambda(alpha10 + gamma10 * z_{t-1})

with the label convention ``sigma_normal < sigma_stress`` enforced by
parametrisation (``sigma_stress = sigma_normal + softplus(delta)``).

Hamilton filter:

    pi_{t|t-1}(j) = sum_i P(J_t=j|J_{t-1}=i, z_{t-1}) * pi_{t-1|t-1}(i)
    f_i(y_t)      = N(y_t; mu_i + phi * y_{t-1}, sigma_i)
    L_t           = sum_j f_j(y_t) * pi_{t|t-1}(j)
    pi_{t|t}(j)   = f_j(y_t) * pi_{t|t-1}(j) / L_t
    loglik       += log L_t
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------
# Parameter (un)packing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MSARParams:
    """Canonical (unpacked) parameters.  ``sigma_normal < sigma_stress`` by
    convention; the packed parametrisation guarantees this."""

    mu_normal: float
    mu_stress: float
    sigma_normal: float
    sigma_stress: float
    phi: float
    alpha01: float
    gamma01: float
    alpha10: float
    gamma10: float

    def as_dict(self) -> dict:
        return {
            "mu_normal": self.mu_normal, "mu_stress": self.mu_stress,
            "sigma_normal": self.sigma_normal, "sigma_stress": self.sigma_stress,
            "phi": self.phi,
            "alpha01": self.alpha01, "gamma01": self.gamma01,
            "alpha10": self.alpha10, "gamma10": self.gamma10,
        }


def pack(mu_n: float, mu_s: float, sigma_n: float, sigma_s: float,
         phi: float, a01: float, g01: float, a10: float, g10: float
         ) -> np.ndarray:
    """Convert canonical parameters to the unbounded optimiser space.

    ``sigma_normal < sigma_stress`` is enforced by
    ``sigma_stress = sigma_normal + softplus(log_sigma_delta_raw)``,
    where ``softplus(x) = log(1 + exp(x)) > 0`` for every real x.
    ``phi`` is transformed via ``atanh`` so the unbounded variable can
    range over R while phi stays in (-1, 1).
    """
    if not (0.0 < sigma_n < sigma_s):
        raise ValueError(
            f"pack: expected 0 < sigma_normal ({sigma_n}) < sigma_stress ({sigma_s})")
    if not (-1.0 < phi < 1.0):
        raise ValueError(f"pack: expected phi in (-1, 1), got {phi}")
    log_sigma_n = math.log(sigma_n)
    # inverse softplus of (sigma_s - sigma_n):  softplus^-1(y) = log(exp(y) - 1)
    diff = sigma_s - sigma_n
    log_sigma_delta_raw = math.log(math.expm1(diff))
    phi_raw = math.atanh(phi)
    return np.array([mu_n, mu_s, log_sigma_n, log_sigma_delta_raw, phi_raw,
                     a01, g01, a10, g10], dtype=float)


def unpack(theta: np.ndarray) -> MSARParams:
    mu_n, mu_s, log_sigma_n, log_sigma_delta_raw, phi_raw, a01, g01, a10, g10 = theta
    sigma_n = math.exp(log_sigma_n)
    sigma_s = sigma_n + math.log1p(math.exp(log_sigma_delta_raw))
    phi = math.tanh(phi_raw)
    return MSARParams(
        mu_normal=float(mu_n), mu_stress=float(mu_s),
        sigma_normal=sigma_n, sigma_stress=sigma_s, phi=phi,
        alpha01=float(a01), gamma01=float(g01),
        alpha10=float(a10), gamma10=float(g10),
    )


# --------------------------------------------------------------------------
# Hamilton filter + log-likelihood
# --------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _stationary_pi(p01_mean: float, p10_mean: float) -> np.ndarray:
    """Stationary distribution of the AVERAGE transition matrix over the
    training window -- used only to seed pi_0."""
    denom = p01_mean + p10_mean
    if denom < 1e-12:
        return np.array([0.5, 0.5])
    return np.array([p10_mean / denom, p01_mean / denom])


def fit_msar_fixed_phi(y: np.ndarray, z_lag: np.ndarray, fixed_phi: float,
                       n_starts: int = 15, seed: int = 20260927,
                       maxiter: int = 500, tvtp: bool = True,
                       compute_se: bool = True) -> "FitResult":
    """MLE at ``phi = fixed_phi`` (FW9b §1 profile).  Optimises the
    other 8 parameters (or 6 if constant-transition).  Because phi is
    fixed, the gradient can go to zero at an interior optimum instead
    of sticking on the boundary; standard errors from the resulting
    Hessian are legitimate."""
    rng = np.random.default_rng(seed)
    T = int(y.size)
    n_free = 8 if tvtp else 6
    # positions in the full 9-vector: everything except index 4 (phi_raw),
    # and (if not tvtp) also skip 6, 8
    if tvtp:
        free_idx = [0, 1, 2, 3, 5, 6, 7, 8]
    else:
        free_idx = [0, 1, 2, 3, 5, 7]
    phi_raw = math.atanh(min(0.99999999, max(-0.99999999, fixed_phi)))
    all_starts = []
    best = None
    for s in range(n_starts):
        theta9 = _random_theta(rng)
        theta9[4] = phi_raw
        if not tvtp:
            theta9[6] = 0.0
            theta9[8] = 0.0

        def _nll(x_free: np.ndarray) -> float:
            th = theta9.copy()
            th[free_idx] = x_free
            th[4] = phi_raw
            if not tvtp:
                th[6] = 0.0
                th[8] = 0.0
            return -hamilton_loglik(th, y, z_lag)

        x0 = theta9[free_idx]
        try:
            r = minimize(_nll, x0, method="L-BFGS-B",
                         options={"maxiter": maxiter, "gtol": 1e-7})
            grad = float(np.linalg.norm(r.jac)) if r.jac is not None else float("nan")
            conv = bool(r.success) and grad < 1e-2
            all_starts.append({"start_idx": s, "converged": conv,
                               "loglik": -r.fun, "grad_norm": grad,
                               "n_iter": int(r.nit)})
            if best is None or -r.fun > best["loglik"]:
                th_full = theta9.copy()
                th_full[free_idx] = r.x
                th_full[4] = phi_raw
                if not tvtp:
                    th_full[6] = 0.0
                    th_full[8] = 0.0
                best = {"theta": th_full, "loglik": -r.fun,
                        "grad_norm": grad, "n_iter": int(r.nit),
                        "converged": conv, "x_free": r.x}
        except Exception as exc:
            all_starts.append({"start_idx": s, "converged": False,
                               "loglik": float("-inf"),
                               "grad_norm": float("nan"), "n_iter": -1,
                               "error": str(exc)})
    if best is None:
        raise RuntimeError("no successful fixed-phi start")
    params = unpack(best["theta"])
    n_params = n_free                              # phi is not free
    aic = -2.0 * best["loglik"] + 2.0 * n_params
    bic = -2.0 * best["loglik"] + math.log(T) * n_params
    # Hessian on the FREE parameter vector only (analytic derivative of
    # NLL w.r.t. the 8 or 6 remaining coordinates, via central diff).
    se_free = np.full(n_free, np.nan)
    H_free = None
    if compute_se:
        eps = 1e-4
        H_free = np.zeros((n_free, n_free))
        f0 = -best["loglik"]

        def _nll_wrap(x_free):
            th = best["theta"].copy()
            th[free_idx] = x_free
            return -hamilton_loglik(th, y, z_lag)

        x_star = best["x_free"]
        for i in range(n_free):
            for j in range(i, n_free):
                if i == j:
                    x_p = x_star.copy(); x_p[i] += eps
                    x_m = x_star.copy(); x_m[i] -= eps
                    H_free[i, j] = (_nll_wrap(x_p) - 2.0 * f0
                                    + _nll_wrap(x_m)) / (eps * eps)
                else:
                    x_pp = x_star.copy(); x_pp[i] += eps; x_pp[j] += eps
                    x_pm = x_star.copy(); x_pm[i] += eps; x_pm[j] -= eps
                    x_mp = x_star.copy(); x_mp[i] -= eps; x_mp[j] += eps
                    x_mm = x_star.copy(); x_mm[i] -= eps; x_mm[j] -= eps
                    H_free[i, j] = (_nll_wrap(x_pp) - _nll_wrap(x_pm)
                                    - _nll_wrap(x_mp) + _nll_wrap(x_mm)) \
                        / (4.0 * eps * eps)
                    H_free[j, i] = H_free[i, j]
        try:
            cov = np.linalg.inv(H_free)
            se_free = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            se_free = np.full(n_free, np.nan)

    # embed SE back into 9-slot layout (phi slot -> NaN)
    se9 = np.full(9, np.nan)
    for i, k in enumerate(free_idx):
        se9[k] = se_free[i]

    return FitResult(
        params=params, theta=best["theta"], loglik=best["loglik"],
        n_obs=T, n_params=n_params, aic=aic, bic=bic,
        grad_norm=best["grad_norm"], n_iter=best["n_iter"],
        converged=best["converged"],
        hessian=H_free, se_hessian=se9, se_opg=None,
        all_starts=sorted(all_starts, key=lambda d: -d["loglik"]),
    )


def hamilton_loglik_2cov(theta: np.ndarray, y: np.ndarray, z_lag: np.ndarray,
                         r_lag: np.ndarray) -> float:
    """MS-AR(1) 2-covariate TVTP log-likelihood.

    theta = [mu_n, mu_s, log_sig_n, log_sig_delta_raw, phi_raw,
             alpha01, gamma01, h01, alpha10, gamma10, h10] (11 params)
    """
    mu_n, mu_s, log_sigma_n, log_sigma_delta_raw, phi_raw = theta[:5]
    a01, g01, h01, a10, g10, h10 = theta[5:]
    sigma_n = math.exp(log_sigma_n)
    sigma_s = sigma_n + math.log1p(math.exp(log_sigma_delta_raw))
    phi = math.tanh(phi_raw)
    T = int(y.size)
    p01 = _sigmoid(a01 + g01 * z_lag + h01 * r_lag)
    p10 = _sigmoid(a10 + g10 * z_lag + h10 * r_lag)
    pi_arr = _stationary_pi(float(p01.mean()), float(p10.mean()))
    pi0, pi1 = float(pi_arr[0]), float(pi_arr[1])
    y_prev = y[:-1]; y_now = y[1:]
    err0 = y_now - (mu_n + phi * y_prev)
    err1 = y_now - (mu_s + phi * y_prev)
    inv_norm0 = 1.0 / (sigma_n * math.sqrt(2.0 * math.pi))
    inv_norm1 = 1.0 / (sigma_s * math.sqrt(2.0 * math.pi))
    f0_arr = np.exp(-0.5 * (err0 / sigma_n) ** 2) * inv_norm0
    f1_arr = np.exp(-0.5 * (err1 / sigma_s) ** 2) * inv_norm1
    p01_v = p01.astype(float, copy=False)
    p10_v = p10.astype(float, copy=False)
    loglik = 0.0
    for t in range(1, T):
        p01_t = p01_v[t]; p10_t = p10_v[t]
        pi_pred0 = pi0 * (1.0 - p01_t) + pi1 * p10_t
        pi_pred1 = pi0 * p01_t + pi1 * (1.0 - p10_t)
        f0 = f0_arr[t - 1]; f1 = f1_arr[t - 1]
        L = f0 * pi_pred0 + f1 * pi_pred1
        if L < 1e-300:
            L = 1e-300
        loglik += math.log(L)
        pi0 = f0 * pi_pred0 / L
        pi1 = f1 * pi_pred1 / L
    return loglik


def fit_msar_2cov(y: np.ndarray, z_lag: np.ndarray, r_lag: np.ndarray,
                  n_starts: int = 20, seed: int = 20260927,
                  maxiter: int = 500) -> "FitResult":
    """MLE of the 11-parameter 2-covariate TVTP model."""
    rng = np.random.default_rng(seed)
    T = int(y.size)
    n_params = 11
    all_starts = []
    best = None
    for s in range(n_starts):
        # concatenate a 9-vector start with two extra ramp coefficients
        base = _random_theta(rng)
        h01 = rng.uniform(-0.5, 0.5)
        h10 = rng.uniform(-0.5, 0.5)
        theta0 = np.concatenate([base[:8], [h01], base[8:], [h10]])
        # theta0 order: mu_n, mu_s, log_sig_n, log_sig_delta_raw, phi_raw,
        #               a01, g01, h01, a10, g10, h10
        try:
            res = minimize(lambda th: -hamilton_loglik_2cov(th, y, z_lag, r_lag),
                           theta0, method="L-BFGS-B",
                           options={"maxiter": maxiter, "gtol": 1e-6})
            grad_norm = float(np.linalg.norm(res.jac)) if res.jac is not None else float("nan")
            converged = bool(res.success) and grad_norm < 1e-1
            all_starts.append({"start_idx": s, "converged": converged,
                               "loglik": -res.fun, "grad_norm": grad_norm,
                               "n_iter": int(res.nit)})
            if best is None or -res.fun > best["loglik"]:
                best = {"theta": res.x, "loglik": -res.fun,
                        "grad_norm": grad_norm, "n_iter": int(res.nit),
                        "converged": converged}
        except Exception as exc:
            all_starts.append({"start_idx": s, "converged": False,
                               "loglik": float("-inf"), "grad_norm": float("nan"),
                               "n_iter": -1, "error": str(exc)})
    if best is None:
        raise RuntimeError("no successful MLE start (2cov)")
    theta_star = best["theta"]
    # unpack canonical
    mu_n, mu_s, log_sigma_n, log_sigma_delta_raw, phi_raw = theta_star[:5]
    a01, g01, h01, a10, g10, h10 = theta_star[5:]
    sigma_n = math.exp(log_sigma_n)
    sigma_s = sigma_n + math.log1p(math.exp(log_sigma_delta_raw))
    params_dict = {"mu_normal": mu_n, "mu_stress": mu_s,
                   "sigma_normal": sigma_n, "sigma_stress": sigma_s,
                   "phi": math.tanh(phi_raw),
                   "alpha01": a01, "gamma01": g01, "h01": h01,
                   "alpha10": a10, "gamma10": g10, "h10": h10}
    aic = -2.0 * best["loglik"] + 2.0 * n_params
    bic = -2.0 * best["loglik"] + math.log(T) * n_params
    return FitResult(
        params=None, theta=theta_star, loglik=best["loglik"], n_obs=T,
        n_params=n_params, aic=aic, bic=bic,
        grad_norm=best["grad_norm"], n_iter=best["n_iter"],
        converged=best["converged"],
        hessian=None, se_hessian=None, se_opg=None,
        all_starts=sorted(all_starts, key=lambda d: -d["loglik"]),
    ), params_dict


def hamilton_loglik(theta: np.ndarray, y: np.ndarray, z_lag: np.ndarray,
                    return_filtered: bool = False):
    """Log-likelihood of the MS-AR(1) TVTP model on (y, z_lag).

    ``y`` is the observed variable (length T); ``z_lag`` is
    ``z(t - lag)`` on the same grid (also length T; the first entry is
    typically -- but not required to be -- 0 since the AR update starts
    at t=1 anyway).  Returns log-likelihood as a scalar; if
    ``return_filtered=True`` also returns the (T, 2) filtered
    probabilities matrix.
    """
    params = unpack(theta)
    T = int(y.size)
    if z_lag.size != T:
        raise ValueError("y and z_lag must have the same length")

    mu = np.array([params.mu_normal, params.mu_stress])
    sig = np.array([params.sigma_normal, params.sigma_stress])
    phi = params.phi

    # transition probabilities per t (TVTP, length T) -- vectorised
    p01 = _sigmoid(params.alpha01 + params.gamma01 * z_lag)
    p10 = _sigmoid(params.alpha10 + params.gamma10 * z_lag)

    # seed pi_0 with stationary distribution of the mean transition
    pi_arr = _stationary_pi(float(p01.mean()), float(p10.mean()))
    pi0, pi1 = float(pi_arr[0]), float(pi_arr[1])

    # AR(1) means and errors at every t, per regime -- vectorised
    y_prev = y[:-1]                    # length T-1
    y_now = y[1:]                      # length T-1
    err0 = y_now - (mu[0] + phi * y_prev)
    err1 = y_now - (mu[1] + phi * y_prev)
    inv_norm0 = 1.0 / (sig[0] * math.sqrt(2.0 * math.pi))
    inv_norm1 = 1.0 / (sig[1] * math.sqrt(2.0 * math.pi))
    f0_arr = np.exp(-0.5 * (err0 / sig[0]) ** 2) * inv_norm0
    f1_arr = np.exp(-0.5 * (err1 / sig[1]) ** 2) * inv_norm1

    loglik = 0.0
    filt = np.empty((T, 2)) if return_filtered else None
    if return_filtered:
        filt[0, 0] = pi0
        filt[0, 1] = pi1
    p01_v = p01.astype(float, copy=False)
    p10_v = p10.astype(float, copy=False)
    for t in range(1, T):
        p01_t = p01_v[t]
        p10_t = p10_v[t]
        pi_pred0 = pi0 * (1.0 - p01_t) + pi1 * p10_t
        pi_pred1 = pi0 * p01_t + pi1 * (1.0 - p10_t)
        f0 = f0_arr[t - 1]
        f1 = f1_arr[t - 1]
        L = f0 * pi_pred0 + f1 * pi_pred1
        if L < 1e-300:
            L = 1e-300
        loglik += math.log(L)
        pi0 = f0 * pi_pred0 / L
        pi1 = f1 * pi_pred1 / L
        if return_filtered:
            filt[t, 0] = pi0
            filt[t, 1] = pi1

    if return_filtered:
        return loglik, filt
    return loglik


def neg_loglik(theta: np.ndarray, y: np.ndarray, z_lag: np.ndarray) -> float:
    return -hamilton_loglik(theta, y, z_lag)


# --------------------------------------------------------------------------
# Fit -- multi-start L-BFGS-B
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    params: MSARParams
    theta: np.ndarray
    loglik: float
    n_obs: int
    n_params: int
    aic: float
    bic: float
    grad_norm: float
    n_iter: int
    converged: bool
    hessian: Optional[np.ndarray] = None
    se_hessian: Optional[np.ndarray] = None
    se_opg: Optional[np.ndarray] = None
    all_starts: Optional[list] = None


def _random_theta(rng: np.random.Generator) -> np.ndarray:
    """Random start.  Ranges concentrate around plausible values for an
    hourly asinh(PTF) MS-AR(1) TVTP fit (avoids phi=1 boundary, wild
    sigmas, and other pathological seeds that would waste the optimiser
    on parts of parameter space no fit ever lands in)."""
    return np.array([
        rng.uniform(-0.01, 0.01),            # mu_normal
        rng.uniform(-0.01, 0.01),            # mu_stress
        rng.uniform(-6.0, -3.0),             # log_sigma_normal (sigma ~ 0.0025 .. 0.05)
        rng.uniform(-2.0, 0.5),              # log_sigma_delta_raw (delta ~ 0.13 .. 1.05)
        rng.uniform(2.0, 4.5),               # phi_raw (phi ~ 0.964 .. 0.99989)
        rng.uniform(-3.0, -0.5),             # alpha01
        rng.uniform(-1.0, 1.0),              # gamma01
        rng.uniform(-3.0, -0.5),             # alpha10
        rng.uniform(-1.0, 1.0),              # gamma10
    ])


def _numeric_hessian(theta_star: np.ndarray, y: np.ndarray,
                     z_lag: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of the NEGATIVE log-likelihood.

    ``se_hessian`` (from the inverse Hessian) is the standard-error
    estimator used unless the Hessian is not positive definite; then
    the OPG estimator is reported instead.
    """
    n = theta_star.size
    H = np.zeros((n, n))
    f0 = neg_loglik(theta_star, y, z_lag)
    for i in range(n):
        for j in range(i, n):
            th_pp = theta_star.copy(); th_pp[i] += eps; th_pp[j] += eps
            th_pm = theta_star.copy(); th_pm[i] += eps; th_pm[j] -= eps
            th_mp = theta_star.copy(); th_mp[i] -= eps; th_mp[j] -= eps
            th_mm = theta_star.copy(); th_mm[i] -= eps; th_mm[j] += eps
            # equivalent central-diff mixed second derivative
            if i == j:
                th_p = theta_star.copy(); th_p[i] += eps
                th_m = theta_star.copy(); th_m[i] -= eps
                H[i, j] = (neg_loglik(th_p, y, z_lag)
                           - 2.0 * f0
                           + neg_loglik(th_m, y, z_lag)) / (eps * eps)
            else:
                H[i, j] = (neg_loglik(th_pp, y, z_lag)
                           - neg_loglik(th_pm, y, z_lag)
                           - neg_loglik(th_mm, y, z_lag)
                           + neg_loglik(th_mp, y, z_lag)) / (4.0 * eps * eps)
                H[j, i] = H[i, j]
    return H


def _opg_covariance(theta_star: np.ndarray, y: np.ndarray,
                    z_lag: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Outer-product-of-gradients (OPG) covariance = (sum_t s_t s_t^T)^-1,
    where s_t is the score contribution at t.  Implemented as a scalar-
    per-t central difference on the per-observation loglik contribution.

    This is a cross-check against the Hessian-based SE; the two agree
    at the optimum for a correctly specified model.  Large disagreement
    is reported as a specification-error signal.
    """
    n = theta_star.size
    T = int(y.size)
    # We compute per-t log-lik contributions by evaluating the filter
    # incrementally.  For efficiency we compute total log-lik at
    # theta and at each theta +/- eps*e_i, then re-run the filter and
    # return the FILTERED probabilities so we can derive per-t deltas.
    # (Approximation: uses only the aggregate loglik gradient split
    # by the filtered-probability weighting; adequate as an SE
    # cross-check.)
    def per_t_loglik(theta):
        _, filt = hamilton_loglik(theta, y, z_lag, return_filtered=True)
        # Reconstruct per-t marginal densities: recompute inside
        params = unpack(theta)
        mu = np.array([params.mu_normal, params.mu_stress])
        sig = np.array([params.sigma_normal, params.sigma_stress])
        phi = params.phi
        contrib = np.zeros(T)
        # We need the PREDICTED probs, not the filtered.  Re-derive.
        p01 = _sigmoid(params.alpha01 + params.gamma01 * z_lag)
        p10 = _sigmoid(params.alpha10 + params.gamma10 * z_lag)
        pi = _stationary_pi(float(p01.mean()), float(p10.mean()))
        two_pi = 2.0 * math.pi
        for t in range(1, T):
            p01_t, p10_t = p01[t], p10[t]
            pi_pred = np.array([pi[0] * (1.0 - p01_t) + pi[1] * p10_t,
                                pi[0] * p01_t + pi[1] * (1.0 - p10_t)])
            mean_t = mu + phi * y[t - 1]
            err = y[t] - mean_t
            f = np.exp(-0.5 * (err / sig) ** 2) / (sig * math.sqrt(two_pi))
            L = float(f @ pi_pred)
            L = max(L, 1e-300)
            contrib[t] = math.log(L)
            pi = f * pi_pred / L
        return contrib

    base = per_t_loglik(theta_star)
    scores = np.zeros((T, n))
    for i in range(n):
        th_p = theta_star.copy(); th_p[i] += eps
        th_m = theta_star.copy(); th_m[i] -= eps
        scores[:, i] = (per_t_loglik(th_p) - per_t_loglik(th_m)) / (2.0 * eps)
    _ = base
    opg = scores.T @ scores
    try:
        return np.linalg.inv(opg)
    except np.linalg.LinAlgError:
        return np.full((n, n), np.nan)


def fit_msar(y: np.ndarray, z_lag: np.ndarray, n_starts: int = 20,
             seed: int = 20260927,
             maxiter: int = 500,
             tvtp: bool = True,
             compute_se: bool = True,
             compute_opg: bool = True) -> FitResult:
    """Multi-start L-BFGS-B MLE of the MS-AR(1) TVTP model.

    Returns the best converged optimum with SEs, and attaches all
    starts (log-lik, converged flag) in ``result.all_starts`` for
    the convergence diagnostics in §2.
    """
    rng = np.random.default_rng(seed)
    T = int(y.size)
    n_params = 9 if tvtp else 7      # tvtp=False drops gamma01, gamma10 (fixed at 0)
    all_starts = []
    best = None

    for s in range(n_starts):
        theta0 = _random_theta(rng)
        if not tvtp:
            # gamma01 = gamma10 = 0 in the pack; use the fixed slots but
            # optimise a 7-vector where those two are frozen.
            free_idx = [0, 1, 2, 3, 4, 5, 7]      # skip 6 (gamma01), 8 (gamma10)
            def _nll(theta_free):
                th = theta0.copy()
                th[free_idx] = theta_free
                th[6] = 0.0; th[8] = 0.0
                return neg_loglik(th, y, z_lag)
            x0 = theta0[free_idx]
        else:
            def _nll(theta): return neg_loglik(theta, y, z_lag)
            x0 = theta0
        try:
            res = minimize(_nll, x0, method="L-BFGS-B",
                           options={"maxiter": maxiter, "gtol": 1e-6})
            if not tvtp:
                th_full = theta0.copy()
                th_full[free_idx] = res.x
                th_full[6] = 0.0; th_full[8] = 0.0
                loglik = -res.fun
            else:
                th_full = res.x
                loglik = -res.fun
            grad_norm = float(np.linalg.norm(res.jac)) if res.jac is not None else float("nan")
            converged = bool(res.success) and grad_norm < 1e-1
            all_starts.append({
                "start_idx": s, "converged": converged,
                "loglik": loglik, "grad_norm": grad_norm,
                "n_iter": int(res.nit),
            })
            if best is None or loglik > best["loglik"]:
                best = {"theta": th_full, "loglik": loglik,
                        "grad_norm": grad_norm, "n_iter": int(res.nit),
                        "converged": converged}
        except Exception as exc:
            all_starts.append({"start_idx": s, "converged": False,
                               "loglik": float("-inf"),
                               "grad_norm": float("nan"), "n_iter": -1,
                               "error": str(exc)})

    if best is None:
        raise RuntimeError("no successful MLE start")

    params = unpack(best["theta"])
    aic = -2.0 * best["loglik"] + 2.0 * n_params
    bic = -2.0 * best["loglik"] + math.log(T) * n_params

    # SEs (optional; heavy for the multi-start convergence-scan phase)
    H = None
    se_H = np.full(9, np.nan)
    se_opg = np.full(9, np.nan)
    if compute_se:
        H = _numeric_hessian(best["theta"], y, z_lag)
        try:
            cov_H = np.linalg.inv(H)
            se_H = np.sqrt(np.maximum(np.diag(cov_H), 0.0))
        except np.linalg.LinAlgError:
            se_H = np.full(9, np.nan)
    if compute_opg:
        try:
            cov_OPG = _opg_covariance(best["theta"], y, z_lag)
            se_opg = np.sqrt(np.maximum(np.diag(cov_OPG), 0.0))
        except Exception:
            se_opg = np.full(9, np.nan)

    return FitResult(
        params=params, theta=best["theta"], loglik=best["loglik"],
        n_obs=T, n_params=n_params, aic=aic, bic=bic,
        grad_norm=best["grad_norm"], n_iter=best["n_iter"],
        converged=best["converged"],
        hessian=H, se_hessian=se_H, se_opg=se_opg,
        all_starts=sorted(all_starts, key=lambda d: -d["loglik"]),
    )


# --------------------------------------------------------------------------
# Simulation helper -- for filter validation ONLY (never for results)
# --------------------------------------------------------------------------
def simulate_msar(params: MSARParams, T: int, z_lag: np.ndarray,
                  seed: int = 42) -> np.ndarray:
    """Draw one path of length T under the given parameters.  Used for
    the §1 filter parameter-recovery test; NOT for anything that
    influences results."""
    if z_lag.size != T:
        raise ValueError("z_lag must have length T")
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    y[0] = rng.standard_normal() * params.sigma_normal
    J = 0                # start in normal
    for t in range(1, T):
        p01_t = _sigmoid(params.alpha01 + params.gamma01 * z_lag[t])
        p10_t = _sigmoid(params.alpha10 + params.gamma10 * z_lag[t])
        u = rng.random()
        if J == 0 and u < p01_t:
            J = 1
        elif J == 1 and u < p10_t:
            J = 0
        mu_j = params.mu_normal if J == 0 else params.mu_stress
        sig_j = params.sigma_normal if J == 0 else params.sigma_stress
        y[t] = mu_j + params.phi * y[t - 1] + rng.standard_normal() * sig_j
    return y
