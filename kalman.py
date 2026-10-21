"""Filtre de Kalman Schwartz-Smith (2000), port du Kalman_Estimation.m fourni.

Convention stricte : 7 paramètres structurels + N racines de variances
mesurées ; R = diag(psi[7:]**2). P0 reste fixe pendant l'optimisation.
"""
from __future__ import annotations

import numpy as np


def matrices(psi: np.ndarray, maturities: np.ndarray, dt: float = 1 / 252):
    k, sx, lx, mu, se, le, rho = np.asarray(psi, dtype=float)[:7]
    tau = np.asarray(maturities, dtype=float)
    one = -np.expm1(-k * tau)
    c = np.array([0.0, mu * dt])
    A = np.array([[np.exp(-k * dt), 0.0], [0.0, 1.0]])
    Q = np.array([
        [-np.expm1(-2 * k * dt) * sx**2 / (2 * k),
         -np.expm1(-k * dt) * rho * sx * se / k],
        [-np.expm1(-k * dt) * rho * sx * se / k, se**2 * dt],
    ])
    p1 = -np.expm1(-2 * k * tau) * sx**2 / (2 * k)
    p2 = se**2 * tau
    p3 = 2 * one * rho * sx * se / k
    d = (mu - le) * tau - one * lx / k + 0.5 * (p1 + p2 + p3)
    H = np.column_stack((np.exp(-k * tau), np.ones_like(tau)))
    return A, c, Q, d, H


def kalman(y: np.ndarray, psi: np.ndarray, maturities: np.ndarray,
           x0: np.ndarray, P0: np.ndarray, dt: float = 1 / 252,
           details: bool = False):
    """Retourne NLL ou (NLL, traces) quand details=True.

    Même prédiction/correction et covariance non-Joseph que le MATLAB.
    Pseudo-inverse MATLAB reproduite avec un seuil lié à la valeur singulière max.
    """
    y, psi = np.asarray(y, float), np.asarray(psi, float)
    nobs, N = y.shape
    if len(psi) != 7 + N:
        raise ValueError(f"{7 + N} paramètres attendus ; reçus : {len(psi)}")
    if (not np.isfinite(psi).all() or not np.isfinite(y).all()
            or psi[0] <= 0 or psi[1] <= 0 or psi[4] <= 0):
        return (np.inf, None) if details else np.inf
    A, c, Q, d, H = matrices(psi, maturities, dt)
    R = np.diag(psi[7:]**2)
    x, P = np.asarray(x0, float).copy(), np.asarray(P0, float).copy()
    traces = None
    if details:
        traces = {"states": np.empty((nobs, 2)),
                  "fit_errors": np.empty_like(y),
                  "innovations": np.empty_like(y),
                  "predictions": np.empty_like(y),
                  "covariances": np.empty((nobs, 4))}
    logdet_sum = 0.0
    quadratic_sum = 0.0
    for t in range(nobs):
        x_pred = A @ x + c
        P_pred = A @ P @ A.T + Q
        z_pred = H @ x_pred + d
        innovation = y[t] - z_pred
        F = H @ P_pred @ H.T + R
        sign, logdet = np.linalg.slogdet(F)
        if sign <= 0 or not np.isfinite(logdet):
            return (np.inf, None) if details else np.inf
        # np.linalg.pinv uses rtol; MATLAB pinv uses max(size(F))*eps*largest singular.
        inverse = np.linalg.pinv(F, rcond=N * np.finfo(float).eps)
        quadratic = float(innovation @ inverse @ innovation)
        K = P_pred @ H.T @ inverse
        x = x_pred + K @ innovation
        P = P_pred - K @ H @ P_pred
        logdet_sum += logdet
        quadratic_sum += quadratic
        if details:
            traces["states"][t] = x
            traces["fit_errors"][t] = y[t] - (H @ x + d)
            traces["innovations"][t] = innovation
            traces["predictions"][t] = z_pred
            traces["covariances"][t] = P.ravel()
    nll = 0.5 * (N * nobs * np.log(2 * np.pi) + logdet_sum + quadratic_sum)
    return (float(nll), traces) if details else float(nll)


def _ou_integral(rate: float, duration: np.ndarray | float):
    """(1-exp(-rate*t))/rate, y compris pour rate -> 0."""
    t = np.asarray(duration, dtype=float)
    if abs(rate) < 1e-9:
        return t * (1 - rate*t/2 + (rate*t)**2/6)
    return -np.expm1(-rate*t) / rate


def matrices_ssx(psi: np.ndarray, maturities: np.ndarray, dt: float = 1 / 252):
    """Extension SSX: two correlated OU factors under P, futures priced under Q.

    P: dX=-k X dt+sx dWx; dE=(mu-gamma E)dt+se dWe.
    Q: dX=(-k X-lx)dt+sx dWx; dE=(mu-gamma E-le)dt+se dWe.
    corr(dWx,dWe)=rho. Observation: log F=H*x+d.
    """
    k, sx, lx, mu, se, le, rho, gamma = np.asarray(psi, float)[:8]
    tau = np.asarray(maturities, float)
    e1, e2 = np.exp(-k*tau), np.exp(-gamma*tau)
    A = np.diag([np.exp(-k*dt), np.exp(-gamma*dt)])
    c = np.array([0., mu * _ou_integral(gamma, dt)])
    Q = np.array([[sx*sx*_ou_integral(2*k, dt),
                   rho*sx*se*_ou_integral(k+gamma, dt)],
                  [rho*sx*se*_ou_integral(k+gamma, dt),
                   se*se*_ou_integral(2*gamma, dt)]])
    v = sx*sx*_ou_integral(2*k, tau) + se*se*_ou_integral(2*gamma, tau) + 2*rho*sx*se*_ou_integral(k+gamma, tau)
    d = -lx*_ou_integral(k, tau) + (mu-le)*_ou_integral(gamma, tau) + .5*v
    H = np.column_stack([e1, e2])
    return A, c, Q, d, H


def kalman_ssx(y: np.ndarray, psi: np.ndarray, maturities: np.ndarray,
               x0: np.ndarray, P0: np.ndarray, dt: float = 1/252,
               details: bool = False):
    """Exact Gaussian SSX likelihood using one-step-ahead innovations.

    Same predict-then-update timing as the existing SS Python filter.
    Unlike the supplied SSX MATLAB file, the likelihood includes the first
    observation consistently in all three terms.
    """
    y, psi = np.asarray(y, float), np.asarray(psi, float)
    nobs, n = y.shape
    invalid = (len(psi) != 8+n or not np.isfinite(psi).all() or
               not np.isfinite(y).all() or min(psi[0],psi[1],psi[4],psi[7]) <= 0 or
               not (-1 <= psi[6] <= 1))
    if invalid:
        return (np.inf, None) if details else np.inf
    try:
        A,c,Q,d,H = matrices_ssx(psi,maturities,dt)
        R = np.diag(psi[8:]**2)
        x,P = np.array(x0,float).copy(), np.array(P0,float).copy()
        traces = None
        if details:
            traces = {'states':np.empty((nobs,2)), 'innovations':np.empty_like(y),
                      'fit_errors':np.empty_like(y), 'predictions':np.empty_like(y),
                      'covariances':np.empty((nobs,4))}
        loss = 0.
        for t in range(nobs):
            x = A@x+c
            P = A@P@A.T+Q
            P = (P+P.T)/2
            z = H@x + d
            v = y[t]-z
            F = H@P@H.T + R
            F = (F+F.T)/2
            # Cholesky solves are better conditioned than direct inverse/pinv.
            C = np.linalg.cholesky(F)
            whitened = np.linalg.solve(C,v)
            ph = P@H.T
            K = np.linalg.solve(C.T,np.linalg.solve(C,ph.T)).T
            loss += .5*(n*np.log(2*np.pi) + 2*np.log(np.diag(C)).sum() + whitened@whitened)
            x = x + K@v
            I=np.eye(2)
            P = (I-K@H)@P@(I-K@H).T + K@R@K.T  # Joseph PSD update
            P = (P+P.T)/2
            if details:
                traces['states'][t] = x
                traces['innovations'][t] = v
                traces['predictions'][t] = z
                traces['fit_errors'][t] = y[t] - (H@x+d)
                traces['covariances'][t] = P.ravel()
        return (float(loss),traces) if details else float(loss)
    except (np.linalg.LinAlgError,ValueError,FloatingPointError,OverflowError):
        return (np.inf,None) if details else np.inf
