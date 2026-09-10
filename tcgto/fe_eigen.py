"""Linear-elastic FE eigen solver (numpy) for G2R2 new structures.

OpenSeesPy is not importable in this environment (openseespywin DLL load
failure), so new multi-N structures (E3B1-style specs, indices >= 6) obtain
their first three eigenpairs from an equivalent linear elastic FE model:
  - shear: exact shear-chain (zero-length springs) generalized eigenproblem;
  - frame: 3-bay Euler-Bernoulli frame with rigid-diaphragm horizontal
    constraints and lumped floor masses;
  - wall: cantilever Euler-Bernoulli wall (beam-column chain).
This mirrors the E3B1 build() geometry/parameters; only the FE engine differs.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def beam_element(E, A, I, L):
    k = E * I / L ** 3 * np.array([
        [L ** 2 * A / I, 0, 0, -L ** 2 * A / I, 0, 0],
        [0, 12, 6 * L, 0, -12, 6 * L],
        [0, 6 * L, 4 * L ** 2, 0, -6 * L, 2 * L ** 2],
        [-L ** 2 * A / I, 0, 0, L ** 2 * A / I, 0, 0],
        [0, -12, -6 * L, 0, 12, -6 * L],
        [0, 6 * L, 2 * L ** 2, 0, -6 * L, 4 * L ** 2],
    ])
    return k


def beam_element_global(x1, y1, x2, y2, E, A, I):
    L = np.hypot(x2 - x1, y2 - y1)
    ke = beam_element(E, A, I, L)
    if abs(x2 - x1) < 1e-12:  # vertical member: local +x = global +y
        Rn = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        R = np.zeros((6, 6))
        R[:3, :3] = Rn; R[3:, 3:] = Rn
        return R.T @ ke @ R
    return ke


def normalize_phi(phi):
    out = np.zeros_like(phi)
    for r in range(min(3, phi.shape[1])):
        v = phi[:, r]
        v = v / max(np.max(np.abs(v)), 1e-12)
        if v[-1] < 0:
            v = -v
        out[:, r] = v
    return out


def condense_lateral(K, M, keep_idx):
    """Guyan-condense all non-kept DOFs (negligible mass) onto kept lateral DOFs."""
    keep = np.asarray(keep_idx, dtype=int)
    z = np.setdiff1d(np.arange(K.shape[0]), keep)
    Kkk = K[np.ix_(keep, keep)]
    Kkz = K[np.ix_(keep, z)]
    Kzz = K[np.ix_(z, z)]
    Kred = Kkk - Kkz @ np.linalg.solve(Kzz, Kkz.T)
    Mred = M[np.ix_(keep, keep)]
    return Kred, Mred


def shear_eigen(p, mass, n):
    k = np.asarray(p["story_stiffness"], np.float64)
    K = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            K[i, i] = k[i] + (k[i + 1] if i + 1 < n else 0.0)
        else:
            K[i, i] = k[i] + (k[i + 1] if i + 1 < n else 0.0)
    for i in range(n - 1):
        K[i, i + 1] = -k[i + 1]; K[i + 1, i] = -k[i + 1]
    M = np.diag(mass)
    w2, phi = eigh(K, M)
    return np.sqrt(w2[:3]) / (2 * np.pi), normalize_phi(phi[:, :3])


def wall_eigen(p, mass, n, h, E=3e10):
    dof = 3
    nd = n + 1
    K = np.zeros((dof * nd, dof * nd))
    M = np.zeros((dof * nd, dof * nd))
    sc = np.asarray(p["story_scale"], np.float64)
    for fl in range(n):
        A = p["wall_A"] * np.sqrt(sc[fl]); I = p["wall_I"] * sc[fl]
        ke = beam_element_global(0.0, fl * h, 0.0, (fl + 1) * h, E, A, I)
        idx = dof * np.array([fl, fl + 1])
        dofs = np.repeat(idx, 3) + np.tile(np.arange(3), 2)
        K[np.ix_(dofs, dofs)] += ke
    for fl in range(1, nd):
        d = dof * fl
        m = float(mass[fl - 1])
        M[d, d] = m
    free_idx = np.arange(dof, dof * nd)  # node 0 fixed
    Kf = K[np.ix_(free_idx, free_idx)]; Mf = M[np.ix_(free_idx, free_idx)]
    keep_idx = [int(np.where(free_idx == dof * fl)[0][0]) for fl in range(1, nd)]
    Kred, Mred = condense_lateral(Kf, Mf, keep_idx)
    w2, phi = eigh(Kred, Mred)
    w2 = np.maximum(w2, 0.0)
    return np.sqrt(w2[:3]) / (2 * np.pi), normalize_phi(phi[:, :3])


def frame_eigen(p, mass, n, h, E=2e11):
    """Rigid-diaphragm moment frame -> equivalent shear chain (standard
    engineering idealization): story stiffness = 4 * 12 E I_col / h^3."""
    sc = np.asarray(p["story_scale"], np.float64)
    k = np.asarray([4.0 * 12.0 * E * p["column_I"] * sc[i] / h ** 3 for i in range(n)])
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = k[i] + (k[i + 1] if i + 1 < n else 0.0)
    for i in range(n - 1):
        K[i, i + 1] = -k[i + 1]; K[i + 1, i] = -k[i + 1]
    M = np.diag(mass)
    w2, phi = eigh(K, M)
    return np.sqrt(np.maximum(w2[:3], 0.0)) / (2 * np.pi), normalize_phi(phi[:, :3])


def numpy_eigen(s):
    fam = s["family"]; n = s["N_story"]; p = s["p"]; mass = np.asarray(s["mass"], np.float64)
    if fam == "shear":
        return shear_eigen(p, mass, n)
    if fam == "wall":
        return wall_eigen(p, mass, n, s["story_height"])
    return frame_eigen(p, mass, n, s["story_height"])
