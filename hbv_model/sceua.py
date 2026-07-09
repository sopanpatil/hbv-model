# sceua.py
import numpy as np


def sceua(objective, bounds, args=(), n_complexes=5, n_evolution_steps=None,
          maxiter=10000, seed=None, tol=1e-6, verbose=True):
    """
    Shuffled Complex Evolution (SCE-UA) optimiser (minimization).

    Based on Duan, Sorooshian & Gupta (1992),
    "Effective and Efficient Global Optimization for Conceptual
    Rainfall-Runoff Models", Water Resources Research.

    Parameters
    ----------
    objective : callable
        Function to minimize, takes a 1D parameter array followed by
        any additional arguments passed via `args`.
    bounds : list of (lo, hi) tuples
        Parameter bounds, one per dimension.
    args : tuple, optional
        Extra arguments passed to objective after the parameter vector,
        e.g. args=(precip, evap, q_obs, dates, startdate, enddate).
    n_complexes : int
        Number of complexes (p). More complexes = more thorough but slower.
        Typically 2-10; 5 is a good default for 7-13 parameters.
    n_evolution_steps : int or None
        Number of evolution steps per complex per shuffle (q).
        Defaults to 2*n_dims + 1 if None.
    maxiter : int
        Maximum total function evaluations.
    seed : int or None
        Random seed for reproducibility.
    tol : float
        Stop early if improvement over last shuffle falls below this.
    verbose : bool
        Print progress every shuffle.

    Returns
    -------
    best_params : ndarray
        Best parameter vector found.
    best_score : float
        Best (minimized) objective value.
    """
    rng    = np.random.default_rng(seed)
    n_dims = len(bounds)
    lo     = np.array([b[0] for b in bounds])
    hi     = np.array([b[1] for b in bounds])

    p = n_complexes
    m = 2 * n_dims + 1
    q = n_evolution_steps or (2 * n_dims + 1)
    s = p * m

    # ---- 1. Initialise population ----
    population = rng.uniform(lo, hi, size=(s, n_dims))
    scores     = np.array([objective(pt, *args) for pt in population])
    n_evals    = s

    idx        = np.argsort(scores)
    population = population[idx]
    scores     = scores[idx]

    if verbose:
        print(f"SCE-UA | complexes={p}, points/complex={m}, dims={n_dims}")
        print(f"  Initial best: {-scores[0]:.4f}  (evals: {n_evals})")

    prev_best = scores[0]

    # ---- 2. Main loop ----
    shuffle = 0
    while n_evals < maxiter:
        shuffle += 1

        # ---- 3. Partition into complexes ----
        complexes = []
        for k in range(p):
            idx_k   = np.arange(k, s, p)
            cx_pts  = population[idx_k].copy()
            cx_scrs = scores[idx_k].copy()
            complexes.append((cx_pts, cx_scrs, idx_k))

        # ---- 4. Evolve each complex (CCE) ----
        for k in range(p):
            cx_pts, cx_scrs, idx_k = complexes[k]

            for _ in range(q):
                cx_pts, cx_scrs = _cce_step(
                    objective, args, cx_pts, cx_scrs, lo, hi, rng
                )
                n_evals += 1

            complexes[k] = (cx_pts, cx_scrs, idx_k)

        # ---- 5. Reassemble and re-sort population ----
        for k in range(p):
            cx_pts, cx_scrs, idx_k = complexes[k]
            population[idx_k] = cx_pts
            scores[idx_k]     = cx_scrs

        idx        = np.argsort(scores)
        population = population[idx]
        scores     = scores[idx]

        if verbose:
            print(f"  Shuffle {shuffle:4d} | Best KGE: {-scores[0]:.4f} "
                  f"| Evals: {n_evals}")

        # ---- 6. Convergence check ----
        improvement = abs(prev_best - scores[0])
        if improvement < tol and shuffle > 10:
            if verbose:
                print(f"  Converged at shuffle {shuffle} "
                      f"(improvement {improvement:.2e} < tol {tol:.2e})")
            break
        prev_best = scores[0]

    return population[0], scores[0]


def _cce_step(objective, args, cx_pts, cx_scrs, lo, hi, rng):
    """
    One step of the Competitive Complex Evolution (CCE) algorithm.
    Evolves the worst point in the complex using a weighted simplex step.
    """
    m      = len(cx_pts)
    n_dims = cx_pts.shape[1]

    idx     = np.argsort(cx_scrs)
    cx_pts  = cx_pts[idx]
    cx_scrs = cx_scrs[idx]

    probs   = np.array([2 * (m + 1 - (i + 1)) / (m * (m + 1))
                         for i in range(m)])
    sub_idx = rng.choice(m, size=min(m, n_dims + 1), replace=False, p=probs)
    sub_idx = np.sort(sub_idx)
    sub_pts  = cx_pts[sub_idx]
    sub_scrs = cx_scrs[sub_idx]

    worst_idx   = -1
    worst_pt    = sub_pts[worst_idx].copy()
    worst_score = sub_scrs[worst_idx]

    centroid = np.mean(sub_pts[:-1], axis=0)

    # ---- Reflection ----
    reflected = np.clip(2.0 * centroid - worst_pt, lo, hi)
    ref_score = objective(reflected, *args)

    if ref_score < worst_score:
        cx_pts[sub_idx[worst_idx]]  = reflected
        cx_scrs[sub_idx[worst_idx]] = ref_score
    else:
        # ---- Contraction ----
        contracted = np.clip(0.5 * (worst_pt + centroid), lo, hi)
        con_score  = objective(contracted, *args)

        if con_score < worst_score:
            cx_pts[sub_idx[worst_idx]]  = contracted
            cx_scrs[sub_idx[worst_idx]] = con_score
        else:
            # ---- Random replacement ----
            cx_pts[sub_idx[worst_idx]]  = rng.uniform(lo, hi)
            cx_scrs[sub_idx[worst_idx]] = objective(
                cx_pts[sub_idx[worst_idx]], *args
            )

    return cx_pts, cx_scrs
