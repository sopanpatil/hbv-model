# calibration.py
"""
Usage
-----
>>> from hbv_model.hbv import HBVModel
>>> from hbv_model import calibrate_sceua
>>>
>>> best_params, best_score = calibrate_sceua(
...     model_cls=HBVModel,
...     precip=precip, temp=temp, evap=evap,
...     q_obs=q_obs,
...     dates=df['date'],
...     start_date='1990-10-01',
...     end_date='2010-09-30',
...     n_complexes=7,
...     maxiter=15000,
...     seed=42,
...     verbose=True,
... )
"""
import numpy as np
import pandas as pd

from . import metrics as metrics_module
from .sceua import sceua


def calibrate_sceua(model_cls, precip, temp, evap, q_obs, dates=None,
                     start_date=None, end_date=None, metric='kge',
                     n_complexes=5, n_evolution_steps=None,
                     maxiter=10000, seed=None, tol=1e-6, verbose=True):
    """
    Calibrate a model against observed discharge using SCE-UA.

    Parameters
    ----------
    model_cls : type
        A model class from hbv_model.hbv, i.e. HBVModel. Must define
        PARAM_NAMES, PARAM_BOUNDS,
        and a run(precip, temp, evap) method.
    precip, temp, evap : array_like
        Forcing arrays, same length as q_obs. Passed to every model the
        same way - HBV uses all three (temp drives the snow routine). Should be the full-length series (not
        pre-sliced to start_date/end_date): the model runs over the
        whole series so any warm-up period before start_date still
        drives the model states, and masking to start_date/end_date is
        applied only when scoring.
    q_obs : array_like
        Observed discharge, same length as the forcing arrays.
    dates : array_like, optional
        Dates corresponding to q_obs/forcing, used with start_date/
        end_date to restrict the scoring period.
    start_date, end_date : str or Timestamp, optional
        Restrict the period used for scoring (e.g. exclude a warm-up
        period). The model still runs over the full forcing series.
    metric : str
        'kge' or 'nse'. Defaults to 'kge'.
    n_complexes, n_evolution_steps, maxiter, seed, tol, verbose :
        Passed through to sceua().

    Returns
    -------
    best_params : dict
        Calibrated parameter dict, keyed by model_cls.PARAM_NAMES.
    best_score : float
        The metric value achieved (higher is better, e.g. KGE/NSE).
    """
    if metric == 'kge':
        metric_fn = metrics_module.kge
    elif metric == 'nse':
        metric_fn = metrics_module.nash_sutcliffe
    else:
        raise ValueError(f"Unknown metric '{metric}', expected 'kge' or 'nse'")

    q_obs = np.asarray(q_obs, dtype=np.float64)

    # ---- Pre-compute mask ONCE - avoids recomputing inside every objective call ----
    mask = ~np.isnan(q_obs)
    if dates is not None:
        dates = pd.to_datetime(dates)
        if start_date is not None:
            mask &= (dates >= pd.Timestamp(start_date))
        if end_date is not None:
            mask &= (dates <= pd.Timestamp(end_date))
    q_obs_masked = q_obs[mask]

    bounds = [model_cls.PARAM_BOUNDS[name] for name in model_cls.PARAM_NAMES]

    def objective(param_values):
        params = dict(zip(model_cls.PARAM_NAMES, param_values))
        model = model_cls(params)
        Q_sim, _ = model.run(precip, temp, evap)
        sim = Q_sim[mask]
        score = metric_fn(q_obs_masked, sim)
        return -score if np.isfinite(score) else 1e6

    best_param_array, best_neg_score = sceua(
        objective, bounds,
        n_complexes=n_complexes, n_evolution_steps=n_evolution_steps,
        maxiter=maxiter, seed=seed, tol=tol, verbose=verbose,
    )

    best_params = dict(zip(model_cls.PARAM_NAMES, best_param_array))
    return best_params, -best_neg_score
