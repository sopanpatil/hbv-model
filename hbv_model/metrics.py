# metrics.py
"""
Goodness-of-fit metrics for rainfall-runoff model evaluation.

Usage
-----
>>> from rainfallrunoff import metrics
>>> score = metrics.kge(q_obs, q_sim, dates=df['date'], start_date='1990-10-01', end_date='2010-09-30')
"""
import numpy as np
import pandas as pd


def _apply_mask(obs, sim, dates=None, start_date=None, end_date=None):
    obs, sim = np.asarray(obs, dtype=np.float64), np.asarray(sim, dtype=np.float64)
    mask = ~np.isnan(obs)
    if dates is not None:
        dates = pd.to_datetime(dates)
        if start_date is not None:
            mask &= (dates >= pd.Timestamp(start_date))
        if end_date is not None:
            mask &= (dates <= pd.Timestamp(end_date))
    return obs[mask], sim[mask]


def nash_sutcliffe(obs, sim, dates=None, start_date=None, end_date=None):
    """
    Nash-Sutcliffe Efficiency (NSE).

    1 = perfect fit, 0 = no better than the mean of observations,
    can be arbitrarily negative. Returns nan if obs has zero variance
    (denominator would be zero).
    """
    obs, sim = _apply_mask(obs, sim, dates, start_date, end_date)
    if np.std(obs) == 0:
        return np.nan
    num = np.sum((obs - sim) ** 2)
    den = np.sum((obs - np.mean(obs)) ** 2)
    return 1 - num / den


def kge(obs, sim, dates=None, start_date=None, end_date=None):
    """
    Kling-Gupta Efficiency (KGE), Gupta et al. (2009).

    1 = perfect fit. Decomposes into correlation (r), variability
    ratio (alpha) and bias ratio (beta). Returns nan if obs or sim has
    zero variance, or if obs has zero mean (correlation/ratios are
    undefined in these cases) - this happens occasionally during
    calibration when a parameter set produces a constant or all-zero
    simulated series (e.g. stores still filling during the whole
    scoring window); sceua's objective function should treat nan as a
    poor score, not crash or warn.
    """
    obs, sim = _apply_mask(obs, sim, dates, start_date, end_date)
    if np.std(obs) == 0 or np.std(sim) == 0 or np.mean(obs) == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def kge_components(obs, sim, dates=None, start_date=None, end_date=None):
    """
    Return KGE and its three components (r, alpha, beta) as a dict.
    Useful for diagnostics beyond the single combined score. Returns
    nan for all four if obs or sim has zero variance, or obs has zero
    mean (see kge() docstring).
    """
    obs, sim = _apply_mask(obs, sim, dates, start_date, end_date)
    if np.std(obs) == 0 or np.std(sim) == 0 or np.mean(obs) == 0:
        return {'kge': np.nan, 'r': np.nan, 'alpha': np.nan, 'beta': np.nan}
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    score = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {'kge': score, 'r': r, 'alpha': alpha, 'beta': beta}
