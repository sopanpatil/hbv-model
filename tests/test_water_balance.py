# test_water_balance.py
"""
Water-balance tests for the HBV model.

    P - ETact - Qgen - (S_end - S_start) == 0,   S = SP + WC + SM + UZ + LZ

Run with:  python -m pytest tests/
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hbv_model.hbv import HBVModel  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "example_data", "synthetic_catchment.csv")

BASE = {"TT": 0.0, "CFMAX": 3.0, "CFR": 0.05, "CWH": 0.1, "FC": 50.0, "LP": 0.7, "BETA": 6.0,
        "K0": 0.3, "K1": 0.1, "K2": 0.02, "UZL": 20.0, "PERC": 2.0, "MAXBAS": 1.0}


def balance(params, precip, temp, evap, init=None):
    """Return (water-balance residual in mm, model, Qsim, states)."""
    init = init or {}
    model = HBVModel(params, initial_states=init)
    q, states = model.run(precip, temp, evap)
    s0 = (init.get("SP", 0.0) + init.get("WC", 0.0) + init.get("SM", 0.5 * params["FC"])
          + init.get("UZ", 0.0) + init.get("LZ", 0.0))
    s1 = sum(model.final_states.values())
    resid = np.sum(precip) - states["ETact"].sum() - states["Qgen"].sum() - (s1 - s0)
    return resid, model, q, states


def random_params(rng):
    return {k: rng.uniform(*HBVModel.PARAM_BOUNDS[k]) for k in HBVModel.PARAM_NAMES}


def test_one_day_soil_overflow_is_recharged():
    # 30 mm rain onto SM = 45 of FC = 50 with BETA = 6: recharge = 30 * 0.9**6 = 15.943 mm,
    # so SM would reach 59.057 mm; the 9.057 mm above FC must go to recharge.
    resid, model, _, states = balance(BASE, np.array([30.0]), np.array([10.0]), np.array([0.0]),
                                      init={"SM": 45.0})
    assert abs(resid) < 1e-12
    assert model.final_states["SM"] == BASE["FC"]
    total_recharge = 30.0 - (BASE["FC"] - 45.0)  # everything that did not stay in the soil
    assert np.isclose(model.final_states["UZ"] + model.final_states["LZ"] + states["Qgen"][0],
                      total_recharge)


def test_water_balance_random_params_example_data():
    df = pd.read_csv(DATA)
    p, t, e = (df[c].to_numpy(float) for c in ["precip_mm", "temp_c", "pet_mm"])
    rng = np.random.default_rng(0)
    for _ in range(200):
        resid, *_ = balance(random_params(rng), p, t, e)
        assert abs(resid) < 1e-8 * p.sum()


def test_water_balance_stress_large_inputs_small_fc():
    # Heavy, intermittent rain and snow with FC at its lower bound, so SM often
    # reaches FC within a time step.
    rng = np.random.default_rng(1)
    n = 3000
    p = np.where(rng.random(n) < 0.3, rng.gamma(0.8, 25.0, n), 0.0)
    t = 8.0 + 12.0 * np.sin(2 * np.pi * np.arange(n) / 365.0) + rng.normal(0, 3, n)
    e = np.clip(2.0 + 2.0 * np.sin(2 * np.pi * np.arange(n) / 365.0), 0, None)
    for _ in range(200):
        params = random_params(rng)
        params["FC"] = HBVModel.PARAM_BOUNDS["FC"][0]
        resid, model, _, states = balance(params, p, t, e)
        assert abs(resid) < 1e-8 * p.sum()
        assert states["SM"].max() <= params["FC"]


def test_routing_conserves_volume():
    # Triangular weights sum to one, so routed and generated runoff totals match except
    # for runoff generated in the last MAXBAS days, which is still in transit at the end.
    df = pd.read_csv(DATA)
    p, t, e = (df[c].to_numpy(float) for c in ["precip_mm", "temp_c", "pet_mm"])
    rng = np.random.default_rng(2)
    for _ in range(50):
        params = random_params(rng)
        _, model, q, states = balance(params, p, t, e)
        tail = states["Qgen"][-int(round(params["MAXBAS"])):].sum()
        assert states["Qgen"].sum() - tail - 1e-9 <= q.sum() <= states["Qgen"].sum() + 1e-9
