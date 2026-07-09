# hbv.py
"""
HBV rainfall-runoff model (lumped, single elevation zone).

    PARAM_NAMES   : list[str]            - ordered parameter names
    PARAM_BOUNDS  : dict[str, (lo, hi)]  - calibration bounds per parameter
    run(precip, temp, evap) -> (Qsim, states); sets self.final_states
"""
import numpy as np
import pandas as pd
from numba import njit


# =============================================================================
# HBV
# =============================================================================

@njit(cache=True)
def _hbv_run_numba(precip, temp, evap,
                    TT, CFMAX, CFR, CWH,
                    FC, LP, BETA,
                    K0, K1, K2, UZL, PERC, MAXBAS_int,
                    SP, WC, SM, UZ, LZ):
    n = len(precip)
    Qsim   = np.zeros(n)
    SM_out = np.zeros(n)
    SP_out = np.zeros(n)
    UZ_out = np.zeros(n)
    LZ_out = np.zeros(n)
    ET_out = np.zeros(n)

    for t in range(n):
        P, T, PE = precip[t], temp[t], evap[t]

        if T < TT:
            SP += P
            rain = 0.0
        else:
            rain = P

        if T > TT:
            melt = CFMAX * (T - TT)
            if melt > SP:
                melt = SP
            SP -= melt
            WC += melt
        else:
            refreeze = CFR * CFMAX * (TT - T)
            if refreeze > WC:
                refreeze = WC
            WC -= refreeze
            SP += refreeze

        max_WC = CWH * SP
        if WC > max_WC:
            excess = WC - max_WC
            WC = max_WC
        else:
            excess = 0.0

        in_soil = rain + excess

        if FC > 0.0:
            recharge = in_soil * (SM / FC) ** BETA
        else:
            recharge = in_soil
        if recharge > in_soil:
            recharge = in_soil
        SM += in_soil - recharge
        if SM > FC:
            SM = FC

        if SM > LP * FC:
            ETact = PE
        else:
            ETact = PE * (SM / (LP * FC)) if LP > 0.0 else 0.0
        if ETact > SM:
            ETact = SM
        SM -= ETact
        if SM < 0.0:
            SM = 0.0

        UZ += recharge
        perc = PERC if UZ > PERC else UZ
        UZ -= perc
        LZ += perc

        Q0 = (UZ - UZL) * K0 if UZ > UZL else 0.0
        UZ -= Q0
        Q1 = UZ * K1
        UZ -= Q1
        Q2 = LZ * K2
        LZ -= Q2

        SM_out[t] = SM
        SP_out[t] = SP
        UZ_out[t] = UZ
        LZ_out[t] = LZ
        ET_out[t]  = ETact
        Qsim[t]    = Q0 + Q1 + Q2

    return Qsim, SM_out, SP_out, UZ_out, LZ_out, ET_out, SP, WC, SM, UZ, LZ


@njit(cache=True)
def _triangular_routing_numba(Q, maxbas):
    if maxbas <= 1:
        return Q.copy()
    weights = np.zeros(maxbas)
    half = maxbas / 2.0
    for i in range(maxbas):
        weights[i] = (i + 1) if i < half else (maxbas - i)
    total = 0.0
    for w in weights:
        total += w
    for i in range(maxbas):
        weights[i] /= total
    n = len(Q)
    Q_routed = np.zeros(n)
    for t in range(n):
        for j in range(maxbas):
            if t - j >= 0:
                Q_routed[t] += weights[j] * Q[t - j]
    return Q_routed


class HBVModel:
    """
    HBV hydrological model (lumped, single elevation zone).

    Example
    -------
    >>> params = {
    ...     'TT': 0.0, 'CFMAX': 3.0, 'CFR': 0.05, 'CWH': 0.1,
    ...     'FC': 250.0, 'LP': 0.7, 'BETA': 2.0,
    ...     'K0': 0.3, 'K1': 0.1, 'K2': 0.02,
    ...     'UZL': 20.0, 'PERC': 2.0, 'MAXBAS': 3.0
    ... }
    >>> model = HBVModel(params)
    >>> Q, states = model.run(precip, temp, evap)
    """

    PARAM_NAMES = ['TT', 'CFMAX', 'CFR', 'CWH', 'FC', 'LP', 'BETA',
                   'K0', 'K1', 'K2', 'UZL', 'PERC', 'MAXBAS']

    PARAM_BOUNDS = {
        'TT':     (-2.5, 2.5),
        'CFMAX':  (0.5, 10.0),
        'CFR':    (0.0, 0.1),
        'CWH':    (0.0, 0.2),
        'FC':     (50.0, 500.0),
        'LP':     (0.3, 1.0),
        'BETA':   (1.0, 6.0),
        'K0':     (0.05, 0.5),
        'K1':     (0.01, 0.3),
        'K2':     (0.001, 0.15),
        'UZL':    (0.0, 100.0),
        'PERC':   (0.0, 6.0),
        'MAXBAS': (1.0, 7.0),
    }

    def __init__(self, params=None, initial_states=None):
        self.params = params if params is not None else {}
        self.initial_states = initial_states if initial_states is not None else {}

    def set_params(self, params):
        self.params = params

    def _validate_params(self):
        missing = [p for p in self.PARAM_NAMES if p not in self.params]
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

    def run(self, precip, temp, evap):
        self._validate_params()
        p = self.params
        precip = np.asarray(precip, dtype=np.float64)
        temp   = np.asarray(temp,   dtype=np.float64)
        evap   = np.asarray(evap,   dtype=np.float64)

        FC   = p['FC']
        init = self.initial_states
        SP   = float(init.get('SP', 0.0))
        WC   = float(init.get('WC', 0.0))
        SM   = float(init.get('SM', FC * 0.5))
        UZ   = float(init.get('UZ', 0.0))
        LZ   = float(init.get('LZ', 0.0))

        maxbas_int = int(round(p['MAXBAS']))

        Qsim, SM_out, SP_out, UZ_out, LZ_out, ET_out, \
            SP_f, WC_f, SM_f, UZ_f, LZ_f = _hbv_run_numba(
                precip, temp, evap,
                float(p['TT']),   float(p['CFMAX']), float(p['CFR']),  float(p['CWH']),
                float(p['FC']),   float(p['LP']),    float(p['BETA']),
                float(p['K0']),   float(p['K1']),    float(p['K2']),
                float(p['UZL']),  float(p['PERC']),  maxbas_int,
                SP, WC, SM, UZ, LZ
            )

        self.final_states = {
            'SP': SP_f, 'WC': WC_f, 'SM': SM_f, 'UZ': UZ_f, 'LZ': LZ_f
        }

        Q_routed = _triangular_routing_numba(Qsim, maxbas_int)

        states = {
            'SM': SM_out, 'SP': SP_out, 'UZ': UZ_out,
            'LZ': LZ_out, 'ETact': ET_out, 'Qgen': Qsim
        }

        return Q_routed, states
