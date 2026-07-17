# hbv-model

A standalone implementation of the HBV rainfall-runoff model (Bergström, 1992; Lindström et al., 1997), with calibration via the Shuffled Complex Evolution algorithm (SCE-UA; Duan et al., 1994) using the Kling-Gupta Efficiency (Gupta et al., 2009) as the calibration objective.

## Contents

- `hbv_model/hbv.py` — the HBV model (spatially lumped, single elevation-zone version): snow accumulation/melt, soil moisture, and two groundwater response reservoirs (upper and lower zone), routed via a triangular weighting function (MAXBAS).
- `hbv_model/metrics.py` — Kling-Gupta Efficiency (KGE) and related performance metrics.
- `hbv_model/sceua.py` — the Shuffled Complex Evolution (SCE-UA) optimiser.
- `hbv_model/calibration.py` — wraps the model and metric into an objective function and calls `sceua`.
- `run_calibration_example.py` — a runnable example calibrating HBV against a single synthetic catchment.
- `example_data/synthetic_catchment.csv` — a small, synthetic (not real) daily precipitation/temperature/PET/discharge time series, included purely so the example script runs immediately without requiring external data.
- `calibrated_parameters.csv` — calibrated HBV parameter sets for 671 CAMELS-GB v2 catchments, one row per catchment. Columns: `gauge_id`; the 13 HBV parameters (`TT`, `CFMAX`, `CFR`, `CWH`, `FC`, `LP`, `BETA`, `K0`, `K1`, `K2`, `UZL`, `PERC`, `MAXBAS`); `calibration_kge` (KGE achieved over the calibration period, WY1990–2009); `validation_kge` (KGE achieved over the independent validation period, WY2010–2019); and `used_in_analysis` (boolean, `True` where `validation_kge >= 0.5`; 621 of 671 catchments are `True`, the remaining 50 either fell below this threshold or have missing validation KGE).

## Using your own data

The example script and synthetic dataset are provided only to demonstrate the calibration procedure end-to-end. To calibrate against real catchments, replace `example_data/synthetic_catchment.csv` with your own daily precipitation, temperature, PET, and observed discharge series in the same column format.

The catchment data this model was originally applied to (CAMELS-GB v2) is publicly available from the NERC Environmental Data Service and is not redistributed here; see Coxon et al. (2026) for access.

## Installation

```bash
pip install numpy pandas numba scipy
```

## Usage

```bash
python run_calibration_example.py
```

This calibrates HBV against the bundled synthetic catchment and prints the calibrated parameters and achieved KGE.

## Citation

If you use this code, please cite the archived release (see `CITATION.cff`).

## References

- Bergström, S. (1992). The HBV model – Its structure and applications. SMHI Reports RH, No. 4.
- Duan, Q., Sorooshian, S., & Gupta, V. K. (1994). Optimal use of the SCE-UA global optimization method for calibrating watershed models. Journal of Hydrology, 158, 265–284.
- Coxon, G., Zheng, Y., Barbedo, R., Cooper, H., Fileni, F., Fowler, H. J., Fry, M., Green, A., Gribbin, T., Harfoot, H., Lewis, E., Neto, G. G. R., Qiu, X., Salwey, S., & Wendt, D. E. (2026). CAMELS-GB v2: hydrometeorological time series and landscape attributes for 671 catchments in Great Britain. Earth System Science Data, 18, 4345–4371.
- Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria. Journal of Hydrology, 377(1–2), 80–91.
- Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997). Development and test of the distributed HBV-96 hydrological model. Journal of Hydrology, 201, 272–288.
