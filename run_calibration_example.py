# run_calibration_example.py
"""
This is a demonstration of the calibration procedure only - the
bundled data (example_data/synthetic_catchment.csv) is synthetic, not
a real catchment, and the resulting "calibration" is not physically
meaningful. To calibrate against a real catchment, replace the input
CSV with your own precipitation/temperature/PET/discharge series in
the same column format (see README.md).

Usage
-----
    python run_calibration_example.py
"""
import pandas as pd

from hbv_model.hbv import HBVModel
from hbv_model import calibrate_sceua

DATA_PATH = "example_data/synthetic_catchment.csv"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Loaded {len(df)} days of synthetic catchment data from {DATA_PATH}")

    best_params, best_score = calibrate_sceua(
        model_cls=HBVModel,
        precip=df["precip_mm"].values,
        temp=df["temp_c"].values,
        evap=df["pet_mm"].values,
        q_obs=df["discharge_mm"].values,
        dates=df["date"],
        start_date=df["date"].iloc[0],
        end_date=df["date"].iloc[-1],
        metric="kge",
        n_complexes=5,
        maxiter=3000,   # kept small so the example runs quickly; increase
                        # for a real catchment calibration (e.g. 10000-20000)
        seed=42,
        verbose=True,
    )

    print("\nCalibrated parameters:")
    for name, value in best_params.items():
        print(f"  {name:8s} = {value:.4f}")
    print(f"\nAchieved KGE: {best_score:.4f}")


if __name__ == "__main__":
    main()
