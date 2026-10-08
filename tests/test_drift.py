import numpy as np
import pandas as pd

from churn.config import load_config
from churn.monitoring.drift import analyze, psi
from churn.monitoring.simulate import make_traffic

cfg = load_config()


def test_psi_is_zero_for_identical_distributions_and_large_for_shifted():
    assert psi(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0
    assert psi(np.array([0.9, 0.1]), np.array([0.2, 0.8])) > 0.2


def _reference():
    ref = pd.read_csv("models/champion/reference.csv")
    ref["SeniorCitizen"] = ref["SeniorCitizen"].astype(str)
    return ref


def test_no_drift_on_normal_traffic():
    report = analyze(_reference(), make_traffic("normal", 500), cfg)
    assert not report["retrain_recommended"]


def test_drift_detected_on_shifted_traffic():
    report = analyze(_reference(), make_traffic("drift", 500, seed=1), cfg)
    drifted = {f["feature"] for f in report["features"] if f["drifted"]}
    assert report["retrain_recommended"]
    assert {"MonthlyCharges", "Contract", "tenure"} <= drifted
