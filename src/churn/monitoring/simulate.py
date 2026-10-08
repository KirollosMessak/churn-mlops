"""Send simulated production traffic through the API so monitoring has something to watch.

Scenarios:
  normal  customers look like the training data (no drift expected)
  drift   the market changed: a price increase on fiber plans and many new
          month-to-month customers paying by electronic check

Usage: python -m churn.monitoring.simulate --scenario drift --n 500
"""

import argparse

import numpy as np
from fastapi.testclient import TestClient

from churn.api import app
from churn.config import load_config
from churn.data import load_dataset


def make_traffic(scenario: str, n: int, seed: int = 0):
    cfg = load_config()
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    _, _, test = load_dataset()  # unseen customers, never used in training
    rng = np.random.default_rng(seed)
    df = test[feats].sample(n, replace=True, random_state=seed).reset_index(drop=True)

    if scenario == "drift":
        # 1) 25% price increase on fiber plans
        fiber = df["InternetService"] == "Fiber optic"
        df.loc[fiber, "MonthlyCharges"] = (df.loc[fiber, "MonthlyCharges"] * 1.25).clip(upper=500).round(2)
        # 2) A promotion brings in many new, short-tenure month-to-month customers
        new = rng.random(n) < 0.45
        df.loc[new, "Contract"] = "Month-to-month"
        df.loc[new, "PaymentMethod"] = "Electronic check"
        df.loc[new, "tenure"] = rng.integers(0, 6, new.sum())
        df["TotalCharges"] = np.minimum(df["TotalCharges"], df["tenure"] * df["MonthlyCharges"]).round(2)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["normal", "drift"], default="normal")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = make_traffic(args.scenario, args.n, args.seed)
    records = df.to_dict("records")
    for r in records:
        r["SeniorCitizen"] = str(r["SeniorCitizen"])
    flagged = 0
    with TestClient(app) as client:  # in-process: same code path as real HTTP requests
        for i in range(0, len(records), 200):
            resp = client.post("/predict/batch", json={"customers": records[i : i + 200]})
            resp.raise_for_status()
            flagged += resp.json()["contacted"]
    print(
        f"Sent {len(records)} '{args.scenario}' requests; {flagged} flagged as likely to churn "
        f"({flagged / len(records):.0%})"
    )


if __name__ == "__main__":
    main()
