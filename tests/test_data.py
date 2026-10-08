import pandas as pd
import pytest

from churn.data import DataValidationError, clean, load_dataset, load_raw, validate


@pytest.fixture(scope="module")
def df():
    return clean(load_raw())


def test_clean_fills_blank_total_charges_for_new_customers():
    raw = load_raw()
    blanks = raw["TotalCharges"].astype(str).str.strip() == ""
    cleaned = clean(raw)
    assert blanks.sum() > 0
    assert (cleaned.loc[blanks, "TotalCharges"] == 0).all()
    assert (cleaned.loc[blanks, "tenure"] == 0).all()


def test_real_data_passes_validation(df):
    validate(df)


def test_validation_reports_every_problem_at_once(df):
    bad = df.head(50).copy()
    bad.loc[0, "Contract"] = "Weekly"
    bad.loc[1, "tenure"] = -5
    bad.loc[2, "gender"] = None
    with pytest.raises(DataValidationError) as err:
        validate(bad)
    msg = str(err.value)
    assert "Contract" in msg and "tenure" in msg and "gender" in msg


def test_validation_rejects_missing_columns(df):
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate(df.drop(columns=["Contract"]))


def test_split_is_stratified_and_disjoint():
    train, val, test = load_dataset()
    ids = [set(x["customerID"]) for x in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    rates = [x["Churn"].mean() for x in (train, val, test)]
    assert max(rates) - min(rates) < 0.01
    assert sum(len(x) for x in (train, val, test)) == len(pd.read_csv("data/raw/telco_churn.csv"))
