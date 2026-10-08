"""Loading, cleaning, validation and splitting of the churn data."""

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import load_config, project_path

# Allowed values per categorical column. Anything else is rejected by validate().
CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}
NUMERIC_RANGES = {"tenure": (0, 120), "MonthlyCharges": (0, 500), "TotalCharges": (0, 60_000)}


class DataValidationError(ValueError):
    """Raised when data fails validation; the message lists every problem found."""


def load_raw(path=None) -> pd.DataFrame:
    cfg = load_config()
    return pd.read_csv(path or project_path(cfg["data"]["raw_path"]))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known quirks of the raw export so it matches the schema."""
    df = df.copy()
    # TotalCharges is text with blanks for brand-new customers (tenure 0): they've paid nothing yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", "0"))
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    if "Churn" in df:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def validate(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Check columns, missing values, categories and numeric ranges. Raise with all problems at once."""
    cfg = load_config()
    expected = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    if require_target:
        expected = expected + [cfg["data"]["target"]]
    problems = []

    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing columns: {missing_cols}")
    if len(df) == 0:
        raise DataValidationError("Dataset is empty")

    nulls = df[expected].isna().sum()
    for col, n in nulls[nulls > 0].items():
        problems.append(f"{col}: {n} missing values")

    for col, allowed in CATEGORIES.items():
        bad = set(df[col].dropna().astype(str)) - set(allowed)
        if bad:
            problems.append(f"{col}: unexpected values {sorted(bad)}")

    for col, (lo, hi) in NUMERIC_RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            problems.append(f"{col}: non-numeric values")
        elif ((values < lo) | (values > hi)).any():
            problems.append(f"{col}: values outside [{lo}, {hi}]")

    if require_target:
        target = cfg["data"]["target"]
        if not set(df[target].unique()) <= {0, 1}:
            problems.append(f"{target}: must be 0/1")
        rate = df[target].mean()
        if not 0.05 <= rate <= 0.6:
            problems.append(f"{target}: churn rate {rate:.1%} looks wrong (expected 5-60%)")

    if problems:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(problems))
    return df


def split(df: pd.DataFrame):
    """Stratified train / validation / test split, reproducible via random_state."""
    cfg = load_config()["data"]
    train_val, test = train_test_split(
        df, test_size=cfg["test_size"], stratify=df[cfg["target"]], random_state=cfg["random_state"]
    )
    train, val = train_test_split(
        train_val,
        test_size=cfg["val_size"],
        stratify=train_val[cfg["target"]],
        random_state=cfg["random_state"],
    )
    return train, val, test


def load_dataset():
    """Load, clean, validate and split. The single entry point used by training."""
    return split(validate(clean(load_raw())))
