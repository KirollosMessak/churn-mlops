"""Model pipelines: shared preprocessing + one of three classifiers."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn.config import load_config


def build_preprocessor() -> ColumnTransformer:
    feats = load_config()["features"]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), feats["numeric"]),
            # handle_unknown="ignore": an unseen category in production won't crash the API
            ("cat", OneHotEncoder(handle_unknown="ignore"), feats["categorical"]),
        ]
    )


def build_model(name: str, params: dict, seed: int) -> Pipeline:
    if name == "logistic_regression":
        clf = LogisticRegression(**params, class_weight="balanced", random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(**params, class_weight="balanced", n_jobs=-1, random_state=seed)
    elif name == "xgboost":
        clf = XGBClassifier(**params, eval_metric="logloss", n_jobs=-1, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", build_preprocessor()), ("model", clf)])
