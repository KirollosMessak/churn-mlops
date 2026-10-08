"""Prediction API. Serves the champion model exported by the training pipeline.

Run locally:  uvicorn churn.api:app --reload      (docs at http://localhost:8000/docs)
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from churn.config import load_config, project_path

YesNo = Literal["Yes", "No"]
NoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer's account data. Invalid values are rejected with a clear 422 error."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal["0", "1"]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: NoInternet
    OnlineBackup: NoInternet
    DeviceProtection: NoInternet
    TechSupport: NoInternet
    StreamingTV: NoInternet
    StreamingMovies: NoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: float = Field(ge=0, le=60_000)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": "0",
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 2,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 95.5,
                    "TotalCharges": 191.0,
                }
            ]
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool = Field(description="True if probability >= the profit-optimized threshold")
    risk_level: Literal["low", "medium", "high"]
    recommended_action: str
    model_version: int
    request_id: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    contacted: int
    expected_campaign_profit: float


class ModelState:
    """The loaded champion model plus its metadata."""

    def __init__(self, model_dir: Path):
        self.pipeline = joblib.load(model_dir / "model.joblib")
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self.threshold = self.metadata["threshold"]
        self.features = self.metadata["features"]
        self.version = self.metadata["model_version"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    model_dir = Path(os.environ.get("MODEL_DIR", project_path(cfg["paths"]["champion_dir"])))
    app.state.model = ModelState(model_dir)
    app.state.log_path = Path(os.environ.get("PREDICTION_LOG", project_path(cfg["paths"]["prediction_log"])))
    app.state.log_path.parent.mkdir(parents=True, exist_ok=True)
    app.state.business = cfg["business"]
    app.state.stats = {"started_at": time.time(), "predictions": 0, "errors": 0}
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predicts which customers are likely to cancel, and whether a retention offer pays off.",
    lifespan=lifespan,
)


def _risk(p: float, threshold: float) -> tuple[str, str]:
    if p >= max(0.7, threshold):
        return "high", "Contact now with a retention offer"
    if p >= threshold:
        return "medium", "Include in this month's retention campaign"
    return "low", "No action needed"


def _predict(request: Request, customers: list[Customer]) -> list[Prediction]:
    state: ModelState = request.app.state.model
    rows = pd.DataFrame([c.model_dump() for c in customers])[state.features]
    try:
        probas = state.pipeline.predict_proba(rows)[:, 1]
    except Exception as exc:  # never leak a stack trace to the client
        request.app.state.stats["errors"] += 1
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    now = datetime.now(UTC).isoformat(timespec="seconds")
    results = []
    with open(request.app.state.log_path, "a", encoding="utf-8") as log:
        for row, p in zip(rows.to_dict("records"), probas, strict=False):
            p = float(p)
            level, action = _risk(p, state.threshold)
            pred = Prediction(
                churn_probability=round(p, 4),
                will_churn=p >= state.threshold,
                risk_level=level,
                recommended_action=action,
                model_version=state.version,
                request_id=uuid.uuid4().hex[:12],
            )
            # Every prediction is logged with its inputs: this feeds drift monitoring
            log.write(
                json.dumps(
                    {
                        "timestamp": now,
                        "request_id": pred.request_id,
                        "features": row,
                        "churn_probability": pred.churn_probability,
                        "model_version": state.version,
                    }
                )
                + "\n"
            )
            results.append(pred)
    request.app.state.stats["predictions"] += len(results)
    return results


@app.post("/predict", response_model=Prediction, summary="Predict churn for one customer")
def predict(customer: Customer, request: Request):
    return _predict(request, [customer])[0]


@app.post("/predict/batch", response_model=BatchResponse, summary="Score up to 1,000 customers")
def predict_batch(batch: BatchRequest, request: Request):
    preds = _predict(request, batch.customers)
    biz = request.app.state.business
    contacted = [p for p in preds if p.will_churn]
    # Expected value of contacting the flagged customers, using the model's own probabilities
    profit = sum(
        p.churn_probability * biz["offer_success_rate"] * biz["saved_customer_value"] - biz["offer_cost"]
        for p in contacted
    )
    return BatchResponse(predictions=preds, contacted=len(contacted), expected_campaign_profit=round(profit, 2))


@app.get("/health", summary="Liveness/readiness check")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
        "model_version": request.app.state.model.version,
    }


@app.get("/model-info", summary="Which model is serving and how good it is")
def model_info(request: Request):
    return request.app.state.model.metadata


@app.get("/metrics", summary="Basic service metrics")
def metrics(request: Request):
    s = request.app.state.stats
    return {
        "uptime_seconds": round(time.time() - s["started_at"]),
        "predictions_served": s["predictions"],
        "prediction_errors": s["errors"],
        "model_version": request.app.state.model.version,
    }
