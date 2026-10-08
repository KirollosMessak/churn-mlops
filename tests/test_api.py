import pytest
from fastapi.testclient import TestClient

from churn.api import Customer, app

EXAMPLE = Customer.model_config["json_schema_extra"]["examples"][0]


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setenv("PREDICTION_LOG", str(tmp_path_factory.mktemp("logs") / "predictions.jsonl"))
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_probability_and_action(client):
    body = client.post("/predict", json=EXAMPLE).json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["recommended_action"]


def test_risky_customer_scores_higher_than_loyal_customer(client):
    loyal = dict(
        EXAMPLE,
        tenure=70,
        Contract="Two year",
        PaymentMethod="Credit card (automatic)",
        InternetService="DSL",
        MonthlyCharges=55.0,
        TotalCharges=3850.0,
    )
    risky = client.post("/predict", json=EXAMPLE).json()["churn_probability"]
    safe = client.post("/predict", json=loyal).json()["churn_probability"]
    assert risky > safe


def test_invalid_input_is_rejected_with_field_names(client):
    resp = client.post("/predict", json=dict(EXAMPLE, Contract="Weekly", tenure=-1))
    assert resp.status_code == 422
    fields = {e["loc"][-1] for e in resp.json()["detail"]}
    assert {"Contract", "tenure"} <= fields


def test_batch_and_prediction_logging(client):
    resp = client.post("/predict/batch", json={"customers": [EXAMPLE] * 3}).json()
    assert len(resp["predictions"]) == 3
    log = client.app.state.log_path.read_text().splitlines()
    assert len(log) >= 3 and '"features"' in log[-1]


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["test_metrics"]["roc_auc"] > 0.8
