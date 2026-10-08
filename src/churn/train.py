"""Training pipeline: validate data -> train candidates -> track in MLflow -> quality gate -> promote.

Usage: python -m churn.train
Exit code 1 if the new model fails the quality gate (so CI fails and nothing is deployed).
"""

import json
import os
import sys
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from churn.config import ROOT, load_config, project_path
from churn.data import load_dataset
from churn.evaluation import best_threshold, compute_metrics, plot_evaluation, plot_feature_importance
from churn.features import build_model

CHAMPION_ALIAS = "champion"

# The model outputs 0/1 integers; MLflow's schema hint about integer columns doesn't apply here.
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")


def setup_mlflow(cfg: dict) -> None:
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not os.path.isabs(uri.removeprefix("sqlite:///")):
        uri = f"sqlite:///{(ROOT / uri.removeprefix('sqlite:///')).as_posix()}"
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", uri))
    mlflow.set_experiment(cfg["mlflow"]["experiment"])


def train_candidates(cfg, train, val):
    """Train every model in the config, log each to MLflow, return results sorted by validation AUC."""
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    target, seed, biz = cfg["data"]["target"], cfg["data"]["random_state"], cfg["business"]
    results = []
    for name, params in cfg["models"].items():
        with mlflow.start_run(run_name=name) as run:
            pipeline = build_model(name, params, seed)
            pipeline.fit(train[feats], train[target])
            proba = pipeline.predict_proba(val[feats])[:, 1]
            threshold = best_threshold(val[target], proba, biz)
            metrics = compute_metrics(val[target], proba, threshold, biz)

            mlflow.log_params({"model_type": name, **params})
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()} | {"threshold": threshold})
            # tenure as float so the logged schema tolerates missing values (MLflow integer-column hint)
            example = val[feats].head(3).astype({"tenure": "float64"})
            info = mlflow.sklearn.log_model(pipeline, name="model", input_example=example)
            print(f"  {name:<20} val ROC-AUC {metrics['roc_auc']:.4f}  profit ${metrics['profit']:,.0f}")
            results.append(
                {
                    "name": name,
                    "run_id": run.info.run_id,
                    "model_uri": info.model_uri,
                    "pipeline": pipeline,
                    "threshold": threshold,
                    "val": metrics,
                }
            )
    return sorted(results, key=lambda r: r["val"]["roc_auc"], reverse=True)


def champion_auc(client: MlflowClient, model_name: str) -> float | None:
    try:
        version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None  # no production model yet
    return float(version.tags.get("test_roc_auc", "nan"))


def export_champion(cfg, best, test_metrics, version, train, feats):
    """Write the self-contained bundle the API serves (no MLflow server needed at serving time)."""
    out = project_path(cfg["paths"]["champion_dir"])
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], out / "model.joblib")
    # Reference data for drift monitoring: what "normal" inputs look like
    train[feats].to_csv(out / "reference.csv", index=False)
    metadata = {
        "model_name": cfg["mlflow"]["registered_model"],
        "model_version": version,
        "model_type": best["name"],
        "threshold": best["threshold"],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": feats,
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "mlflow_run_id": best["run_id"],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> int:
    cfg = load_config()
    setup_mlflow(cfg)
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    target, biz, gate = cfg["data"]["target"], cfg["business"], cfg["quality_gate"]
    model_name = cfg["mlflow"]["registered_model"]

    print("1/4 Loading and validating data")
    train, val, test = load_dataset()
    print(f"    train {len(train)} | val {len(val)} | test {len(test)} | churn rate {train[target].mean():.1%}")

    print("2/4 Training candidate models")
    best = train_candidates(cfg, train, val)[0]
    print(f"    best on validation: {best['name']}")

    print("3/4 Final evaluation on the held-out test set")
    proba = best["pipeline"].predict_proba(test[feats])[:, 1]
    test_metrics = compute_metrics(test[target], proba, best["threshold"], biz)
    with mlflow.start_run(run_id=best["run_id"]), tempfile.TemporaryDirectory() as tmp:
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        plot_evaluation(test[target], proba, best["threshold"], biz, Path(tmp) / "evaluation.png")
        plot_feature_importance(best["pipeline"], Path(tmp) / "feature_importance.png")
        mlflow.log_artifacts(tmp, "plots")
        reports = project_path(cfg["paths"]["reports_dir"])
        reports.mkdir(exist_ok=True)
        for f in ("evaluation.png", "feature_importance.png"):
            (reports / f).write_bytes((Path(tmp) / f).read_bytes())

    print("4/4 Registering and running the quality gate")
    client = MlflowClient()
    version = mlflow.register_model(best["model_uri"], model_name).version
    client.set_model_version_tag(model_name, version, "test_roc_auc", f"{test_metrics['roc_auc']:.4f}")
    auc, current = test_metrics["roc_auc"], champion_auc(client, model_name)
    vs_champion = "no champion yet" if current is None else f"champion {current:.4f}"
    checks = {
        f"test ROC-AUC {auc:.4f} >= {gate['min_roc_auc']}": auc >= gate["min_roc_auc"],
        f"not worse than current production model ({vs_champion})": current is None
        or auc >= current - gate["max_regression_vs_champion"],
        "beats the 'contact everyone' baseline on profit": test_metrics["profit"]
        > test_metrics["profit_contact_everyone"],
    }
    passed = all(checks.values())
    for desc, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {desc}")

    if passed:
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
        client.set_model_version_tag(model_name, version, "quality_gate", "passed")
        meta = export_champion(cfg, best, test_metrics, version, train, feats)
        print(f"    Promoted {model_name} v{version} ({best['name']}) to '{CHAMPION_ALIAS}'")
    else:
        client.set_model_version_tag(model_name, version, "quality_gate", "failed")
        meta = None
        print(f"    {model_name} v{version} rejected; production model unchanged")

    write_summary(best, test_metrics, checks, passed, version, meta)
    return 0 if passed else 1


def write_summary(best, m, checks, passed, version, meta) -> None:
    """Markdown summary for the terminal, the reports folder and the GitHub Actions job page."""
    lines = [
        f"## Training run: {'✅ promoted' if passed else '❌ rejected'} (model v{version}, {best['name']})",
        "",
        "| Test metric | Value |",
        "|---|---|",
        f"| ROC-AUC | {m['roc_auc']:.4f} |",
        f"| PR-AUC | {m['pr_auc']:.4f} |",
        f"| Precision / Recall @ {best['threshold']:.2f} | {m['precision']:.3f} / {m['recall']:.3f} |",
        f"| Campaign profit (model) | ${m['profit']:,.0f} |",
        f"| Campaign profit (contact everyone) | ${m['profit_contact_everyone']:,.0f} |",
        "",
        "**Quality gate**",
        "",
        *[f"- {'✅' if ok else '❌'} {desc}" for desc, ok in checks.items()],
    ]
    text = "\n".join(lines) + "\n"
    project_path("reports").mkdir(exist_ok=True)
    project_path("reports/train_summary.md").write_text(text, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    sys.exit(main())
