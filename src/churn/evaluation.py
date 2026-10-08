"""Model metrics, business-value threshold tuning and evaluation plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def campaign_profit(y_true, proba, threshold: float, business: dict) -> float:
    """Profit of offering a retention deal to every customer with proba >= threshold.

    A contacted churner stays with probability `offer_success_rate` (we keep their value);
    every contact costs `offer_cost`, whether or not the customer was going to churn.
    """
    y_true = np.asarray(y_true)
    contacted = np.asarray(proba) >= threshold
    saved = (contacted & (y_true == 1)).sum() * business["offer_success_rate"]
    return float(saved * business["saved_customer_value"] - contacted.sum() * business["offer_cost"])


def best_threshold(y_true, proba, business: dict) -> float:
    """Threshold that maximizes campaign profit (tuned on validation data, never on test)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    profits = [campaign_profit(y_true, proba, t, business) for t in grid]
    return float(grid[int(np.argmax(profits))])


def compute_metrics(y_true, proba, threshold: float, business: dict) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "profit": campaign_profit(y_true, proba, threshold, business),
        # Baselines: what you'd earn with no model
        "profit_contact_everyone": campaign_profit(y_true, proba, 0.0, business),
        "profit_contact_nobody": 0.0,
    }


def plot_evaluation(y_true, proba, threshold: float, business: dict, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")

    prec, rec, _ = precision_recall_curve(y_true, proba)
    axes[1].plot(rec, prec, label=f"AP = {average_precision_score(y_true, proba):.3f}")
    axes[1].axhline(np.mean(y_true), ls="--", color="gray", label="random")
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")

    grid = np.arange(0.05, 0.96, 0.01)
    profits = [campaign_profit(y_true, proba, t, business) for t in grid]
    axes[2].plot(grid, profits)
    axes[2].axvline(threshold, ls="--", color="crimson", label=f"chosen threshold = {threshold:.2f}")
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set(title="Retention campaign profit", xlabel="Decision threshold", ylabel="Profit ($)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_feature_importance(pipeline, path, top: int = 15) -> None:
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(model.coef_[0])
    order = np.argsort(values)[-top:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[i].split("__", 1)[-1] for i in order], values[order])
    ax.set(title="Top features")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
