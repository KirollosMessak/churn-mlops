"""Data and prediction drift detection: compares recent production inputs to the training data.

- Numeric features: Population Stability Index (PSI) + Kolmogorov-Smirnov test
- Categorical features: PSI over category frequencies
- Predictions: shift in average predicted churn probability

Writes reports/drift_report.html and reports/drift_report.json.
Usage: python -m churn.monitoring.drift [--window 500] [--fail-on-drift]
Exit code 2 with --fail-on-drift when retraining is recommended (for schedulers/CI).
"""

import argparse
import base64
import html
import io
import json
import sys
from datetime import UTC, datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from churn.config import load_config, project_path

EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI = sum((a - e) * ln(a / e)) over bins. <0.1 stable, 0.1-0.2 moderate, >0.2 significant."""
    e, a = np.clip(expected, EPS, None), np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_bins(ref: pd.Series, cur: pd.Series, bins: int = 10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref)
    a = np.histogram(cur, edges)[0] / len(cur)
    return e, a


def category_freqs(ref: pd.Series, cur: pd.Series):
    cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    e = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return cats, e, a


def load_recent(log_path, window: int) -> pd.DataFrame:
    if not log_path.exists():
        raise SystemExit(f"No prediction log at {log_path}. Send traffic first (churn.monitoring.simulate).")
    lines = log_path.read_text(encoding="utf-8").splitlines()[-window:]
    rows = [json.loads(line) for line in lines if line.strip()]
    df = pd.DataFrame([r["features"] for r in rows])
    df["churn_probability"] = [r["churn_probability"] for r in rows]
    return df


def analyze(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict) -> dict:
    mon, feats = cfg["monitoring"], cfg["features"]
    results = []
    for col in feats["numeric"]:
        e, a = numeric_bins(reference[col], current[col])
        score = psi(e, a)
        p = ks_2samp(reference[col], current[col]).pvalue
        results.append(
            {
                "feature": col,
                "type": "numeric",
                "psi": score,
                "ks_pvalue": float(p),
                "drifted": bool(score > mon["psi_threshold"] or p < mon["ks_pvalue"]),
                "ref_mean": float(reference[col].mean()),
                "cur_mean": float(current[col].mean()),
            }
        )
    for col in feats["categorical"]:
        cats, e, a = category_freqs(reference[col], current[col])
        score = psi(e, a)
        top = cats[int(np.argmax(np.abs(a - e)))]
        results.append(
            {
                "feature": col,
                "type": "categorical",
                "psi": score,
                "ks_pvalue": None,
                "drifted": score > mon["psi_threshold"],
                "biggest_change": f"{top}: {e[cats.index(top)]:.0%} → {a[cats.index(top)]:.0%}",
            }
        )
    results.sort(key=lambda r: r["psi"], reverse=True)
    share = sum(r["drifted"] for r in results) / len(results)
    return {
        "features": results,
        "drift_share": share,
        "retrain_recommended": bool(share >= mon["drift_share_to_retrain"]),
    }


def _plot(reference, current, col, numeric: bool) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    if numeric:
        bins = np.histogram_bin_edges(pd.concat([reference[col], current[col]]), 20)
        ax.hist(reference[col], bins, density=True, alpha=0.55, label="training")
        ax.hist(current[col], bins, density=True, alpha=0.55, label="production")
    else:
        cats, e, a = category_freqs(reference[col], current[col])
        x = np.arange(len(cats))
        ax.bar(x - 0.2, e, 0.4, label="training")
        ax.bar(x + 0.2, a, 0.4, label="production")
        ax.set_xticks(x, [c[:14] for c in cats], rotation=20, fontsize=7)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def write_html(report: dict, reference, current, path) -> None:
    rows = "".join(
        f"<tr class='{'bad' if r['drifted'] else ''}'><td>{html.escape(r['feature'])}</td><td>{r['type']}</td>"
        f"<td>{r['psi']:.3f}</td><td>{'' if r['ks_pvalue'] is None else f'{r["ks_pvalue"]:.2g}'}</td>"
        f"<td>{html.escape(r.get('biggest_change') or f'mean {r["ref_mean"]:.1f} → {r["cur_mean"]:.1f}')}</td>"
        f"<td>{'⚠️ drift' if r['drifted'] else '✓ stable'}</td></tr>"
        for r in report["features"]
    )
    charts = "".join(
        f"<img src='data:image/png;base64,{_plot(reference, current, r['feature'], r['type'] == 'numeric')}'>"
        for r in report["features"][:6]
    )
    verdict = (
        "<div class='verdict bad'>⚠️ Retraining recommended</div>"
        if report["retrain_recommended"]
        else "<div class='verdict ok'>✓ No action needed</div>"
    )
    p = report["predictions"]
    path.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Drift report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1f2933}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:6px 8px;border-bottom:1px solid #e4e7eb;text-align:left}}
tr.bad td{{background:#fff1f0}}.verdict{{font-size:22px;font-weight:600;padding:12px 16px;border-radius:8px;margin:12px 0}}
.verdict.bad{{background:#fff1f0;color:#a61b1b}}.verdict.ok{{background:#effcf6;color:#0e6245}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{background:#f5f7fa;border-radius:8px;padding:10px 14px}}
.card b{{display:block;font-size:20px}}img{{margin:6px}}</style></head><body>
<h1>Model monitoring: drift report</h1><p>Generated {report["generated_at"]} · model v{report["model_version"]}
· {report["window"]} recent predictions vs {report["reference_size"]} training rows</p>{verdict}
<div class="cards"><div class="card">Features drifted<b>{sum(r["drifted"] for r in report["features"])} / {len(report["features"])}</b></div>
<div class="card">Avg predicted churn (training)<b>{p["reference_mean"]:.1%}</b></div>
<div class="card">Avg predicted churn (production)<b>{p["current_mean"]:.1%}</b></div></div>
<h2>Features</h2><p>PSI &lt; 0.1 stable · 0.1–0.2 moderate · &gt; 0.2 significant shift.</p>
<table><tr><th>Feature</th><th>Type</th><th>PSI</th><th>KS p-value</th><th>Change</th><th>Status</th></tr>{rows}</table>
<h2>Biggest shifts</h2>{charts}</body></html>""",
        encoding="utf-8",
    )


def plot_psi(report: dict, threshold: float, path) -> None:
    feats = report["features"][::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(
        [f["feature"] for f in feats],
        [f["psi"] for f in feats],
        color=["#d64545" if f["drifted"] else "#3e7bfa" for f in feats],
    )
    ax.axvline(threshold, ls="--", color="gray", label=f"drift threshold (PSI {threshold})")
    verdict = "retraining recommended" if report["retrain_recommended"] else "stable"
    ax.set(title=f"Feature drift (PSI): {verdict}", xlabel="Population Stability Index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=500, help="number of most recent predictions to analyze")
    ap.add_argument("--fail-on-drift", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    model_dir = project_path(cfg["paths"]["champion_dir"])
    reference = pd.read_csv(model_dir / "reference.csv")
    reference["SeniorCitizen"] = reference["SeniorCitizen"].astype(str)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    current = load_recent(project_path(cfg["paths"]["prediction_log"]), args.window)

    report = analyze(reference, current, cfg)
    ref_pred = joblib.load(model_dir / "model.joblib").predict_proba(reference[metadata["features"]])[:, 1]
    report |= {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_version": metadata["model_version"],
        "window": len(current),
        "reference_size": len(reference),
        "predictions": {
            "reference_mean": float(ref_pred.mean()),
            "current_mean": float(current["churn_probability"].mean()),
        },
    }

    out = project_path(cfg["paths"]["reports_dir"])
    out.mkdir(exist_ok=True)
    (out / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html(report, reference, current, out / "drift_report.html")
    plot_psi(report, cfg["monitoring"]["psi_threshold"], out / "drift_psi.png")

    drifted = [r["feature"] for r in report["features"] if r["drifted"]]
    print(f"Drifted features: {len(drifted)}/{len(report['features'])} {drifted}")
    print(
        f"Avg predicted churn: training {report['predictions']['reference_mean']:.1%} → "
        f"production {report['predictions']['current_mean']:.1%}"
    )
    print("RETRAINING RECOMMENDED" if report["retrain_recommended"] else "No action needed")
    print(f"Report: {out / 'drift_report.html'}")
    return 2 if args.fail_on_drift and report["retrain_recommended"] else 0


if __name__ == "__main__":
    sys.exit(main())
