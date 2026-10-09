"""Train and compare models for predicting MVP vote share.

Evaluation uses *season-grouped* cross-validation (whole seasons held out), which
mirrors the real task: predict a season we have never seen. A random row split
would leak information because one season's players compete against each other.

Usage:
    python src/train.py
    python src/train.py --test-seasons 2019 2020 2021 2022 2023 2024 2025
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from features import TARGET, build_candidates, feature_columns, load_dataset, top_k_accuracy

RESULTS = Path("results")
MODELS = Path("models")


def get_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42),
    }


def evaluate(name, model, train, test, cols):
    model.fit(train[cols], train[TARGET])
    test = test.copy()
    test["pred"] = np.clip(model.predict(test[cols]), 0, 1)
    return {
        "model": name,
        "rmse": float(np.sqrt(mean_squared_error(test[TARGET], test["pred"]))),
        "r2": float(r2_score(test[TARGET], test["pred"])),
        "top1_acc": top_k_accuracy(test, "pred", 1),
        "top3_acc": top_k_accuracy(test, "pred", 3),
    }, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--test-seasons", type=int, nargs="+", default=list(range(2016, 2026)),
                    help="seasons (ending year) held out for testing")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    df = build_candidates(load_dataset(args.data))
    cols = feature_columns(df)
    train = df[~df["Season"].isin(args.test_seasons)]
    test = df[df["Season"].isin(args.test_seasons)]
    print(f"Train: {len(train)} rows / {train['Season'].nunique()} seasons | "
          f"Test: {len(test)} rows / {test['Season'].nunique()} seasons | {len(cols)} features")

    rows, preds = [], {}
    for name, model in get_models().items():
        metrics, scored = evaluate(name, model, train, test, cols)
        rows.append(metrics)
        preds[name] = scored
        print(f"{name:18s} RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.3f}  "
              f"top1={metrics['top1_acc']:.2f}  top3={metrics['top3_acc']:.2f}")

    table = pd.DataFrame(rows).sort_values(["top1_acc", "rmse"], ascending=[False, True])
    table.to_csv(RESULTS / "model_comparison.csv", index=False)

    best_name = table.iloc[0]["model"]
    print(f"\nBest model: {best_name}")

    # Per-season predictions of the best model (for the report / slides).
    best = preds[best_name]
    summary = []
    for season, g in best.groupby("Season"):
        summary.append({
            "season": int(season),
            "actual_mvp": g.loc[g[TARGET].idxmax(), "Player"],
            "predicted_mvp": g.loc[g["pred"].idxmax(), "Player"],
        })
    pd.DataFrame(summary).to_csv(RESULTS / "per_season_predictions.csv", index=False)

    # Refit best model on ALL data and save for the demo.
    final = get_models()[best_name]
    final.fit(df[cols], df[TARGET])
    joblib.dump({"model": final, "columns": cols, "name": best_name}, MODELS / "best_model.joblib")

    # Plots
    ax = table.set_index("model")[["top1_acc", "top3_acc"]].plot.bar(rot=0, figsize=(7, 4))
    ax.set_ylabel("Fraction of held-out seasons"); ax.set_title("Correct MVP in top-k")
    plt.tight_layout(); plt.savefig(RESULTS / "model_comparison.png", dpi=150); plt.close()

    inner = final[-1] if hasattr(final, "steps") else final
    if hasattr(inner, "feature_importances_"):
        imp = pd.Series(inner.feature_importances_, index=cols).sort_values().tail(12)
        imp.plot.barh(figsize=(7, 5)); plt.title(f"Top features ({best_name})")
        plt.tight_layout(); plt.savefig(RESULTS / "feature_importance.png", dpi=150); plt.close()

    (RESULTS / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"Results saved to {RESULTS}/ and model to {MODELS}/best_model.joblib")


if __name__ == "__main__":
    main()
