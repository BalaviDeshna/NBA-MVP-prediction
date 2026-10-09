"""Live demo: predict the MVP race for any season in the dataset.

The saved model was trained on all seasons, so for an honest demo of a past season
use `--honest`, which retrains without that season first.

Usage:
    python src/predict.py --season 2025
    python src/predict.py --season 2023 --honest --top 10
"""
import argparse

import joblib
import numpy as np

from features import TARGET, build_candidates, load_dataset
from train import get_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True, help="season ending year, e.g. 2025 for 2024-25")
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--honest", action="store_true", help="retrain excluding the chosen season")
    args = ap.parse_args()

    df = build_candidates(load_dataset(args.data))
    bundle = joblib.load("models/best_model.joblib")
    cols, name = bundle["columns"], bundle["name"]
    season = df[df["Season"] == args.season].copy()
    if season.empty:
        raise SystemExit(f"No data for season {args.season}")

    if args.honest:
        model = get_models()[name]
        train = df[df["Season"] != args.season]
        model.fit(train[cols], train[TARGET])
    else:
        model = bundle["model"]

    season["predicted_share"] = np.clip(model.predict(season[cols]), 0, 1)
    ranked = season.sort_values("predicted_share", ascending=False).head(args.top)

    print(f"\nPredicted MVP race, {args.season - 1}-{str(args.season)[2:]}  (model: {name}{', honest' if args.honest else ''})")
    print("-" * 62)
    for i, (_, r) in enumerate(ranked.iterrows(), 1):
        actual = f"   actual share {r[TARGET]:.3f}" if r[TARGET] > 0 else ""
        print(f"{i:>2}. {r['Player']:<26} predicted share {r['predicted_share']:.3f}{actual}")
    print(f"\nPredicted MVP: {ranked.iloc[0]['Player']}")


if __name__ == "__main__":
    main()
