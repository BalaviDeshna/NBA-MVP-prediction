# Write-up skeleton (aim: 2 pages, PDF)

Fill the [brackets] with your real numbers after running `python src/train.py`.

## 1. Problem statement  (~4 lines)
The NBA MVP is chosen by a panel of media voters, each ranking five players. We
ask: can season statistics alone predict the MVP vote share of each candidate, and
which statistics matter most? Task: regression on vote share (0-1) per player-season;
predicted MVP = highest predicted share in a season.

## 2. Dataset  (~6 lines + small table)
- Source: Basketball-Reference, seasons 1979-80 to 2024-25 (3-point era onward).
- [N] player-seasons after filtering (>= 41 games; top-15 by Win Shares or any votes).
- Features: per-game (PTS, TRB, AST, STL, BLK, TOV, shooting %), advanced (PER, TS%,
  USG%, WS, WS/48, BPM, VORP), team win %, plus season-relative percentile ranks.
- Target: MVP `Share` (points won / points possible). ~[x]% of candidates have Share = 0
  (class imbalance -> why we rank within season and report top-k accuracy).

## 3. Approach  (~8 lines)
- Why regression + ranking rather than classification: shares keep information about
  close races; classification of a single winner per season is extremely imbalanced.
- Models: Ridge (baseline), Random Forest, Gradient Boosting, XGBoost.
- Validation: whole seasons held out ([test years]) to avoid leakage between players
  competing in the same season.
- Metrics: RMSE, R^2, top-1 and top-3 accuracy.

## 4. Implementation overview  (~6 lines)
Pipeline: `collect_data.py` (scrape/cache/merge) -> `features.py` (filtering,
relative features) -> `train.py` (compare, refit, save) -> `predict.py` (demo).
Repository: github.com/[user]/nba-mvp-prediction (private, shared with faculty/TAs).

## 5. Results  (figure + table)
- Insert `results/model_comparison.png` and the metrics table.
- Insert `results/feature_importance.png`; comment on the top 3 features.
- Table of predicted vs actual MVP per season (`results/per_season_predictions.csv`);
  discuss 2-3 misses (e.g. close races, narrative-driven votes).

## 6. Conclusions  (~5 lines)
- Best model: [name], top-1 = [x], top-3 = [y].
- Most important features: [..] -> voters reward [..].
- Limitations: no narrative/voter-fatigue info, small data (~[N] rows), era effects.
- Future work: pairwise/ranking loss (LambdaMART), team seed features, playoffs
  of prior season, injury/availability.

---
# Presentation outline (10-12 slides, plus live demo)
1. Title + team + individual roles
2. Problem and why it is interesting
3. Dataset (source, size, features, target)
4. Pipeline diagram (collect -> features -> models -> predict)
5. Feature engineering (relative ranks, candidate filter)
6. Models and validation strategy (season hold-out)
7. Results table + chart
8. Feature importance
9. Predicted vs actual by season (hits and misses)
10. Live demo: `python src/predict.py --season 2025 --honest`
11. Conclusions, limitations, future work
12. Q&A

# Likely Q&A questions - make sure every teammate can answer
- Why season-wise split instead of random split?
- Why regress on vote share instead of classifying the winner?
- Why is R^2 low/high yet top-1 accuracy good (or vice versa)?
- How did you handle traded players (TOT rows)?
- What would you change to improve it?
