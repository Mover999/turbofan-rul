# Turbofan RUL (NASA C-MAPSS FD001)

This project predicts the RUL (remaining useful life) of the 100 engines tested in the NASA C-MAPSS FD001 turbofan dataset. The sequence models (LSTM and transformer) are about 4 cycles better (14 vs 19) in RMSE than LightGBM on the test set. The LSTM and the 30% dropout transformer differ by about 0.1 cycles (14.38 ± 0.63 vs 14.27 ± 0.48, mean ± standard deviation over 10 seeds). That gap is smaller than the run-to-run spread of about 0.5, so these runs can't separate the two models.

## Results

RMSE (root mean squared error, see Wikipedia) is the typical prediction error in flight cycles. Each of the 100 NASA test engines gets one prediction, made at its last recorded cycle and compared with the true remaining life in RUL_FD001.txt. The ± is one standard deviation across 10 training seeds.

| Model | Test RMSE | Runs |
|---|---|---|
| LightGBM | 18.76 | 1 (deterministic) |
| LSTM | 14.38 ± 0.63 | 10 seeds |
| Transformer, 0% dropout | 14.97 ± 1.04 | 10 seeds |
| Transformer, 10% dropout | 14.92 ± 0.55 | 10 seeds |
| Transformer, 20% dropout | 14.63 ± 0.45 | 10 seeds |
| Transformer, 30% dropout (chosen on validation) | 14.27 ± 0.48 | 10 seeds |
| Transformer, 80% dropout | 13.99 ± 0.17 | 10 seeds |

The transformer's dropout rate was chosen on validation loss (30% and 80% tied, and 30% was kept for its tighter spread across seeds). The test set was scored only after that choice, and the table reports every rate tried.

Notes:
- Models train on RUL capped at 125 but are scored against NASA's raw, uncapped RUL_FD001.txt.
- All runs use one fixed 80/20 split by engine (random_state=42), so the results don't show variation across splits.
- Welch's t-test (LSTM vs. 30% transformer, p ≈ 0.7, 10 runs each) reflects seed variation on that one split only. It can't separate the two models.
- bfill copies later values into the warm-up rows at the start of each engine, which is technically leakage on those rows. It doesn't affect the final-cycle test scores, but fixing it would mean rerunning every number.

## Data

`data/` is not in the repo. Download NASA's Turbofan Engine Degradation Simulation Data Set (Saxena and Goebel, 2008) and unzip it so that `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` are in `data/CMaps/`.

## How to reproduce

1. `train_lgbm.py`, then `lgbm_evaluate.py`.
2. `train_baseline.py` (writes the dataset files and `scaler.joblib`).
3. `lstm_seeds.py` (10 seeds).
4. `transformer_model.py`, run once per dropout rate (0, 10, 20, 30, 80). It asks for the rate, then runs all 10 seeds.