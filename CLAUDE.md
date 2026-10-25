# Project: reproducing Baamer & Aljamaan (2026), PeerJ CS e4064

## What this project is
A reproduction of **"Ensemble deep learning models for stock price forecasting: evidence from the
Saudi markets"** by R. Baamer and H. Aljamaan, PeerJ Computer Science 12:e4064 (2026),
DOI 10.7717/peerj-cs.4064.

- Paper PDF: `peerj-cs-4064.pdf` in the project root (rename here if the file name differs).
  Read it when a question is about what the paper *says*.
- Authors' code: https://github.com/ranabameer/saudi-stock-forecasting-ensembles
  (Zenodo 10.5281/zenodo.19046571). Key notebooks: `PrepareFeatursOfData.ipynb` (features),
  the standalone model notebooks (unit grid search), `Ensemble_Stacking_in_Neural_Networks.ipynb`
  (older stacking), `Baseline&WX.ipynb` (final: units table, val-based stacking, baselines, Wilcoxon),
  and `TestingResults.ipynb` (GRU(256) benchmark).
- Data: Kaggle `salwaalzahrani/saudi-stock-exchange-tadawul` (Saudi);
  `paultimothymooney/stock-market-data` (US replication, secondary).
- **Scope: the Saudi market is the priority.** Keep US code working but don't spend effort on it.

## The study, in brief
- 10 TASI companies, one per sector: ADC, ALRAJHI, EMAAR EC, FITAIHI GROUP, GACO, SARCO,
  SAUDI ELECTRICITY, SPIMACO, STC, TASNEE. Data runs to 2020-03-05.
- Split: train ≤ 2015-12-31, validation = 2016, test 2017-01-01 → 2020-03-05.
- Features: 34 overlap technical indicators (Table 4) from pandas-ta, length 5. They were chosen
  by Pearson |r| > 0.8 with close on the train rows of all 10 companies, out of ~130 TIs.
  StandardScaler is fit on train only.
- Task: input window w = 5, direct multi-output forecast of the next 5 closes (h = 5).
- Base models: BiRNN (B), GRU (G), LSTM (L). Each is a single recurrent layer (ReLU) + Dense(5),
  trained with Adam and MSE.
- Ensembles: 4 averaging (AVG(B,G,L), AVG(B,G), AVG(B,L), AVG(G,L)) and 12 stacking
  (meta-learner ANN 128-64-32 / GRU 128 / LSTM 128 × the same 4 combinations), Table 7.
- Baselines: Random Walk with drift, ARIMA(5,1,0), XGBoost, and the GRU(256) benchmark of
  Aseeri (2023).
- Metrics: MAE, MSE, RMSE, MRE (and R²) on de-scaled prices rounded to 2 decimals (Tables 10, 13).
  Also timing (Table 12), and Wilcoxon signed-rank tests on non-overlapping-window RMSE with Holm
  correction, summarized as win/loss counts (Table 11).
- Headline claims: ensembles beat the standalone models; STK_GRU(B,G) / STK_LSTM(B,G) give the best
  RMSE for several companies; AVG(B,G) and AVG(B,G,L) top the Wilcoxon ranking (~88% win rate).

## Paper text vs. repository code
These disagree. **Defaults follow the repo**, because the repo produced the reported numbers.
Each difference is a switch in the pipeline:

| Item | Repo (default) | Paper text | Switch |
|---|---|---|---|
| Base training | 32 epochs, batch 5% of train, lr 1e-3, no early stop | 100 epochs, batch 10%, lr 1e-4, early stop 20, best val loss | `--train-config paper` |
| Stacking meta-learner data | 2016 validation-set base preds | rolling-origin OOF preds on train | `--stacking val/train/oof` (`train` = older notebook) |
| Base-model units | fixed per-company table (picked on test RMSE originally) | picked on validation | `--units repo/grid/<int>` |
| ARIMA | one forecast from end of train, repeated for every window | unspecified | `--fix-quirks` → rolling |
| GRU(256) benchmark | no ffill (non-trading days = 0), MinMax fit per split incl. test | — | `--fix-quirks` |
| Wilcoxon | every model-horizon column vs every horizon's actuals | per horizon | `--wx-mode repo/horizon` |

Other repo behavior reproduced on purpose:
- Series are reindexed to **calendar** days and forward-filled, so "5 days ahead" = 5 calendar rows.
- The `hwma` column is actually `dema(close)` with default length.

**Never "fix" a quirk silently.** Fixes go behind a flag, and the default stays repo-faithful.

## Code
- `tasi_edl_pipeline.py`: the current single-file, end-to-end implementation. Treat it as the
  **reference implementation**; don't modify it during the refactor.
- `REFACTOR_PLAN.md`: plan to split the pipeline into two scripts: `prepare_data.py`
  (acquisition, features, split/scale/window → `data/prepared/`) and `train_report.py`
  (models, ensembles, baselines, metrics, Wilcoxon → `runs/<name>/`). It covers the file
  handoff, the parity checks, and `--reuse-base` / `--report-only`. Follow it for refactor work.
- Implemented (2026-10-24): `prepare_data.py` (raw CSV in `data/saudi/`, output `data/prepared/`),
  `train_report.py` (imports data-side constants/loaders from `prepare_data`), and
  `parity_check.py` (exact checks vs the reference; `--run-dir X --ref-run Y` compares metrics).
  `train_report.py` always saves train-set base preds, so `--reuse-base` works for `val` and `train`.
- Added 2026-10-25: `prepare_data_yahoo.py` (fork, Saudi only) downloads up-to-date prices via
  yfinance (cache `data/yahoo/`, output `data/prepared_yahoo/`). The history starts in 2010-03.
  The split is configurable and stored in `meta.json["split"]`, which `load_prepared` and the
  benchmark read (default = paper split). `train_report.py --source yahoo` picks that folder.
  `--adjust kaggle` puts prices on the Kaggle scale. GACO's missing 2017 capital reduction is
  patched in `MISSING_SPLITS`.
- Indicators use **pandas-ta-classic** (import `pandas_ta_classic`). The original
  pandas-ta 0.3.14b0 is no longer on PyPI.
- Environment: python 3.10/3.11, tensorflow 2.15 (also runs on 2.16+/Keras 3), numpy, pandas 2.x,
  scikit-learn, scipy, statsmodels, xgboost, kagglehub.

## Results so far (repo-default config, full Saudi run, 2026-10-24)
| Company | Our best | RMSE | Paper best (Table 10) | Paper RMSE |
|---|---|---|---|---|
| ADC | AVG(B,G) | 0.2658 | STK_GRU(B,G) | 0.2498 |
| ALRAJHI | AVG(B,L) | 1.1755 | STK_GRU(B,G) | 1.0688 |
| GACO | AVG(B,G) | 0.2800 | STK_LSTM(B,L) | 0.2637 |
| EMAAR EC | GRU | 0.3541 | AVG(B,G) | 0.3259 |
| FITAIHI GROUP | AVG(G,L) | 0.2139 | AVG(G,L) | 0.1908 |
| SARCO | GRU | 1.2684 | AVG(B,G) | 1.2307 |
| SAUDI ELECTRICITY | LSTM | 0.4750 | STK_GRU(B,G) | 0.4651 |
| SPIMACO | AVG(B,G) | 0.5269 | AVG(B,G) | 0.5192 |
| STC | BRNN | 1.8894 | STK_LSTM(B,G) | 1.9293 |
| TASNEE | AVG(B,G) | 0.4463 | STK_LSTM(B,G) | 0.4401 |

- Standalone and averaging models reproduce well (within ~12%; STC beats the paper).
- Wilcoxon: averaging ensembles rank top (AVG(B,G,L) 83%); RW 46% (paper 31%); ARIMA 8% (1.5%);
  Benchmark 2% (11%).
- **Stacking does not reproduce under `--stacking val`.** ALRAJHI stacking RMSE is 13–19 (a level
  shift, MRE ~17%), STC 2.5–9.4, EMAAR 0.75–1.6. Stacking win rates are 37–45% vs the paper's 80–87%.
  Hypothesis: a meta-learner fit on ~357 windows from 2016 can't extrapolate to 2017–2020 price
  levels, so Table 10 likely came from the older notebook (meta-learner on train preds).
  (Unverified; the next runs test it.)
- Seed/GPU noise: the same seed gave ADC BRNN 0.2881 vs 0.2762 (~±4%). That is larger than the
  gaps between the top models, so don't over-read small differences.

## Open next steps
- Test the stacking hypothesis: `--stacking train`, then `--stacking oof`. After the refactor,
  these should reuse saved base predictions instead of retraining.
- Optional seed check on ADC (`--seed 1`, `--seed 2`).

## Working conventions
- Preserve numerical behavior during refactors. Verify parity against `tasi_edl_pipeline.py`
  on ADC before moving on (exact equality for features, windows, RW, ARIMA).
- Full runs are long (30 base models plus 12 meta-learners per run). Use `--quick` or a single
  company (`--companies ADC`) for smoke tests, and ask before launching a full run.
- When the paper and the repo disagree, say which one a change follows.