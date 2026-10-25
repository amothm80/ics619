# Reproducing Baamer & Aljamaan (2026), PeerJ CS e4064

Reproduction of *"Ensemble deep learning models for stock price forecasting: evidence from the
Saudi markets"* (DOI 10.7717/peerj-cs.4064). Defaults follow the authors' repository
(https://github.com/ranabameer/saudi-stock-forecasting-ensembles). Each place where the paper
text differs from the repo is a command-line switch (see [Paper vs. repo](#paper-vs-repo)).

The pipeline is split into two scripts that pass files to each other:

| Script | Does | Writes |
|---|---|---|
| `prepare_data.py` | download, features, split / scale / window | `data/prepared/{market}/{company}/` |
| `train_report.py` | base models, ensembles, baselines, metrics, Wilcoxon | `runs/<name>/` |

`tasi_edl_pipeline.py` is the original single-file version. It is kept as the reference
implementation, and `parity_check.py` compares the two.

## Setup

Python 3.10–3.12 with TensorFlow (2.15, or 2.16+ / Keras 3):

```bash
python -m venv .venv && source .venv/bin/activate
pip install tensorflow numpy pandas scikit-learn scipy statsmodels xgboost kagglehub joblib
pip install -r requirements.txt          # pandas-ta-classic, pyarrow
```

Data comes from Kaggle: `salwaalzahrani/saudi-stock-exchange-tadawul` (Saudi) and
`paultimothymooney/stock-market-data` (US). Public datasets usually download anonymously. If
kagglehub asks for credentials, set `KAGGLE_USERNAME` / `KAGGLE_KEY` or add `~/.kaggle/kaggle.json`.
If you already have the CSV, put it in `data/saudi/Tadawul_stcks.csv` or pass `--saudi-csv`.

## Quick start

```bash
# 1. prepare data once per feature configuration
python prepare_data.py

# 2. smoke test: 2 epochs, 8 units, one company
python train_report.py --run-dir runs/quick --companies ADC --quick

# 3. one company, full settings (ADC: ~3 min on an RTX 5070 Ti; slower with --deterministic)
python train_report.py --run-dir runs/adc --companies ADC

# 4. full Saudi run: 10 companies x (3 base models + 12 meta-learners + baselines); takes hours
python train_report.py --run-dir runs/repo
```

Results print to the console and `runs/<name>/run.log`, and are saved as CSVs (see [Outputs](#outputs)).

## Step 1: `prepare_data.py`

```bash
python prepare_data.py                                   # Saudi, the 34 TIs of Table 4
python prepare_data.py --companies ADC STC               # subset
python prepare_data.py --saudi-csv path/Tadawul_stcks.csv
python prepare_data.py --market both --us-dir path/csvs  # also US (AAPL, BAC, GOOG, JNJ)
python prepare_data.py --feature-selection pearson --out data/prepared_pearson
```

| Flag | Default | Meaning |
|---|---|---|
| `--market` | `saudi` | `saudi`, `us` or `both` |
| `--companies` | all | names as in the paper (`ADC`, `"EMAAR EC"`, …) or US tickers |
| `--data-dir` | `data` | where raw CSVs are found or downloaded |
| `--saudi-csv`, `--us-dir` | – | use local files instead of downloading |
| `--feature-selection` | `fixed` | `fixed` = Table 4 list; `pearson` = re-run \|r\| > threshold selection over ~130 TIs on train rows |
| `--pearson-threshold` | `0.8` | |
| `--out` | `data/prepared` | output folder |
| `--force` | off | rebuild companies that are already prepared (default: skip them) |

Each company folder contains `features.parquet`, `windows.npz` (scaled windows plus
original-scale actuals), `scalers.joblib`, `raw.parquet` (for the GRU(256) benchmark) and
`meta.json`.

### Up-to-date data from Yahoo Finance: `prepare_data_yahoo.py`

A fork of `prepare_data.py` (Saudi only) that downloads current daily prices for the same 10
companies with `yfinance` (`2140.SR`, `1120.SR`, …) and writes the same folder layout to
`data/prepared_yahoo/`. The features, scaling and windowing code is the same.

```bash
pip install yfinance
python prepare_data_yahoo.py                       # paper split; test = 2017-01-01 .. latest
python prepare_data_yahoo.py --refresh             # download again (cache: data/yahoo/{symbol}.csv)
python prepare_data_yahoo.py --train-end 2021-12-31 --val-end 2022-12-31 --out data/prepared_yahoo_2022
python train_report.py --source yahoo --run-dir runs/yahoo
```

| Flag | Default | Meaning |
|---|---|---|
| `--adjust` | `split` | `split` = Yahoo Close, split/bonus-adjusted to today; `kaggle` = split-adjusted as of 2020-03-05, on the Kaggle file's price scale; `total` = also dividend-adjusted |
| `--train-end`, `--val-end` | `2015-12-31`, `2016-12-31` | split boundaries; validation and test start the day after each |
| `--end` | last common date | last date of the test set |
| `--refresh`, `--force`, `--companies`, `--feature-selection`, `--out` | | as in `prepare_data.py` |

How it differs from the Kaggle data:
- Yahoo's Tadawul history starts in **March 2010**, so the default training set covers
  2010-03 to 2015-12 (~2,100 windows, about 40% of the Kaggle training set).
- Kaggle prices are split-adjusted as of 2020-03. With `--adjust split`, prices of companies with
  later splits are on a different scale (ALRAJHI ÷2.4, STC ÷2.5, FITAIHI ÷5), so RMSE in SAR is not
  directly comparable with Table 10. `--adjust kaggle` matches the Kaggle closes over 2010–2020
  (median ratio 1.000 for all 10).
- Yahoo has no record of GACO's 10:6 capital reduction on 2017-02-14. `MISSING_SPLITS` in the
  script adds it.
- GRU(256) benchmark inputs: `change` and `perc_change` are computed from close, `value_traded` is
  approximated as close × volume, and `no_trades` is not available.
- The split is stored in `meta.json["split"]`. `train_report.py` reads it from there (folders
  without it use the paper's split). The repo's per-company units table was chosen on the Kaggle
  data. It is still used by default; `--units grid` re-selects units.

## Step 2: `train_report.py`

```bash
python train_report.py --run-dir runs/repo                         # repo-faithful defaults
python train_report.py --run-dir runs/paper --train-config paper --stacking oof
python train_report.py --run-dir runs/seed1 --seed 1 --companies ADC
python train_report.py --run-dir runs/adc_det --companies ADC --deterministic
```

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `kaggle` | `kaggle` reads `data/prepared`, `yahoo` reads `data/prepared_yahoo` |
| `--data` | from `--source` | prepared-data folder (overrides `--source`) |
| `--run-dir` | `runs/default` | where this experiment's outputs go |
| `--market`, `--companies` | `saudi`, all | |
| `--train-config` | `repo` | `repo`: 32 epochs, batch 5%, lr 1e-3, no early stop. `paper`: 100 epochs, batch 10%, lr 1e-4, early stop 20 |
| `--stacking` | `val` | meta-learner training data: `val` (2016 validation preds, final repo), `train` (train-set preds, older notebook), `oof` (rolling-origin out-of-fold, paper text) |
| `--oof-folds` | `5` | folds for `--stacking oof` |
| `--units` | `repo` | `repo` (fixed per-company table), `grid` (search on validation RMSE), or an integer |
| `--unit-grid` | repo grid | custom grid for `--units grid` |
| `--fix-quirks` | off | rolling ARIMA forecasts; leak-free, forward-filled GRU(256) benchmark |
| `--wx-mode` | `repo` | Wilcoxon `repo` (all columns vs every horizon) or `horizon` |
| `--wx-models` | `paper` | `paper` = the 16 models of Table 11, or `all` |
| `--skip-baselines`, `--skip-wilcoxon` | off | |
| `--seed` | `42` | |
| `--deterministic` | off | TF op determinism (slower; needed for exact reruns) |
| `--save-models` | off | also save `.keras` files |
| `--quick` | off | smoke test: 2 epochs, 8 units, 2 OOF folds |
| `--quiet` | off | hide the training-progress lines (loss / val_loss at epoch 1, every ¼ of the epochs, and the last) |
| `--force` | off | redo companies that already have saved predictions |
| `--reuse-base RUN_DIR` | – | load base-model predictions from another run instead of training |
| `--report-only` | off | no training: rebuild all CSVs and Wilcoxon from saved predictions |

### Resuming, reusing and re-reporting

- **Resume:** rerunning with the same `--run-dir` skips companies that already have
  `preds/{company}.npz`. Use `--force` to redo them.
- **New stacking variant without retraining base models:** base-model predictions on
  train/val/test are always saved, so a stacking variant only trains the 12 meta-learners:
  ```bash
  python train_report.py --run-dir runs/stk_train --reuse-base runs/repo --stacking train
  ```
  `--stacking oof` with `--reuse-base` still trains the fold models it needs.
- **New report from saved predictions** (e.g. a different Wilcoxon mode):
  ```bash
  python train_report.py --run-dir runs/repo --report-only --wx-mode horizon
  ```

## Outputs

Under `runs/<name>/`:

| File | Content |
|---|---|
| `results_{market}_all_models.csv` | MAE, MSE, RMSE, MRE, R², times for every company × model |
| `results_{market}_best_per_company.csv` | best model per company next to the paper's Table 10/13 |
| `results_{market}_rmse_matrix.csv` | model × company RMSE |
| `results_{market}_timing.csv` | mean train/inference time per model (Table 12) |
| `wilcoxon_ranking_{market}_{mode}.csv` | win/loss ranking next to the paper's Table 11 |
| `wilcoxon_pairwise_{market}_{mode}.csv` | every pairwise test with Holm-corrected p-values |
| `preds/{company}.npz` | every model's original-scale predictions, actuals and times |
| `base_preds/{company}.npz` | scaled BRNN/GRU/LSTM predictions on train/val/test (for `--reuse-base`) |
| `predictions/*_block_predictions.csv` | the frame the Wilcoxon test runs on |
| `run.log`, `run_config.json`, `unit_search.json` | log; args + hash of the prepared data used; grid-search cache |

Metrics are computed on de-scaled prices rounded to 2 decimals, as in the repo. Wilcoxon
runs for the Saudi market only, as in the paper.

## Paper vs. repo

The defaults reproduce the repository code, because that code produced the reported numbers.
Quirks that are kept on purpose:

- Series are reindexed to calendar days and forward-filled, so "5 days ahead" means 5 calendar rows.
- The `hwma` feature is actually `dema(close)` with default length.
- ARIMA makes one forecast from the end of train and repeats it for every window (`--fix-quirks` → rolling).
- The GRU(256) benchmark doesn't forward-fill and fits MinMax per split, including test (`--fix-quirks`).
- Wilcoxon compares every model-horizon column with every horizon's actuals (`--wx-mode horizon`).

## Checking parity with the reference

```bash
python parity_check.py                       # exact: features, windows, scalers, RW, ARIMA, XGBoost
python tasi_edl_pipeline.py --companies ADC --deterministic --out-dir runs/ref_adc_det
python train_report.py --run-dir runs/adc_det --companies ADC --deterministic
python parity_check.py --run-dir runs/adc_det --ref-run runs/ref_adc_det
```

With `--deterministic` and the same seed, the two pipelines give identical metrics for every
model on ADC. Without `--deterministic`, GPU nondeterminism moves RMSE by about ±4% between
runs with the same seed, which is larger than the gaps between the top models.
