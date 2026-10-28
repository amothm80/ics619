# ICS619: Saudi stock forecasting with ensemble deep learning

Replication of Baamer & Aljamaan (2026). See `CLAUDE.md` for the methodology and project decisions.

## Setup

```bash
source .venv/bin/activate
```

Project settings (tickers, split dates, `W`, `H`, `SEED`, paths) live in `config.py`; no script takes them as arguments.

### Data sources

Every script reads the source from the `DATA_SOURCE` environment variable (default `kaggle`). All fitted and scored output goes to `artifacts/<source>/`, so the sources never mix.

| `DATA_SOURCE` | Data | Rows from | Train to | Val | Test |
|---|---|---|---|---|---|
| `kaggle` | `data/Tadawul_stcks.csv` (Kaggle: `salwaalzahrani/saudi-stock-exchange-tadawul`) | 2001 | 2015-12-31 | 2016 | 2017-01-01 .. 2020-03-05 |
| `yfinance_recent` | Yahoo `<symbol>.SR`, cached in `data/yfinance/` | 2013-06-29 | 2022-12-31 | 2023 | 2024-01-01 .. yesterday |

```bash
DATA_SOURCE=yfinance_recent python sweep.py ADC
python run_pipeline.py --source yfinance_recent
```

Yahoo caveats:
- Prices are split-adjusted but not dividend-adjusted. Yahoo has no record of GACO's 10:6 capital reduction on 2017-02-14, so `data.py` applies it.
- Before the 2013-06-29 weekend change (Thursday–Friday to Friday–Saturday), Yahoo has no Saturday sessions and dates some Wednesday sessions as Thursday. About 20% of those trading days are missing, so `yfinance_recent` starts after the change.
- A few daily moves after 2013 exceed Tadawul's ±10% limit (`python data.py` lists them). Some may be corporate actions Yahoo does not record. They are not corrected.
- The test end is yesterday, because today's row may be an unfinished session. Results change when the cache is refreshed.

Every training run logs live. A `--- <ticker> <model>` header comes first. Keras models then print one line per epoch (loss, val_loss), XGBoost prints train/val RMSE every 10 trees, and ARIMA prints each candidate order's AIC. `config.py` line-buffers stdout, so logs stream through `tee` or a file redirect, e.g. `python sweep.py | tee artifacts/kaggle/unit_sweep.log`.

## Run everything

```bash
python run_pipeline.py                     # all steps, all tickers, Kaggle
python run_pipeline.py --source yfinance_recent
python run_pipeline.py --start stacking    # resume from a step
python run_pipeline.py ADC STC --units 64 128 --folds 5
```

| Argument | Required | Values | Default |
|---|---|---|---|
| `TICKER ...` | no | any of the ten tickers; passed to `sweep`, `ensembles`, `stacking`, `baseline`, `rankings` | all ten |
| `--source` | no | `kaggle`, `yfinance_recent` | `$DATA_SOURCE`, else `kaggle` |
| `--start` | no | `data`, `select_features`, `preprocessing`, `metrics`, `sweep`, `ensembles`, `stacking`, `baseline`, `rankings` | `data` |
| `--units` | no | one or more integers, passed to `sweep.py` | `sweep.py`'s default |
| `--folds` | no | integer, passed to `stacking.py` | `stacking.py`'s default |
| `--epochs` | no | integer: max training epochs | `100` |
| `--patience` | no | integer: early-stopping patience on val loss; `0` turns early stopping off | `20` |

Runs scripts 0-9 below in order (except `train.py`, which `sweep.py` calls), with `DATA_SOURCE` set, each as its own process. Output streams to the terminal and to `artifacts/<source>/logs/<step>.log`. The run stops at the first step that fails. `select_features`, `preprocessing` and `metrics` always use all ten tickers. Feature selection must, since a feature passes only if it passes for all ten. `--epochs` and `--patience` reach every training step through the `EPOCHS` and `PATIENCE` environment variables. That covers base models, OOF fold models (via the kept epoch), meta-learners, GRU-256, and XGBoost's early-stopping rounds. Saved sweep runs, GRU-256 and OOF forecasts are reused only if they were made with the same epochs and patience; otherwise they are retrained. Delete them to retrain after changing the data or other settings.

## Scripts, in pipeline order

Run each step after the one before it; later steps read the artifacts earlier steps save.

| # | Script | Arguments | Does | Writes |
|---|---|---|---|---|
| 0 | `data.py` | `[--refresh]` | Downloads missing Yahoo histories, prints coverage, split sizes and checks | `data/yfinance/<symbol>.SR.csv` |
| 1 | `select_features.py` | none | Pearson \|r\| > 0.8 with Close on train rows, all ten tickers | `artifacts/<source>/feature_order.json` |
| 2 | `preprocessing.py` | none | Fits train-only scalers per ticker, builds windows, runs checks | `artifacts/<source>/<ticker>/{x_scaler,y_scaler}.joblib`, `features.json` |
| 3 | `metrics.py` | none | Random-walk-with-drift baseline on val and test | `artifacts/<source>/baseline_random_walk.csv` |
| 4 | `train.py` | `KIND TICKER [--units N]` | Trains one base model | `artifacts/<source>/<ticker>/models/<KIND>_<N>.{keras,json}`, `_loss.png` |
| 5 | `sweep.py` | `[TICKER ...] [--units N ...]` | Unit-size sweep, selected on validation loss | `artifacts/<source>/unit_sweep.csv`, `artifacts/<source>/<ticker>/config.json`, models |
| 6 | `ensembles.py` | `[TICKER ...]` | Averaging ensembles of the selected models | `artifacts/<source>/ensembles_avg.csv`, `artifacts/<source>/<ticker>/forecasts.npz` |
| 7 | `stacking.py` | `[TICKER ...] [--folds K]` | Rolling-origin OOF forecasts, then 12 stacking meta-learners | `artifacts/<source>/ensembles_stk.csv`, `artifacts/<source>/<ticker>/oof_k<K>_init50.npz`, `models/STK_*.keras` |
| 8 | `baseline.py` | `[TICKER ...]` | ARIMA, XGBoost and GRU-256 baselines, plus a test ranking of every model so far | `artifacts/<source>/baselines.csv`, `artifacts/<source>/<ticker>/baselines.npz`, `models/XGB.joblib` |
| 9 | `rankings.py` | `[TICKER ...]` | Pairwise Wilcoxon tests on non-overlapping 5-step test windows, Holm-corrected, win/loss counts and mean metrics | `artifacts/<source>/wilcoxon_pairs.csv`, `artifacts/<source>/results_test.csv` |

### `data.py`

```bash
DATA_SOURCE=yfinance_recent python data.py
python data.py --refresh       # download every Yahoo history again
```

| Argument | Required | Values | Default |
|---|---|---|---|
| `--refresh` | no | flag | off |

Downloads any Yahoo history missing from `data/yfinance/` when the source is a yfinance one (always, with `--refresh`). Then, for each ticker of the current source, prints its row range and split sizes, the trading-day check result, and daily moves beyond ±10.5%. Other scripts import the loaders. `load_raw()` returns every ticker in the Kaggle layout, whatever the source.

### `select_features.py`

```bash
python select_features.py
```

No arguments. Scores every pandas-ta-classic indicator (length 5) on the training split only and saves the paper's 34 features that pass, in a fixed order.

### `preprocessing.py`

```bash
python preprocessing.py
```

No arguments. Prepares every ticker: fits `StandardScaler`s for X and Close on train rows, saves them, and prints window shapes per split. `train.py`, `sweep.py` and `ensembles.py` call the same code, so running this by hand is only needed for the checks.

### `metrics.py`

```bash
python metrics.py
```

No arguments. Scores the random walk with drift on val and test for every ticker (MAE, MSE, RMSE, MRE on prices).

### `train.py`

```bash
python train.py GRU ADC --units 64
```

| Argument | Required | Values | Default |
|---|---|---|---|
| `KIND` | yes | `BiRNN`, `GRU`, `LSTM` | |
| `TICKER` | yes | `ADC`, `ALRAJHI`, `EMAAR_EC`, `FITAIHI`, `GACO`, `SARCO`, `SEC`, `SPIMACO`, `STC`, `TASNEE` | |
| `--units` | no | integer | `64` |

Table 8 settings: Adam lr 1e-4, MSE, batch 10% of training windows, 100 epochs, early stopping on val loss (patience 20, best weights restored). Override with the `EPOCHS` and `PATIENCE` environment variables (e.g. `EPOCHS=50 PATIENCE=0 python train.py GRU ADC`). With `PATIENCE=0` there is no early stopping and the last epoch's weights are kept. Logs one line per epoch, then prints val and test RMSE next to the random walk.

### `sweep.py`

```bash
python sweep.py                         # all tickers, units 32 64 128 256
python sweep.py ADC STC --units 64 128  # subset
```

| Argument | Required | Values | Default |
|---|---|---|---|
| `TICKER ...` | no | any of the ten tickers | all ten |
| `--units` | no | one or more integers | `32 64 128 256` |

Trains every architecture × unit size per ticker and keeps the size with the lowest validation loss. Test windows are not predicted. Runs already saved in the current format are reused, so a stopped sweep resumes where it left off. A full run is 120 models, about 70 minutes on one GPU.

### `ensembles.py`

```bash
python ensembles.py
python ensembles.py ADC STC
```

| Argument | Required | Values | Default |
|---|---|---|---|
| `TICKER ...` | no | tickers that have a `config.json` from `sweep.py` | all ten |

Loads the selected BiRNN, GRU and LSTM models and scores AVG(B,G,L), AVG(B,G), AVG(B,L) and AVG(G,L) as equal-weight means. Prints val and test RMSE tables and a count of tickers where each average beats its best member.

### `stacking.py`

```bash
python stacking.py
python stacking.py ADC --folds 5
```

| Argument | Required | Values | Default |
|---|---|---|---|
| `TICKER ...` | no | tickers that have a `config.json` from `sweep.py` | all ten |
| `--folds` | no | integer K | `5` |

Rolling-origin OOF: every fold trains B, G and L on at least the first 50% of the training windows (`INITIAL`), then predicts the next of K blocks. The last H-1 windows before each block are dropped so no training target overlaps a predicted one. Fold models train for Step 9's best epoch count. ANN, GRU-128 and LSTM-128 meta-learners train on the OOF forecasts (early stopping on validation) and are scored on the Step 9 models' val/test forecasts. Needs `ensembles.py` run first for the comparison table. OOF forecasts are cached per ticker.

### `baseline.py`

```bash
python baseline.py
python baseline.py ADC STC
```

| Argument | Required | Values | Default |
|---|---|---|---|
| `TICKER ...` | no | any of the ten tickers | all ten |

- **ARIMA**: Close only. The order (p, 1, q) with p, q in 0..2 is picked by AIC on train closes. Parameters are fitted on train and then frozen. Each val/test window forecasts its 5 closes from the closes up to its last input row, using a Kalman filter with no refit.
- **XGBoost**: the same scaled 5 × 34 windows, flattened, with multi-output trees. Fitted on train windows. Validation drives early stopping only (patience 20).
- **GRU-256**: the Aseeri (2023) benchmark. Reuses `models/GRU_256.keras` from `sweep.py`, or trains it with the Table 8 settings if it is missing.

Prints val/test RMSE per ticker, then ranks every model in `baseline_random_walk.csv`, `ensembles_avg.csv`, `ensembles_stk.csv` and this run by mean per-ticker test RMSE rank. Ranks are used because price levels differ by ticker. Run `metrics.py` first; the ensemble CSVs are optional.

### `rankings.py`

```bash
python rankings.py
python rankings.py ADC STC
```

| Argument | Required | Values | Default |
|---|---|---|---|
| `TICKER ...` | no | tickers with `forecasts.npz`, `baselines.npz` and stacking models | all ten |

Collects the test forecasts of all 23 models per ticker. Sources: `forecasts.npz` (base models and averages), the saved `STK_*` meta-learners, `baselines.npz`, and the random walk. It checks each RMSE against the earlier CSVs. It keeps every 5th stride-1 window (non-overlapping 5-step blocks) and scores one RMSE per block. Every model pair gets a Wilcoxon signed-rank test, and the p-values within each ticker are Holm-Bonferroni corrected (alpha 0.05). Each pair is a win for the model with the lower mean RMSE, or a tie if not significant. Identical forecasts count as a tie. Prints wins, losses, ties and win rate = wins / (wins + losses), summed over tickers, next to mean MAE, MSE, RMSE and MRE. Run steps 3, 6, 7 and 8 first.

## Proofs of concept

### `PoC/sahmk_history.py`

```bash
export SAHMK_API_KEY=...            # Starter plan or higher; never commit the key
python PoC/sahmk_history.py         # ADC (2140), from 2000-01-01
python PoC/sahmk_history.py 7010 --from 2010-01-01
```

| Argument | Required | Values | Default |
|---|---|---|---|
| `SYMBOL` | no | Tadawul symbol | `2140` (ADC) |
| `--from` | no | `YYYY-MM-DD` | `2000-01-01` |

Downloads one ticker's full daily OHLCV history from the SAHMK API (`GET /historical/{symbol}/`, `interval=1d`). It pages 2000 bars at a time while `has_more` is true, and retries on HTTP 429 after `Retry-After`. It checks the row count against the API's `total`. Saves the rows as returned to `PoC/data/<symbol>_1d.csv` (date, OHLC, volume, `adjusted_close`, `turnover`, `trade_count`) and the response metadata to `<symbol>_1d.json`. Prints the date range and rows per weekday, to check the pre-2013 Saturday sessions.

## Library modules (not run directly)

`config.py` (settings, data source, seeds), `data.py` (load from Kaggle or Yahoo, split; also runnable, see above), `features.py` (indicators), `models.py` (`build_rnn`).
