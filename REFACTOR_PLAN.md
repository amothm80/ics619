# Refactor plan: split `tasi_edl_pipeline.py` into two scripts

## Context
- `tasi_edl_pipeline.py` is a single-file reproduction of Baamer & Aljamaan (2026), PeerJ CS e4064.
  See `CLAUDE.md` for the paper, the repo, the decisions, and the results so far.
- Defaults follow the **repository's** final notebook, not the paper text. The repo quirks
  (ARIMA single forecast, GRU(256) benchmark scaling/no-ffill, Wilcoxon all-columns, `hwma` = dema)
  are reproduced on purpose and stay behind `--fix-quirks` / `--wx-mode`. **Do not fix them silently.**
- The Saudi market is the priority. Keep the US path working.

## Goal
Two scripts with a clean file handoff:

1. **`prepare_data.py`** — data acquisition and preprocessing. Run once per feature configuration.
2. **`train_report.py`** — training, ensembles, baselines, metrics, Wilcoxon, and reports. Run
   once per experiment variant.

The split makes data preparation a one-off. It also makes it possible to re-run stacking
variants and reports without retraining the base models.

## Script 1: `prepare_data.py`

**Contains** (moved from the single file, with logic unchanged):
- §0 data-side constants: dates, `N_IN`/`N_OUT`, `SAUDI_COMPANIES`, `US_COMPANIES`, TI lists.
- §2 utilities (`setup_logging`, `safe_name`).
- §3 acquisition and raw loaders.
- §4 feature engineering, including the optional full indicator set and Pearson selection.
- §5 `to_supervised`, `windows_of`, `prepare_split`.

**CLI:**
`--market {saudi,us,both}`, `--companies`, `--data-dir`, `--saudi-csv`, `--us-dir`,
`--feature-selection {fixed,pearson}`, `--pearson-threshold`, `--out` (default `data/prepared`),
`--force`.

**Outputs**, one folder per company: `data/prepared/{market}/{company}/`

| File | Content | Used by |
|---|---|---|
| `features.parquet` | Calendar-daily, ffilled, selected TIs (incl. target), date index | RW, ARIMA (original-scale target), future reuse |
| `windows.npz` | `train_x, train_y, val_x, val_y, test_x, test_y` (scaled), `val_y_eval, test_y_eval` (original scale), `test_dates` (window start dates) | base models, ensembles, XGBoost, metrics |
| `scalers.joblib` | feature scaler + target scaler (fit on train only) | de-scaling predictions |
| `raw.parquet` | That company's raw rows, before reindexing | GRU(256) benchmark (it does its own preprocessing) |
| `meta.json` | feature list, target column, start date, row/window counts, config, library versions | sanity checks in script 2 |

Also writes `data/prepared/{market}/features.json` (the feature list; the Pearson result when
that option is used).

Note: XGBoost uses the same features *minus the target*. StandardScaler works column by column,
so script 2 can drop the target column from `train_x`/`test_x` and get exactly the arrays the
current code builds. No separate XGBoost windows are needed.

## Script 2: `train_report.py`

**Contains:**
- §1 `TrainCfg`/`RunCfg`; model-side constants (unit grid, `COMBOS`, `PAPER_*` tables).
- §6 metrics; §7 models; §8 ensembles; §9 baselines.
- §10 per-company experiment; §11 Wilcoxon; §12 reporting.
- Imports shared data-side constants (dates, `N_IN`/`N_OUT`, `SAUDI_COMPANIES`) from
  `prepare_data`. Do not duplicate them. `prepare_data.py` must keep its CLI under
  `if __name__ == "__main__":` so the import has no side effects.

**CLI:** every current training, reporting and comparison flag:
`--data` (default `data/prepared`), `--run-dir` (e.g. `runs/repo`), `--market`, `--companies`,
`--train-config`, `--stacking`, `--oof-folds`, `--units`, `--unit-grid`, `--fix-quirks`,
`--skip-baselines`, `--skip-wilcoxon`, `--wx-mode`, `--wx-models`, `--seed`, `--deterministic`,
`--save-models`, `--quick`, plus:
- `--reuse-base <run-dir>` — load saved base-model predictions instead of training. This makes
  `--stacking val/train` re-runs cheap. `oof` still has to train the fold models.
- `--report-only` — skip all training and rebuild metrics, tables and Wilcoxon from saved predictions.
- `--force` — redo companies that already have outputs (default: skip them; this replaces `--resume`).

**Outputs** under `runs/<name>/`:
- `base_preds/{company}.npz`: scaled predictions on train/val/test for BRNN, GRU, LSTM;
  train and inference times; chosen units.
- `preds/{company}.npz`: every model's original-scale predictions, with actuals and times
  per model. The benchmark keeps its own actuals.
- `models/{company}/*.keras` (only with `--save-models`).
- Everything the current script writes: `results_{market}_*.csv`, `wilcoxon_*.csv`, `run.log`,
  `unit_search.json`.
- `run_config.json`: args, the prepared-data path, and the hash of its `meta.json`.

## Acceptance criteria (parity with `tasi_edl_pipeline.py`)
Run on ADC with the repo defaults, the same seed and `--deterministic`:
- `features.parquet` equals `engineer_saudi(...)` exactly.
- `windows.npz` arrays equal `prepare_split(...)` exactly.
- Random Walk, ARIMA and XGBoost metrics are identical.
- Base models and ensembles match within seed noise (report any differences).
- `--reuse-base` with `--stacking train` produces stacking results without training any base model.
- `--report-only` reproduces the same CSVs from the saved predictions.

## Work order
1. Create `prepare_data.py` by moving §0 (data side), §2–§5 and the data part of `main`.
   Run it on ADC and check parity of the features and windows.
2. Create `train_report.py` by moving the rest. Change `run_company` to load from the prepared
   folder instead of computing the split.
3. Add base-prediction saving, `--reuse-base` and `--report-only`.
4. Run the parity check on ADC, then on the full Saudi set.
5. Keep `tasi_edl_pipeline.py` untouched until parity is confirmed.
