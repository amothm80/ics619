# ICS619 — Saudi Stock Price Forecasting with Ensemble Deep Learning

## Purpose
1. Understand the training-pipeline methodology in Baamer & Aljamaan (2026), *"Ensemble deep learning models for stock price forecasting: evidence from the Saudi markets"* (PeerJ Comput. Sci., DOI 10.7717/peerj-cs.4064).
2. Replicate the pipeline to end up with the same models, on an up-to-date dataset, with results as close to the paper's as possible.
3. Use the best model to build a price-forecasting app (technology TBD; the proposal suggests Streamlit + SQLite).

## Instructions
1. Code must be clear and concise. Avoid overly complicated scripts and programs.
2. Work through the methodology slowly, one step at a time. Implement and verify each step before moving to the next.
3. Prompts written for code should reflect this purpose and these instructions where applicable.

### Working conventions
- Python, pandas, NumPy, pandas-ta, scikit-learn, TensorFlow/Keras.
- Prefer small, single-purpose modules and functions over large notebooks.
- Fix random seeds. Save every fitted artifact (scalers, feature order, models, config) together per ticker.
- Never let validation or test data influence any fitted step (feature selection, scaling, model/unit selection, early stopping, meta-learner).

## Reference material
- Paper: `peerj-cs-4064.pdf` (project file).
- Proposal: `ICS619 - Project Proposal - ... - final.pdf` (project file).
- Authors' replication code: `I:\workspace\kfupm\ics619old\saudi-stock-forecasting-ensembles` (GitHub `ranabameer/saudi-stock-forecasting-ensembles`, Zenodo 10.5281/zenodo.19046571). Use it for reference only. It differs from the paper; see below.

---

## Paper overview

### Data
- Kaggle "Saudi Stock Exchange (Tadawul)" (Alzahrani, 2020). Daily data, roughly 2001 to 2020-03-05.
- Ten companies, one per sector:

| Sector | Trading name | Start |
|---|---|---|
| Industrials | ADC | 2001-12-31 |
| Financials | ALRAJHI | 2001-12-31 |
| Real estate | EMAAR EC | 2006-10-07 |
| Consumer discretionary | FITAIHI GROUP | 2002-01-26 |
| Consumer staples | GACO | 2002-01-01 |
| Energy | SARCO | 2002-01-23 |
| Utilities | SAUDI ELECTRICITY | 2002-06-08 |
| Healthcare | SPIMACO | 2001-12-31 |
| Communication services | STC | 2003-01-25 |
| Materials | TASNEE | 2001-12-31 |

- Raw columns: open, high, low, close, change, perc_change, volume_traded, value_traded, no_trades, plus identifiers.
- U.S. external check: AAPL, BAC, GOOG and JNJ over the same period with the same protocol.

### Pipeline (all steps per ticker)
1. **Clean.** Sort by date, forward-fill missing values.
2. **Feature engineering.** About 130 technical indicators from `pandas-ta` (length 5 where applicable).
3. **Feature selection.** Pearson correlation with Close, computed on the training split only. Keep indicators with |r| > 0.8 for all ten companies. This gives 34 features:
   `open high low close alma decay dema ema fwma hma hl2 hlc3 hwma jma kama linreg median midpoint midprice pwma quantile rma sinwma sma ssf swma t3 tema trima vwap vwma wcp wma zlma`
4. **Split (chronological).** Train up to 2015-12-31; validation 2016-01-01 to 2016-12-31; test 2017-01-01 to 2020-03-05.
5. **Scale.** `StandardScaler` fitted on train only, per ticker. Use separate scalers for features (X) and target (Close). Apply them unchanged to validation and test.
6. **Windowing.** Input is w=5 timesteps × 34 features, stride 1. Output is the next h=5 closes, predicted in one forward pass (direct multi-output, one model per ticker and architecture). A window sensitivity check over w ∈ {3, 5, 7, 10} found w=5 best (average RMSE 0.043).
7. **Base models.** BiRNN (B), GRU (G), LSTM (L): one recurrent layer followed by Dense(5). The best unit count per ticker comes from the authors' earlier work (Baamer & Aljamaan, 2025), selected on validation data.
8. **Training configuration (Table 8).**

| Setting | Value |
|---|---|
| Optimizer | Adam, lr 1e-4 |
| Loss | MSE |
| Activation | ReLU |
| Batch size | 10% of the training set |
| Epochs | 100 |
| Early stopping | patience 20; keep the lowest-val-loss weights |

9. **Averaging ensembles.** Equal-weight mean of base forecasts: AVG(B,G,L), AVG(B,G), AVG(B,L), AVG(G,L).
10. **Stacking ensembles.** 4 base combinations × 3 meta-learners = 12 per ticker:
    - ANN: Dense 128 → 64 → 32, ReLU.
    - GRU: 128 units.
    - LSTM: 128 units.
    - All meta-learners use Adam and MSE.
    - Meta-input is the base predictions, shaped (samples, 5 horizon steps, n_base_models).
    - **Rolling-origin out-of-fold (OOF):** train base models on expanding windows inside the training period and predict the next block. Those held-out predictions train the meta-learner. Then retrain the base models on the full training set and feed their test predictions to the meta-learner.
11. **Baselines.** Random walk with 5-day drift, univariate ARIMA, XGBoost on the same features, and a GRU-256 benchmark (Aseeri, 2023).
12. **Evaluation.**
    - Metrics: MAE, MSE, RMSE and MRE on inverse-scaled prices.
    - Significance: pairwise Wilcoxon signed-rank tests on RMSE over non-overlapping 5-step windows, Holm–Bonferroni corrected, α = 0.05, summarized as win/loss counts.
13. For every script created, updaet README.MD with description and arguments

### Headline results
- Ensembles beat the standalone models on every metric. GRU is the best standalone model.
- Wilcoxon win rates: AVG(B,G) 88.6%, AVG(B,G,L) 88.4%, STK_GRU(B,G) 87.7%, STK_GRU(B,G,L) 86.8%, GRU 82.6%, LSTM 62.7%, XGBoost 66.9%, Random Walk 30.8%, ARIMA 1.5%.
- Stacking (GRU-based meta-learner) gives the best raw accuracy. Averaging is nearly as good and much cheaper.
- The proposal picks **AVG(B,G)** as the initial app engine, with AVG(B,G,L) as the alternative. The final choice is made on validation results, then confirmed on the untouched test set.

---

## Paper vs. authors' code: known discrepancies
Follow the **paper's** protocol unless deliberately running a "code-faithful" comparison.

| Topic | Paper | Code |
|---|---|---|
| Rows | Trading days | Reindexed to calendar days (`freq="D"`) and forward-filled. Weekends and holidays become copied rows, so "5 days" means 5 calendar rows and some targets are trivial copies. |
| Base-model selection | Validation only | Sweeps 26 unit sizes (2–1024) and scores each on the **test** set. |
| Training config | 100 epochs, lr 1e-4, batch 10%, early stopping 20 | 32 epochs, default lr (1e-3), batch 5%, no early stopping |
| Stacking | Rolling-origin OOF | Meta-learner trained on in-sample base predictions over the training set (leakage). |
| Correlation threshold | 0.8 | `PearsonCorrelation.py` uses 0.5 and min-max normalization. |
| Indicators | — | `hwma` is computed with `ta.dema`. Warm-up NaNs are filled with 0 instead of dropped. |
| Bugs | — | BiRNN meta-learner uses an undefined `BRNN` and saves under the `_gru` name. Predictions are rounded to 2 decimals before scoring. |

### Decisions for this project
- Use trading-day rows only. Do not calendar-reindex.
- Forward-fill only true gaps in trading rows. Do not forward-fill volume onto non-trading days.
- Drop indicator warm-up rows instead of zero-filling them.
- Fit scalers per ticker on train only. Persist them with the models.
- Select unit sizes on validation loss.
- Expect paper-faithful results to be somewhat worse than the published tables. Report this honestly.

### Open questions
- **Data source.** Purpose item 2 asks for an up-to-date dataset, but the paper's Kaggle data ends 2020-03-05. Decide on a source for newer data and whether to (a) first replicate on the original period, then (b) extend.
- **Base-model unit sizes.** These are in Baamer & Aljamaan (2025), which isn't in hand yet.
- **App stack.** Proposal: Streamlit + SQLite. Still to be confirmed.
