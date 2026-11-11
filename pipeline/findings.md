# Hand-built pipeline — decisions and findings (for the report)

Running log of the step-by-step reimplementation in `pipeline/pipeline.ipynb`. Ticker for development: ALRAJHI (1120), Kaggle data to 2020-03-05. Split: train ≤ 2015-12-31, val 2016, test 2017-01-01 … 2020-03-05. All choices were made on train/val only and frozen before the test set was used once (see the end of this document).

## Decisions vs. the paper

| Step | Paper | This implementation | Why |
|---|---|---|---|
| 1 Clean | Forward-fill missing values (authors' code: reindex to calendar days, ffill all columns) | Default: trading-day rows only; zero-volume rows dropped; missing O/H/L filled with O = prev close, H = max(prev C, C), L = min(prev C, C). Option `calendar_ffill=True` reproduces the authors' code | ffilling yesterday's H/L can create impossible bars (35 of 36 GACO rows with missing high); zero-volume rows are suspensions, some with close = 0 |
| 2 Indicators | ~130 pandas-ta indicators, length 5 | pandas-ta-classic, every indicator with `length=5, lookahead=False`; 369 columns for ALRAJHI | `dpo` is centred by default (uses 3 future days); causality test (recompute without last 50 rows) → 0 failing columns |
| 3 Selection | abs(r) > 0.8 with close, train only, all 10 tickers → 34 | Same rule: 105 of 368 candidates pass; all 34 paper features pass (weakest hwma 0.984, alma 0.987); paper's 34 used | Newer library has many more price-level functions (bands, Heikin-Ashi, ln/sqrt/floor of price) |
| 5 Scale | StandardScaler, train only | Same; separate X (34 cols) and y (close) scalers per ticker | y scaler maps predictions back to riyals |
| 6 Windows | w = 5, h = 5, stride 1 | Same; windows built per split (no target crosses a split boundary) | ALRAJHI train: 3,680 windows (5,081 with calendar ffill) |
| 7 Training | Adam lr 1e-4, MSE, ReLU, batch 10%, 100 epochs, early stopping patience 20 | lr 1e-3, epochs ≤ 1000 with early stopping (patience 20, restore best), batch 10%, shuffle=True; `enable_op_determinism()` | With lr 1e-4 / 100 epochs early stopping never triggers (GRU-64 needed ~540 epochs). lr 1e-3: ~111 epochs, val MSE 0.001609 vs 0.001589. Without determinism, reruns with the same seed reordered the results table |
| 8 Units | From authors' earlier work | Sweep {16, 32, 64, 128, 256} × 3 seeds, choose by mean val MSE | Paper's unit sizes not available |
| 11 Stacking | Rolling-origin OOF (authors' code: in-sample meta-training, leaky) | Walk-forward OOF: first 40% of train windows never predicted; remaining 60% in 5 blocks; per block, fit on all earlier windows, early-stop on the last 250 windows before the block, 5-window purge gaps on both sides. Meta-learners trained on the full OOF set (2,208 windows), early-stopped on val using full-train base predictions | Purge needed because 5-day targets of the windows just before a block overlap the block's targets |
| 12 Baselines | Random walk with drift ("fixed to align with the five-day horizon"), univariate ARIMA, XGBoost on the same features, GRU-256 (Aseeri, 2023) | RW: last close + k × mean daily change on train. ARIMA(p,1,q), order by AIC on train over p, q ∈ {0,1,2}, parameters fixed, 5-step dynamic forecast from each window's last input day. XGBoost on flattened windows (5 × 34 = 170 inputs → 5 outputs), depth 4, lr 0.05, early stopping on val. GRU-256 with the same training config | ARIMA alignment verified: ARIMA(0,1,0) forecasts equal naive exactly, on val and on test |
| 13 Evaluation | MAE, MSE, RMSE, MRE on prices; pairwise Wilcoxon on RMSE over non-overlapping 5-step windows, Holm, α = 0.05, win/loss counts | Same. Models trained on train only (early stopping on val), not refit on train+val. Win rate = wins / (wins + losses); ties = not significant | Paper's exact win-rate denominator unclear, so raw win/loss/tie counts are reported too |

## Findings (ALRAJHI, validation 2016)

**Data quality.** Raw Kaggle bars violating low ≤ open/close ≤ high: ALRAJHI 568, GACO 671, SPIMACO 621 (mostly 2002–2014, off by 1–2 halalas). Left as-is.

**Naive baseline** (next 5 closes = last close): val MSE (scaled) 0.001562; RMSE 0.8662 SAR. Per-step RMSE (SAR) 0.488 / 0.708 / 0.845 / 0.992 / 1.147, growing like sqrt(h), the random-walk signature.

**Single GRU-64 (seed 42).** 100 epochs at lr 1e-4: val 0.002544 (not converged). Converged: 0.001589 (+1.7% vs naive). Per-step RMSE 0.517 / 0.722 / 0.854 / 0.984 / 1.153, worse than naive on day 1, within ~1% after.

**Unit sweep, trading days, mean of seeds 1–3 (val MSE):**

| Units | BiRNN | GRU | LSTM |
|---|---|---|---|
| 16 | 0.001962 | 0.001592 | 0.001585 |
| 32 | 0.001806 | 0.001619 | 0.001613 |
| 64 | 0.001767 | 0.001615 | 0.001647 |
| 128 | 0.001721 | 0.001596 | 0.001613 |
| 256 | 0.001744 | 0.001610 | 0.001648 |

Chosen: BiRNN-128, GRU-16, LSTM-16. Seed std (0.00001–0.00005) is the same order as the differences between sizes, so the unit choice is within noise for GRU/LSTM. A 100-epoch cap biased the choice toward large models (all picked 128), since small models converge slowly.

**Calendar ffill (authors' code) vs trading days**, mean of 3 seeds:

| | Trading days | Calendar ffill |
|---|---|---|
| Naive val MSE | 0.001562 | 0.001192 (−24%) |
| Best GRU | 16: +1.9% vs naive | 128: +2.3% |
| Best LSTM | 16: +1.5% | 16: +4.8% |
| Best BiRNN | 128: +10% | 128: +10.8% |

Calendar ffill lowers absolute errors only because weekend/holiday targets are copies of the previous close; the gap to naive does not improve. Likely contributes to the paper's lower published errors.

**Run-to-run variability (seed 42, before determinism was enabled), RMSE vs naive:**

| | Run 1 | Run 2 | Run 3 (deterministic) |
|---|---|---|---|
| G | −2.4% | −2.4% | −2.6% |
| AVG(B,G,L) | −2.7% | −1.7% | −2.6% |
| AVG(G,L) | −2.5% | −1.1% | −2.5% |
| AVG(B,G) | −2.6% | −2.7% | −2.3% |
| L | −0.8% | +2.6% | −0.5% |
| B | +2.9% | +2.3% | +3.3% |

Stable: GRU-16 always beats naive (~2.5%); BiRNN always worst; G and the averages containing G within ~0.3% of each other, so their order is noise. LSTM is the most seed-sensitive.

**Steps 9–12: full val table (deterministic run, RMSE in SAR):**

| Model | MAE | RMSE | MRE | RMSE vs naive |
|---|---|---|---|---|
| G (GRU-16) | 0.6155 | 0.8437 | 0.0172 | −2.6% |
| AVG(B,G,L) | 0.6134 | 0.8438 | 0.0171 | −2.6% |
| AVG(G,L) | 0.6155 | 0.8445 | 0.0172 | −2.5% |
| STK_ANN(B,G,L) | 0.6118 | 0.8449 | 0.0171 | −2.5% |
| STK_ANN(B,G) | 0.6144 | 0.8452 | 0.0171 | −2.4% |
| AVG(B,G) | 0.6159 | 0.8461 | 0.0172 | −2.3% |
| STK_ANN(G,L) | 0.6267 | 0.8495 | 0.0175 | −1.9% |
| AVG(B,L) | 0.6284 | 0.8583 | 0.0175 | −0.9% |
| L (LSTM-16) | 0.6341 | 0.8615 | 0.0177 | −0.5% |
| RW with drift | 0.6292 | 0.8637 | 0.0175 | −0.3% |
| Naive | 0.6316 | 0.8662 | 0.0176 | — |
| ARIMA(0,1,2) | 0.6327 | 0.8664 | 0.0176 | +0.0% |
| STK_ANN(B,L) | 0.6433 | 0.8733 | 0.0179 | +0.8% |
| GRU-256 (benchmark) | 0.6333 | 0.8765 | 0.0177 | +1.2% |
| STK_LSTM(B,G,L) | 0.6649 | 0.8791 | 0.0184 | +1.5% |
| STK_GRU(B,G) | 0.6388 | 0.8814 | 0.0177 | +1.8% |
| STK_LSTM(B,G) | 0.6729 | 0.8928 | 0.0185 | +3.1% |
| B (BiRNN-128) | 0.6641 | 0.8950 | 0.0185 | +3.3% |
| STK_LSTM(G,L) | 0.7032 | 0.9161 | 0.0194 | +5.8% |
| STK_GRU(B,L) | 0.6871 | 0.9268 | 0.0190 | +7.0% |
| STK_GRU(B,G,L) | 0.7250 | 0.9473 | 0.0199 | +9.4% |
| STK_LSTM(B,L) | 0.7282 | 0.9486 | 0.0200 | +9.5% |
| STK_GRU(G,L) | 0.7692 | 0.9900 | 0.0211 | +14.3% |
| XGBoost | 0.7515 | 1.0254 | 0.0210 | +18.4% |

**Baselines (Step 12).**
- Random walk with drift: drift = 0.000265 per day (scaled), estimated on 2002–2015. Practically equal to naive (−0.3%). The paper reports random walk as the worst model on most tickers; with a drift estimated on training data it is one of the stronger models here, so the paper's drift setting must have been much larger.
- ARIMA: AIC on train selects (0,1,2), AIC −10759.8. Forecasts are practically identical to naive (0.8664 vs 0.8662): close to a pure random walk.
- XGBoost: early stopping kept 124 trees. Worst model on val (+18% vs naive). The paper finds XGBoost better than random walk and ARIMA. Trees predict price levels as piecewise constants and cannot extrapolate; a target relative to the last close is a possible fairness variant.
- GRU-256 (Aseeri, 2023): stopped at 127 epochs, val 0.001600; worse than GRU-16 and naive. Consistent with the paper's statement that GRU-256 is beaten by most alternatives.
**Out-of-fold track record of the base models (RMSE in SAR per OOF block):**

| Block | Period | B | G | L | Naive |
|---|---|---|---|---|---|
| 0 | 2007-02 – 2008-11 | 3.502 | 3.834 | 3.501 | 2.615 |
| 1 | 2008-11 – 2010-08 | 1.561 | 1.547 | 1.694 | 1.358 |
| 2 | 2010-08 – 2012-06 | 0.971 | 0.905 | 0.917 | 0.889 |
| 3 | 2012-06 – 2014-03 | 0.805 | 0.830 | 0.779 | 0.734 |
| 4 | 2014-03 – 2015-12 | 1.234 | 1.220 | 1.196 | 1.166 |

Out of fold, no base model beats naive in any block. Fold 0's early-stopping year is the 2006 crash (ALRAJHI ~141 → 42, mean daily move 3.46 SAR vs ~0.40 in 2013–14), giving early-stopping losses ~0.13–0.15 vs ~0.0012 in later folds; scaled MSE is not comparable across periods.

**Stacking robustness check: meta-learners trained without OOF block 0 (2007–08), val RMSE:**

| | Full OOF | Without 2007–08 |
|---|---|---|
| STK_ANN(B,G,L) | 0.8449 | 0.8549 |
| STK_ANN(B,G) | 0.8452 | 0.8637 |
| STK_ANN(G,L) | 0.8495 | 0.8628 |
| STK_ANN(B,L) | 0.8733 | 0.8920 |
| STK_GRU(B,G) | 0.8814 | 0.8651 |
| STK_GRU(B,G,L) | 0.9473 | 0.8976 |
| STK_GRU(G,L) | 0.9900 | 0.8958 |
| STK_GRU(B,L) | 0.9268 | 0.9082 |
| STK_LSTM(B,G,L) | 0.8791 | 0.8666 |
| STK_LSTM(B,G) | 0.8928 | 0.8685 |
| STK_LSTM(G,L) | 0.9161 | 0.8954 |
| STK_LSTM(B,L) | 0.9486 | 0.9151 |

All recurrent meta-learners improve without the crisis years, but the best (STK_GRU(B,G), 0.8651) only ties naive (0.8662); all ANN meta-learners get worse. Decision (before test): keep the full OOF set.

**Overall on val.** Ranking GRU ≈ LSTM > BiRNN (as in the paper). GRU-16 alone, the averages containing G, and the ANN meta-learner all beat naive by ~2.5% RMSE and are within noise of each other. With honest OOF meta-training, stacking is at best equal to averaging (ANN meta-learner); GRU/LSTM meta-learners are worse than naive. Random walk with drift and ARIMA are effectively naive; XGBoost and GRU-256 are worse than naive.

## Test set (ALRAJHI, 2017-01-01 … 2020-03-05), used once

Setup frozen before test: base models BiRNN-128, GRU-16, LSTM-16 (seed 42), meta-learners on the full OOF set, baselines as above. A first test run accidentally used the trimmed (without 2007–08) meta-learners because a variable was overwritten; it was rerun with the pre-chosen full-OOF meta-learners. No setting was changed after seeing test results.

**Test metrics (SAR):**

| Model | MAE | RMSE | MRE | RMSE vs naive |
|---|---|---|---|---|
| ARIMA(0,1,2) | 0.9025 | 1.2791 | 0.0162 | −0.1% |
| RW with drift | 0.9026 | 1.2791 | 0.0162 | −0.1% |
| Naive | 0.9027 | 1.2802 | 0.0162 | — |
| G (GRU-16) | 0.9251 | 1.2958 | 0.0166 | +1.2% |
| AVG(G,L) | 0.9323 | 1.3074 | 0.0168 | +2.1% |
| AVG(B,G,L) | 0.9549 | 1.3293 | 0.0171 | +3.8% |
| AVG(B,G) | 0.9622 | 1.3355 | 0.0173 | +4.3% |
| L (LSTM-16) | 0.9725 | 1.3587 | 0.0174 | +6.1% |
| AVG(B,L) | 0.9982 | 1.3786 | 0.0179 | +7.7% |
| GRU-256 | 1.0253 | 1.4370 | 0.0183 | +12.2% |
| B (BiRNN-128) | 1.0751 | 1.4647 | 0.0192 | +14.4% |
| XGBoost | 1.1005 | 1.5563 | 0.0196 | +21.6% |
| STK_ANN(B,G,L) | 1.2497 | 1.7487 | 0.0216 | +36.6% |
| STK_ANN(B,L) | 1.2685 | 1.7634 | 0.0220 | +37.7% |
| STK_GRU(B,G) | 1.2569 | 1.7711 | 0.0215 | +38.3% |
| STK_GRU(B,L) | 1.2754 | 1.7803 | 0.0220 | +39.1% |
| STK_ANN(B,G) | 1.2807 | 1.7867 | 0.0219 | +39.6% |
| STK_LSTM(B,L) | 1.4399 | 1.8960 | 0.0251 | +48.1% |
| STK_ANN(G,L) | 1.4520 | 1.9805 | 0.0246 | +54.7% |
| STK_LSTM(B,G) | 1.5182 | 2.0077 | 0.0262 | +56.8% |
| STK_LSTM(B,G,L) | 1.5717 | 2.0567 | 0.0272 | +60.7% |
| STK_GRU(B,G,L) | 1.7473 | 2.3172 | 0.0295 | +81.0% |
| STK_LSTM(G,L) | 1.8044 | 2.3308 | 0.0308 | +82.1% |
| STK_GRU(G,L) | 1.7976 | 2.3788 | 0.0303 | +85.8% |

**Wilcoxon signed-rank on per-window RMSE (non-overlapping 5-day windows), Holm, α = 0.05; 24 models, 23 comparisons each:**

| Model | Win | Loss | Tie | Win rate |
|---|---|---|---|---|
| Naive | 20 | 0 | 3 | 100% |
| RW with drift | 20 | 0 | 3 | 100% |
| ARIMA | 17 | 0 | 6 | 100% |
| G | 11 | 2 | 10 | 85% |
| AVG(G,L) | 10 | 0 | 13 | 100% |
| AVG(B,G,L) | 10 | 2 | 11 | 83% |
| AVG(B,G) | 9 | 2 | 12 | 82% |
| L | 9 | 3 | 11 | 75% |
| AVG(B,L) | 8 | 5 | 10 | 62% |
| STK_GRU(B,G) | 7 | 3 | 13 | 70% |
| STK_GRU(B,L) | 7 | 3 | 13 | 70% |
| STK_ANN(B,G,L) | 7 | 4 | 12 | 64% |
| STK_ANN(B,G) | 7 | 4 | 12 | 64% |
| STK_ANN(B,L) | 7 | 4 | 12 | 64% |
| XGBoost | 6 | 3 | 14 | 67% |
| GRU-256 | 6 | 7 | 10 | 46% |
| B | 5 | 9 | 9 | 36% |
| STK_LSTM(B,L) | 5 | 16 | 2 | 24% |
| STK_LSTM(B,G) | 4 | 18 | 1 | 18% |
| STK_ANN(G,L) | 3 | 14 | 6 | 18% |
| STK_LSTM(B,G,L) | 1 | 19 | 3 | 5% |
| STK_GRU(B,G,L) | 0 | 20 | 3 | 0% |
| STK_GRU(G,L) | 0 | 20 | 3 | 0% |
| STK_LSTM(G,L) | 0 | 21 | 2 | 0% |

Trimmed meta-learners (robustness variant, accidentally evaluated first): STK_ANN 1.46–1.72 RMSE, STK_LSTM 1.79–2.04, STK_GRU 4.61–5.63 (MRE 5.5–6.7%): the recurrent meta-learners trained without 2007–08 collapse on test.

**Conclusions (ALRAJHI).**
- No model beats naive on test. Naive, random walk with drift and ARIMA are never significantly beaten (0 losses). On val, GRU-16 and the averages beat naive by ~2.5%; on test they are 1–4% worse. The val advantage came from selection on val, a favourable seed, and differences smaller than run-to-run noise.
- The relative ranking of the neural models holds: G > L > B; averages that include G are the best ensembles; GRU-256 and BiRNN are near the bottom.
- With leak-free OOF meta-training, every stacked model is far worse than naive on test (+37% to +86% RMSE), although STK_ANN matched the averages on val. Stacking does not transfer from 2016 to 2017–2020.
- This is the opposite of the paper's headline (ensembles beat everything; random walk wins 30.8%, ARIMA 1.5%). Likely contributors in the paper: in-sample (leaky) meta-training, calendar-day forward-fill, unit sizes scored on test, a large fixed random-walk drift.
- Next: run the whole chain for all ten tickers to see whether ALRAJHI is typical (done for Kaggle and Yahoo data, see below).

## All ten tickers, Kaggle data (test 2017-01-01 … 2020-03-05)

`pipeline/pipeline.py` (trading days), results in `pipeline/results/`. Same frozen setup as the ALRAJHI run (BiRNN-128, GRU-16, LSTM-16, seed 42, full-OOF stacking); unit sizes from ALRAJHI reused for all tickers. 26 models: the 24 above plus the authors' code-faithful RW (paper) and ARIMA (paper), defined in the Yahoo section below.

**Test RMSE (SAR), all models.** Sorted by mean RMSE relative to naive; "Beat naive" counts stocks with lower RMSE than naive.

| Model | ADC | ALRAJHI | EMAAR EC | FITAIHI | GACO | SARCO | SEC | SPIMACO | STC | TASNEE | Mean vs naive | Beat naive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive | 0.2954 | 1.2802 | 0.3920 | 0.2325 | 0.3162 | 1.5296 | 0.5525 | 0.5975 | 2.3100 | 0.5364 | — | — |
| RW drift | 0.2956 | 1.2791 | 0.3910 | 0.2327 | 0.3178 | 1.5304 | 0.5528 | 0.5990 | 2.3095 | 0.5367 | +0.1% | 3 |
| ARIMA | 0.3016 | 1.2791 | 0.3916 | 0.2334 | 0.3347 | 1.5310 | 0.5572 | 0.6193 | 2.3111 | 0.5385 | +1.3% | 2 |
| AVG(B,G,L) | 0.3069 | 1.3293 | 0.4154 | 0.2619 | 0.3671 | 1.6519 | 0.5817 | 0.6300 | 2.3449 | 0.6134 | +7.7% | 0 |
| AVG(B,L) | 0.3077 | 1.3786 | 0.4273 | 0.2572 | 0.3586 | 1.6408 | 0.5946 | 0.6391 | 2.3486 | 0.6180 | +8.4% | 0 |
| AVG(B,G) | 0.3080 | 1.3355 | 0.4132 | 0.2788 | 0.3620 | 1.6493 | 0.5870 | 0.6300 | 2.4237 | 0.6154 | +8.7% | 0 |
| AVG(G,L) | 0.3162 | 1.3074 | 0.4130 | 0.2572 | 0.4220 | 1.7056 | 0.5832 | 0.6345 | 2.3512 | 0.6146 | +9.8% | 0 |
| STK_ANN(B,G,L) | 0.3042 | 1.7487 | 0.4109 | 0.2495 | 0.3351 | 1.6201 | 0.7225 | 0.6605 | 2.4690 | 0.5768 | +11.9% | 0 |
| L | 0.3149 | 1.3587 | 0.4326 | 0.2496 | 0.4532 | 1.6976 | 0.6007 | 0.6529 | 2.3750 | 0.6224 | +12.2% | 0 |
| STK_ANN(B,G) | 0.3097 | 1.7867 | 0.4115 | 0.2404 | 0.3334 | 1.6194 | 0.7710 | 0.6365 | 2.5610 | 0.5630 | +12.6% | 0 |
| G | 0.3346 | 1.2958 | 0.4053 | 0.2829 | 0.4506 | 1.7811 | 0.5760 | 0.6332 | 2.3996 | 0.6177 | +12.8% | 0 |
| B | 0.3148 | 1.4647 | 0.4371 | 0.2840 | 0.3594 | 1.6345 | 0.6442 | 0.6573 | 2.6052 | 0.6312 | +13.2% | 0 |
| STK_ANN(B,L) | 0.3329 | 1.7634 | 0.4139 | 0.2371 | 0.3426 | 1.7051 | 0.7295 | 0.6480 | 2.5937 | 0.5798 | +13.9% | 0 |
| GRU-256 | 0.2996 | 1.4370 | 0.4194 | 0.3667 | 0.4017 | 1.5979 | 0.6016 | 0.6436 | 2.4578 | 0.6313 | +15.1% | 0 |
| STK_ANN(G,L) | 0.3503 | 1.9805 | 0.4087 | 0.2643 | 0.3663 | 1.6661 | 0.8576 | 0.6226 | 3.0934 | 0.5894 | +21.9% | 0 |
| STK_GRU(B,G,L) | 0.3344 | 2.3172 | 0.4467 | 0.2327 | 0.3891 | 1.7989 | 0.8121 | 0.6354 | 3.6372 | 0.6066 | +27.3% | 0 |
| STK_LSTM(G,L) | 0.3521 | 2.3308 | 0.4270 | 0.2373 | 0.4111 | 1.8226 | 0.8552 | 0.6503 | 3.3939 | 0.5819 | +28.0% | 0 |
| XGBoost | 0.3334 | 1.5563 | 0.4444 | 0.2629 | 0.8189 | 1.7155 | 0.6190 | 0.6715 | 2.8834 | 0.6440 | +30.1% | 0 |
| STK_GRU(B,G) | 0.3071 | 1.7711 | 0.4389 | 0.2328 | 0.4886 | 1.8186 | 0.8957 | 0.6106 | 4.9585 | 0.5688 | +31.3% | 0 |
| STK_GRU(B,L) | 0.3405 | 1.7803 | 0.4438 | 0.2311 | 0.5804 | 1.8161 | 0.9072 | 0.6178 | 4.1994 | 0.5852 | +32.8% | 1 |
| STK_LSTM(B,G,L) | 0.3205 | 2.0567 | 0.4337 | 0.2334 | 0.3861 | 1.6314 | 1.0924 | 0.6414 | 5.0245 | 0.5736 | +33.8% | 0 |
| STK_LSTM(B,L) | 0.3608 | 1.8960 | 0.4321 | 0.2358 | 0.3683 | 1.6254 | 0.8062 | 0.6347 | 6.8996 | 0.5732 | +36.2% | 0 |
| STK_LSTM(B,G) | 0.3101 | 2.0077 | 0.4396 | 0.2372 | 0.4225 | 1.7143 | 0.8379 | 0.6320 | 6.4846 | 0.5735 | +36.7% | 0 |
| RW (paper) | 0.4242 | 1.7667 | 0.5310 | 0.3256 | 0.4567 | 2.1085 | 0.7723 | 0.8608 | 3.3698 | 0.7419 | +40.7% | 0 |
| STK_GRU(G,L) | 0.3491 | 2.3788 | 0.4240 | 0.2364 | 0.5277 | 1.9551 | 0.8642 | 0.6239 | 5.4430 | 0.6012 | +41.7% | 0 |
| ARIMA (paper) | 1.7501 | 23.03 | 3.0867 | 3.2284 | 10.49 | 5.6948 | 5.3057 | 4.6262 | 22.53 | 6.3636 | +1115.4% | 0 |

Paper's best RMSE per ticker (Table 10, calendar-filled data, not directly comparable): ADC 0.2498, ALRAJHI 1.0688, EMAAR EC 0.3259, FITAIHI 0.1908, GACO 0.2637, SARCO 1.2307, SEC 0.4651, SPIMACO 0.5192, STC 1.9293, TASNEE 0.4401.

**Wilcoxon win–loss–tie per stock** (signed-rank on per-window RMSE, non-overlapping 5-day windows, Holm within each stock, α = 0.05; 25 comparisons per model per stock, 250 pooled). Win rate = wins / (wins + losses).

| Model | ADC | ALRAJHI | EMAAR EC | FITAIHI | GACO | SARCO | SEC | SPIMACO | STC | TASNEE | Pooled W–L–T | Win rate | Paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive | 18–0–7 | 22–0–3 | 21–0–4 | 12–0–13 | 23–0–2 | 22–0–3 | 23–0–2 | 5–0–20 | 19–0–6 | 18–0–7 | 183–0–67 | 100.0% |  |
| RW drift | 18–0–7 | 22–0–3 | 23–0–2 | 12–0–13 | 23–0–2 | 21–0–4 | 23–0–2 | 5–0–20 | 19–0–6 | 16–0–9 | 182–0–68 | 100.0% |  |
| ARIMA | 13–0–12 | 18–0–7 | 21–0–4 | 12–0–13 | 21–0–4 | 16–0–9 | 23–0–2 | 2–0–23 | 19–0–6 | 13–0–12 | 158–0–92 | 100.0% |  |
| STK_ANN(B,G) | 12–1–12 | 8–4–13 | 2–3–20 | 13–0–12 | 18–2–5 | 7–1–17 | 6–14–5 | 2–5–18 | 10–3–12 | 10–0–15 | 88–33–129 | 72.7% |  |
| STK_ANN(B,G,L) | 19–0–6 | 8–4–13 | 3–1–21 | 10–8–7 | 17–2–6 | 6–2–17 | 8–11–6 | 2–13–10 | 10–3–12 | 9–2–14 | 92–46–112 | 66.7% |  |
| AVG(B,G,L) | 10–3–12 | 11–2–12 | 6–3–16 | 5–15–5 | 10–6–9 | 7–3–15 | 16–3–6 | 3–0–22 | 15–0–10 | 1–8–16 | 84–43–123 | 66.1% | 88.4% |
| AVG(B,G) | 7–3–15 | 10–2–13 | 6–3–16 | 3–19–3 | 12–4–9 | 8–2–15 | 14–3–8 | 2–0–23 | 12–0–13 | 2–8–15 | 76–44–130 | 63.3% | 88.6% |
| AVG(B,L) | 11–3–11 | 9–5–11 | 1–6–18 | 6–14–5 | 10–5–10 | 7–3–15 | 14–4–7 | 2–0–23 | 14–0–11 | 1–8–16 | 75–48–127 | 61.0% |  |
| STK_ANN(B,L) | 2–7–16 | 8–4–13 | 2–3–20 | 11–4–10 | 11–3–11 | 1–9–15 | 8–9–8 | 3–0–22 | 10–8–7 | 1–1–23 | 57–48–145 | 54.3% |  |
| GRU-256 | 15–0–10 | 6–6–13 | 3–3–19 | 1–24–0 | 5–8–12 | 15–0–10 | 12–6–7 | 2–0–23 | 10–4–11 | 1–8–16 | 70–59–121 | 54.3% |  |
| AVG(G,L) | 3–8–14 | 11–0–14 | 4–4–17 | 5–14–6 | 6–13–6 | 2–6–17 | 16–3–6 | 3–0–22 | 12–0–13 | 2–8–15 | 64–56–130 | 53.3% |  |
| STK_LSTM(B,G) | 11–0–14 | 5–17–3 | 1–3–21 | 13–0–12 | 6–15–4 | 6–4–15 | 3–12–10 | 4–0–21 | 2–15–8 | 9–2–14 | 60–68–122 | 46.9% |  |
| B | 7–4–14 | 5–8–12 | 1–8–16 | 2–20–3 | 12–3–10 | 8–2–15 | 11–9–5 | 2–1–22 | 8–6–11 | 1–9–15 | 57–70–123 | 44.9% |  |
| STK_GRU(B,G) | 12–0–13 | 8–2–15 | 1–3–21 | 14–0–11 | 4–18–3 | 2–11–12 | 1–18–6 | 4–0–21 | 2–21–2 | 9–0–16 | 57–73–120 | 43.8% | 87.7% |
| STK_LSTM(B,G,L) | 5–10–10 | 2–19–4 | 2–3–20 | 14–0–11 | 8–5–12 | 6–3–16 | 1–18–6 | 3–0–22 | 5–14–6 | 9–0–16 | 55–72–123 | 43.3% |  |
| G | 1–15–9 | 12–2–11 | 8–3–14 | 2–19–4 | 5–14–6 | 1–10–14 | 16–3–6 | 2–2–21 | 11–3–11 | 1–8–16 | 59–79–112 | 42.8% | 82.6% |
| STK_LSTM(B,L) | 2–13–10 | 6–15–4 | 2–3–20 | 11–0–14 | 10–6–9 | 6–4–15 | 5–12–8 | 3–0–22 | 1–16–8 | 1–0–24 | 47–69–134 | 40.5% |  |
| L | 5–7–13 | 8–3–14 | 1–9–15 | 5–13–7 | 3–20–2 | 1–7–17 | 14–3–8 | 2–1–22 | 11–3–11 | 1–9–15 | 51–75–124 | 40.5% | 62.7% |
| STK_GRU(B,G,L) | 3–7–15 | 1–20–4 | 1–7–17 | 14–0–11 | 10–6–9 | 2–14–9 | 4–15–6 | 3–0–22 | 4–14–7 | 1–4–20 | 43–87–120 | 33.1% | 86.8% |
| STK_ANN(G,L) | 1–19–5 | 4–14–7 | 2–1–22 | 5–14–6 | 11–5–9 | 6–3–16 | 3–15–7 | 4–0–21 | 5–13–7 | 1–3–21 | 42–87–121 | 32.6% |  |
| XGBoost | 1–13–11 | 7–3–15 | 1–3–21 | 2–10–13 | 1–24–0 | 1–4–20 | 12–3–10 | 2–0–23 | 3–6–16 | 1–3–21 | 31–69–150 | 31.0% | 66.9% |
| STK_GRU(B,L) | 2–8–15 | 8–3–14 | 1–7–17 | 14–0–11 | 2–23–0 | 1–12–12 | 1–19–5 | 5–0–20 | 3–16–6 | 1–0–24 | 38–88–124 | 30.2% |  |
| STK_LSTM(G,L) | 1–15–9 | 1–22–2 | 1–3–21 | 12–0–13 | 7–13–5 | 2–10–13 | 2–12–11 | 2–0–23 | 3–14–8 | 1–1–23 | 32–90–128 | 26.2% |  |
| RW (paper) | 1–11–13 | 3–3–19 | 1–9–15 | 2–14–9 | 6–3–16 | 1–4–20 | 2–5–18 | 1–24–0 | 3–6–16 | 1–3–21 | 21–82–147 | 20.4% | 30.8% |
| STK_GRU(G,L) | 1–9–15 | 1–21–3 | 1–3–21 | 13–0–12 | 3–21–1 | 1–17–7 | 1–17–7 | 3–0–22 | 1–22–2 | 1–2–22 | 26–112–112 | 18.8% |  |
| ARIMA (paper) | 0–25–0 | 0–25–0 | 0–25–0 | 0–25–0 | 0–25–0 | 0–25–0 | 0–25–0 | 0–25–0 | 0–25–0 | 0–25–0 | 0–250–0 | 0.0% | 1.5% |

**Conclusions (Kaggle, ten tickers).**
- No model beats naive on average. Only per-stock exception: STK_GRU(B,L) on FITAIHI (0.2311 vs 0.2325).
- Naive, RW with drift and ARIMA are never significantly beaten (0 losses in 250).
- Best learned models: the averages (+7.7% to +9.8% vs naive). Stacking: +11.9% (STK_ANN(B,G,L)) to +41.7%.
- The authors' baselines behave as in the paper: RW (paper) 20.4% win rate (paper 30.8%), ARIMA (paper) 0% (paper 1.5%).
- Same picture as the Yahoo run below.

## All ten tickers, Yahoo Finance data (test 2024-01-01 … 2026-10-28)

`pipeline/pipeline_yahoo.py`, results in `pipeline/results_yahoo/`. Same frozen setup as above (BiRNN-128, GRU-16, LSTM-16, seed 42, full-OOF stacking). Only the data changed:

| | Kaggle run | Yahoo run |
|---|---|---|
| Source | Kaggle Tadawul CSV | Yahoo `<symbol>.SR`, Close (split-adjusted, not dividend-adjusted), cached in `data/yfinance/` |
| Start | 2001 | 2013-06-29 (Yahoo's Saudi history is unreliable before the Thu-Fri → Fri-Sat weekend change) |
| Train / val / test | ≤ 2015 / 2016 / 2017 – 2020-03-05 | ≤ 2022 / 2023 / 2024 – 2026-10-28 |
| Fixes | — | GACO 10:6 capital reduction (2017-02-14) applied by hand; Yahoo has no split record for it |

26 models: the 24 above plus the authors' code-faithful baselines. RW (paper): drift = mean change over the window's last 5 closes. ARIMA (paper): ARIMA(5,1,0) fitted on train, one 5-step forecast from the end of train, repeated for every test window.

**Test RMSE (SAR), all models.** Sorted by mean RMSE relative to naive; "Beat naive" counts stocks with lower RMSE than naive.

| Model | ADC | ALRAJHI | EMAAR EC | FITAIHI | GACO | SARCO | SEC | SPIMACO | STC | TASNEE | Mean vs naive | Beat naive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RW drift | 0.7261 | 1.3558 | 0.4905 | 0.1156 | 0.4804 | 1.9472 | 0.3564 | 1.1291 | 0.7009 | 0.2935 | 0.0% | 5 |
| Naive | 0.7261 | 1.3552 | 0.4914 | 0.1156 | 0.4810 | 1.9453 | 0.3552 | 1.1298 | 0.7003 | 0.2946 | — | — |
| ARIMA | 0.7243 | 1.3598 | 0.4913 | 0.1159 | 0.4823 | 1.9599 | 0.3588 | 1.1296 | 0.7010 | 0.2952 | +0.3% | 3 |
| AVG(B,G) | 0.7675 | 1.3704 | 0.5984 | 0.1161 | 0.5512 | 2.0620 | 0.3552 | 1.1615 | 0.7253 | 0.3276 | +6.7% | 0 |
| AVG(B,G,L) | 0.7597 | 1.3588 | 0.6451 | 0.1162 | 0.5477 | 2.0489 | 0.3520 | 1.1587 | 0.7155 | 0.3337 | +7.3% | 1 |
| G | 0.7579 | 1.3779 | 0.6551 | 0.1169 | 0.5530 | 2.0526 | 0.3654 | 1.1743 | 0.7104 | 0.3488 | +8.8% | 0 |
| AVG(B,L) | 0.7707 | 1.3766 | 0.6679 | 0.1169 | 0.5540 | 2.0981 | 0.3527 | 1.1636 | 0.7324 | 0.3335 | +8.8% | 1 |
| AVG(G,L) | 0.7529 | 1.3723 | 0.7152 | 0.1169 | 0.5480 | 2.0428 | 0.3540 | 1.1646 | 0.7033 | 0.3472 | +9.2% | 1 |
| B | 0.8028 | 1.4491 | 0.5697 | 0.1189 | 0.5684 | 2.2261 | 0.3574 | 1.1809 | 0.7768 | 0.3237 | +9.5% | 0 |
| L | 0.7655 | 1.4178 | 0.8739 | 0.1184 | 0.5612 | 2.0876 | 0.3562 | 1.1751 | 0.7165 | 0.3579 | +14.2% | 0 |
| GRU-256 | 0.7843 | 1.4386 | 1.0379 | 0.1169 | 0.5529 | 2.0661 | 0.3515 | 1.1552 | 0.7218 | 0.3131 | +15.8% | 1 |
| STK_ANN(G,L) | 0.7481 | 1.8591 | 0.5528 | 0.1200 | 0.5355 | 3.0518 | 0.3621 | 1.1671 | 0.7212 | 0.4059 | +17.1% | 0 |
| STK_ANN(B,L) | 0.7765 | 1.7810 | 0.5381 | 0.1190 | 0.5653 | 4.3773 | 0.3557 | 1.1602 | 0.7198 | 0.4285 | +24.4% | 0 |
| STK_ANN(B,G) | 0.7630 | 3.3262 | 0.5624 | 0.1186 | 0.5651 | 2.4519 | 0.3554 | 1.1620 | 0.7206 | 0.4039 | +25.4% | 0 |
| STK_ANN(B,G,L) | 0.7573 | 2.5598 | 0.6154 | 0.1186 | 0.5553 | 3.7894 | 0.3593 | 1.1784 | 0.7212 | 0.3749 | +26.7% | 0 |
| STK_GRU(B,G) | 0.7964 | 3.1880 | 1.1331 | 0.1165 | 0.5815 | 2.6669 | 0.3712 | 1.1823 | 0.7294 | 0.4173 | +38.9% | 0 |
| RW (paper) | 1.0089 | 1.9559 | 0.6933 | 0.1598 | 0.6811 | 2.7608 | 0.4936 | 1.5629 | 0.9819 | 0.4072 | +40.2% | 0 |
| STK_GRU(B,L) | 0.7728 | 3.5420 | 2.4348 | 0.1155 | 0.6246 | 2.4536 | 0.3697 | 1.2317 | 0.7382 | 0.4350 | +68.5% | 1 |
| XGBoost | 0.8116 | 1.7146 | 3.1704 | 0.1334 | 0.5702 | 2.1841 | 0.4313 | 1.1932 | 0.7745 | 0.3819 | +69.7% | 0 |
| STK_GRU(G,L) | 0.7907 | 3.2456 | 2.8243 | 0.1189 | 0.5929 | 2.6980 | 0.3688 | 1.2290 | 0.7443 | 0.4335 | +75.4% | 0 |
| STK_LSTM(B,G) | 0.7658 | 3.2794 | 3.4451 | 0.1157 | 0.5635 | 2.9341 | 0.3716 | 1.1533 | 0.7269 | 0.3759 | +85.5% | 0 |
| STK_GRU(B,G,L) | 0.7730 | 8.9092 | 1.5472 | 0.1164 | 0.6357 | 2.4424 | 0.3630 | 1.2151 | 0.7220 | 0.4224 | +89.3% | 0 |
| STK_LSTM(G,L) | 0.7823 | 13.89 | 5.6499 | 0.1180 | 0.5897 | 2.6134 | 0.3748 | 1.1730 | 0.7502 | 0.3876 | +208.9% | 0 |
| STK_LSTM(B,G,L) | 0.7670 | 16.75 | 4.5891 | 0.1227 | 0.5555 | 2.6953 | 0.3800 | 1.1650 | 0.8901 | 0.3876 | +210.5% | 0 |
| STK_LSTM(B,L) | 0.7653 | 17.92 | 5.7777 | 0.1191 | 0.6264 | 2.8794 | 0.3820 | 1.4132 | 0.7488 | 0.3948 | +245.8% | 0 |
| ARIMA (paper) | 3.8543 | 13.72 | 5.4755 | 0.8216 | 3.5239 | 15.80 | 6.9502 | 9.3467 | 6.0675 | 2.4783 | +840.4% | 0 |

**Wilcoxon win–loss–tie per stock** (signed-rank on per-window RMSE, non-overlapping 5-day windows, Holm within each stock, α = 0.05; 25 comparisons per model per stock, 250 pooled). Win rate = wins / (wins + losses).

| Model | ADC | ALRAJHI | EMAAR EC | FITAIHI | GACO | SARCO | SEC | SPIMACO | STC | TASNEE | Pooled W–L–T | Win rate | Paper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive | 7–0–18 | 17–0–8 | 23–0–2 | 3–0–22 | 21–0–4 | 22–0–3 | 4–0–21 | 7–0–18 | 4–0–21 | 23–0–2 | 131–0–119 | 100.0% |  |
| ARIMA | 8–0–17 | 17–0–8 | 23–0–2 | 3–0–22 | 20–0–5 | 22–0–3 | 4–0–21 | 6–0–19 | 4–0–21 | 23–0–2 | 130–0–120 | 100.0% |  |
| RW drift | 7–0–18 | 17–0–8 | 23–0–2 | 3–0–22 | 21–0–4 | 22–0–3 | 4–0–21 | 6–0–19 | 4–0–21 | 23–0–2 | 130–0–120 | 100.0% |  |
| AVG(B,G,L) | 4–0–21 | 18–0–7 | 14–8–3 | 3–0–22 | 5–3–17 | 15–3–7 | 4–0–21 | 4–0–21 | 6–0–19 | 15–5–5 | 88–19–143 | 82.2% | 88.4% |
| AVG(B,G) | 2–1–22 | 16–0–9 | 16–6–3 | 3–0–22 | 4–2–19 | 14–3–8 | 4–0–21 | 4–0–21 | 4–1–20 | 18–4–3 | 85–17–148 | 83.3% | 88.6% |
| AVG(G,L) | 4–0–21 | 16–0–9 | 12–13–0 | 3–0–22 | 5–3–17 | 15–3–7 | 4–0–21 | 4–0–21 | 5–0–20 | 14–6–5 | 82–25–143 | 76.6% |  |
| AVG(B,L) | 2–0–23 | 15–1–9 | 13–9–3 | 3–0–22 | 2–3–20 | 13–4–8 | 4–0–21 | 4–0–21 | 4–1–20 | 14–5–6 | 74–23–153 | 76.3% |  |
| GRU-256 | 1–5–19 | 13–0–12 | 9–15–1 | 2–0–23 | 2–3–20 | 14–3–8 | 4–0–21 | 4–0–21 | 3–0–22 | 21–3–1 | 73–29–148 | 71.6% |  |
| G | 1–4–20 | 15–0–10 | 13–8–4 | 1–0–24 | 3–3–19 | 14–0–11 | 2–13–10 | 3–0–22 | 3–1–21 | 13–5–7 | 68–34–148 | 66.7% | 82.6% |
| L | 1–0–24 | 14–5–6 | 11–14–0 | 3–0–22 | 2–5–18 | 13–4–8 | 3–0–22 | 4–1–20 | 4–0–21 | 11–9–5 | 66–38–146 | 63.5% | 62.7% |
| B | 1–7–17 | 13–5–7 | 17–3–5 | 1–0–24 | 1–4–20 | 7–9–9 | 4–0–21 | 4–0–21 | 2–14–9 | 14–4–7 | 64–46–140 | 58.2% |  |
| STK_ANN(G,L) | 1–0–24 | 11–11–3 | 17–3–5 | 1–0–24 | 8–0–17 | 3–15–7 | 2–0–23 | 4–0–21 | 3–0–22 | 5–13–7 | 55–42–153 | 56.7% |  |
| STK_ANN(B,G) | 1–0–24 | 5–17–3 | 17–4–4 | 1–0–24 | 1–4–20 | 7–9–9 | 4–0–21 | 5–0–20 | 3–0–22 | 6–13–6 | 50–47–153 | 51.5% |  |
| STK_ANN(B,G,L) | 1–0–24 | 9–15–1 | 13–5–7 | 2–0–23 | 1–5–19 | 2–18–5 | 4–0–21 | 5–0–20 | 4–0–21 | 8–10–7 | 49–53–148 | 48.0% |  |
| STK_ANN(B,L) | 1–3–21 | 11–11–3 | 19–3–3 | 1–0–24 | 1–4–20 | 1–23–1 | 4–0–21 | 4–0–21 | 4–0–21 | 2–19–4 | 48–63–139 | 43.2% |  |
| STK_LSTM(B,G) | 1–0–24 | 6–15–4 | 4–21–0 | 4–0–21 | 3–3–19 | 1–18–6 | 3–0–22 | 5–0–20 | 4–0–21 | 11–10–4 | 42–67–141 | 38.5% |  |
| XGBoost | 1–3–21 | 11–8–6 | 5–18–2 | 1–12–12 | 1–3–21 | 8–3–14 | 1–23–1 | 2–0–23 | 2–0–23 | 6–4–15 | 38–74–138 | 33.9% | 66.9% |
| STK_GRU(B,G) | 1–3–21 | 6–16–3 | 9–15–1 | 1–0–24 | 3–3–19 | 2–12–11 | 2–0–23 | 5–0–20 | 4–0–21 | 3–16–6 | 36–65–149 | 35.6% | 87.7% |
| STK_GRU(B,G,L) | 1–0–24 | 4–21–0 | 8–17–0 | 3–0–22 | 1–10–14 | 8–10–7 | 4–0–21 | 1–9–15 | 4–0–21 | 2–19–4 | 36–86–128 | 29.5% | 86.8% |
| RW (paper) | 1–1–23 | 11–9–5 | 13–3–9 | 1–11–13 | 1–4–20 | 4–3–18 | 1–17–7 | 1–3–21 | 1–21–3 | 1–3–21 | 35–75–140 | 31.8% | 30.8% |
| STK_GRU(B,L) | 1–0–24 | 5–18–2 | 6–18–1 | 3–0–22 | 1–5–19 | 7–10–8 | 2–0–23 | 1–18–6 | 4–0–21 | 1–22–2 | 31–91–128 | 25.4% |  |
| STK_LSTM(G,L) | 1–0–24 | 2–22–1 | 1–23–1 | 2–0–23 | 2–5–18 | 5–10–10 | 3–0–22 | 5–0–20 | 3–0–22 | 6–12–7 | 30–72–148 | 29.4% |  |
| STK_LSTM(B,G,L) | 1–0–24 | 1–24–0 | 3–22–0 | 1–3–21 | 4–0–21 | 3–13–9 | 2–0–23 | 5–0–20 | 1–23–1 | 6–12–7 | 27–97–126 | 21.8% |  |
| STK_GRU(G,L) | 1–0–24 | 6–16–3 | 5–19–1 | 1–0–24 | 1–4–20 | 2–13–10 | 3–0–22 | 1–19–5 | 3–0–22 | 1–20–4 | 24–91–135 | 20.9% |  |
| STK_LSTM(B,L) | 1–0–24 | 0–25–0 | 0–24–1 | 2–1–22 | 1–14–10 | 2–15–8 | 2–0–23 | 1–20–4 | 3–0–22 | 5–13–7 | 17–112–121 | 13.2% |  |
| ARIMA (paper) | 0–25–0 | 2–22–1 | 0–23–2 | 0–25–0 | 0–25–0 | 0–25–0 | 0–25–0 | 0–25–0 | 0–25–0 | 0–25–0 | 2–245–3 | 0.8% | 1.5% |

**Conclusions (Yahoo, ten tickers).**
- Same result as ALRAJHI on Kaggle: no model beats naive on average. Naive, RW with drift and ARIMA are never significantly beaten (0 losses in 250).
- The authors' baselines reproduce the paper's win rates on new data (RW 31.8% vs 30.8%, ARIMA 0.8% vs 1.5%). They are 38–44% and 4–19× worse than naive, so the paper's wins over "random walk" and "ARIMA" are wins over mis-specified baselines. This corrects the earlier guess of "a large fixed drift": the drift is the mean change over the last 5 closes, which extrapolates short-term noise.
- Averages are the best learned models (AVG(B,G) +6.7%, AVG(B,G,L) +7.3% vs naive) and the top two by win rate, as in the paper, supporting AVG(B,G) as the app engine. Ranking: averages > single models > stacking.
- Stacking is unstable: recurrent meta-learners blow up on ALRAJHI already on val (+300% to +750%), so validation would have rejected them before test.
- EMAAR EC: 55% of test closes lie below the training minimum (train 13.97–42.88 SAR, test 7.90–20.27). Price-level models extrapolate badly (L +78%, GRU-256 +111%, XGBoost +545%). The other tickers stay mostly within their training range (ADC 4.8%, TASNEE 3.1% outside).
- Data quality: Yahoo is missing about 16 trading days in 2023 for most tickers (230 val rows vs ~248 sessions; STC has 246).