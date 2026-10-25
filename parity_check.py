#!/usr/bin/env python3
"""Parity of prepare_data.py + train_report.py against tasi_edl_pipeline.py (REFACTOR_PLAN.md).

Exact checks, no training: features.parquet, windows.npz, scalers, raw.parquet; Random Walk,
ARIMA and XGBoost predictions recomputed with the reference functions. With --ref-run it also
compares every model's metrics with a reference run (same seed, --deterministic).

    python tasi_edl_pipeline.py --companies ADC --deterministic --out-dir runs/ref_adc_det
    python train_report.py --run-dir runs/adc_det --companies ADC --deterministic
    python parity_check.py --run-dir runs/adc_det --ref-run runs/ref_adc_det
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tasi_edl_pipeline as ref  # noqa: E402
from prepare_data import company_dir, load_prepared  # noqa: E402
from train_report import baseline_xgboost  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--company", default="ADC")
    ap.add_argument("--saudi-csv", default="data/saudi/Tadawul_stcks.csv")
    ap.add_argument("--run-dir", default=None, help="train_report.py run to compare (optional)")
    ap.add_argument("--ref-run", default=None, help="tasi_edl_pipeline.py --out-dir of the same config")
    ap.add_argument("--quick", action="store_true", help="the runs used --quick (XGBoost size)")
    a = ap.parse_args()
    n = a.company
    st = ref.SAUDI_COMPANIES[n][0]
    d = company_dir(a.data, "saudi", n)

    print(f"== prepared data ({d})")
    raw = ref.load_saudi_raw(a.saudi_csv)
    co_ref = ref.engineer_saudi(raw, n, st, ref.SELECTED_TI_SA)
    sp_ref = ref.prepare_split(co_ref, ref.SELECTED_TI_SA, "close")
    sp, co, raw_c, meta = load_prepared(d)
    try:
        pd.testing.assert_frame_equal(co_ref, co, check_exact=True, check_freq=False, check_names=False)
        check("features.parquet == engineer_saudi", True, f"shape {co.shape}")
    except AssertionError as e:
        check("features.parquet == engineer_saudi", False, str(e).splitlines()[0])
    try:
        pd.testing.assert_frame_equal(raw[raw["trading_name"] == n], raw_c, check_exact=True, check_freq=False)
        check("raw.parquet == company raw rows", True, f"shape {raw_c.shape}")
    except AssertionError as e:
        check("raw.parquet == company raw rows", False, str(e).splitlines()[0])
    check("meta feature list == Table 4", meta["features"] == ref.SELECTED_TI_SA)
    for k in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y", "test_y_eval", "val_y_eval"):
        r, m = getattr(sp_ref, k), getattr(sp, k)
        check(f"windows.npz {k}", r.shape == m.shape and np.array_equal(r, m), f"{m.shape}")
    check("target scaler", np.array_equal(sp.target_scaler.mean_, sp_ref.target_scaler.mean_)
          and np.array_equal(sp.target_scaler.scale_, sp_ref.target_scaler.scale_))
    for k in ("train_df", "val_df", "test_df"):
        check(f"split frame {k}", getattr(sp, k).equals(getattr(sp_ref, k)))

    print("== baselines from prepared data vs reference functions (exact)")
    pairs = {
        "Random Walk": (ref.baseline_random_walk(sp_ref, "close")[0], ref.baseline_random_walk(sp, "close")[0]),
        "ARIMA": (ref.baseline_arima(sp_ref, "close")[0], ref.baseline_arima(sp, "close")[0]),
        "XGBoost": (ref.baseline_xgboost(co_ref, ref.SELECTED_TI_SA, "close", a.quick)[0],
                    baseline_xgboost(sp, meta["features"], "close", a.quick)[0]),
    }
    for model, (pr, pn) in pairs.items():
        check(f"{model} predictions identical", np.array_equal(pr, pn),
              f"RMSE {ref.metrics(sp_ref.test_y_eval, pr)['RMSE']:.6f}")

    if a.run_dir and a.ref_run:
        print(f"== metrics: {a.run_dir} vs reference {a.ref_run}")
        r = pd.read_csv(os.path.join(a.ref_run, "results_saudi_all_models.csv"))
        m = pd.read_csv(os.path.join(a.run_dir, "results_saudi_all_models.csv"))
        r, m = r[r["Company"] == n].set_index("Model"), m[m["Company"] == n].set_index("Model")
        check("same models in the same order", list(r.index) == list(m.index))
        cmp = pd.DataFrame({"ref_RMSE": r["RMSE"], "new_RMSE": m["RMSE"], "ref_MAE": r["MAE"], "new_MAE": m["MAE"]})
        cmp["RMSE_diff_%"] = 100 * (cmp["new_RMSE"] - cmp["ref_RMSE"]) / cmp["ref_RMSE"]
        print(cmp.round(4).to_string())
        for model in ("Random Walk", "ARIMA", "XGBoost"):
            same = all(r.loc[model, c] == m.loc[model, c] for c in ("MAE", "MSE", "RMSE", "MRE", "R2"))
            check(f"{model} metrics identical", same)

    print(f"\n{'ALL EXACT CHECKS PASSED' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
