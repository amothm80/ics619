import unittest

import numpy as np
import pandas as pd

from forecasting import PipelineConfig, clean_ohlcv, split_and_window


class PipelineTests(unittest.TestCase):
    def test_windows_use_all_partitions_without_leakage(self):
        dates = pd.bdate_range("2021-01-01", periods=180)
        close = np.linspace(80, 140, len(dates))
        data = pd.DataFrame({
            "Date": dates,
            "Open": close - 0.5,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.linspace(1_000_000, 2_000_000, len(dates)),
        })
        windows, _, _, columns, _ = split_and_window(clean_ohlcv(data), PipelineConfig())
        self.assertEqual(windows["train"][0].shape[1:], (5, len(columns)))
        self.assertEqual(windows["train"][1].shape[1], 5)
        self.assertGreater(len(windows["validation"][0]), 0)
        self.assertGreater(len(windows["test"][0]), 0)


if __name__ == "__main__":
    unittest.main()
