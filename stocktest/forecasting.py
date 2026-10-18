"""Minimal, leakage-safe Saudi stock forecasting pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PipelineConfig:
    input_days: int = 5
    forecast_days: int = 5
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    target_column: str = "Close"


def download_tadawul_history(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data for a Yahoo Finance Tadawul ticker (for example, 1120.SR)."""
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker and date range.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    return clean_ohlcv(data)


def clean_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Validate, sort, and retain trading-session OHLCV records only."""
    frame = data.copy()
    frame.columns = [str(column).strip().title() for column in frame.columns]
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required).drop_duplicates("Date").sort_values("Date")
    frame = frame[frame["Close"] > 0].reset_index(drop=True)
    if len(frame) < 60:
        raise ValueError("At least 60 trading sessions are required.")
    return frame


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add a compact, causal technical-feature set; no future values are used."""
    frame = clean_ohlcv(data).set_index("Date")
    close = frame["Close"]
    frame["Return_1"] = close.pct_change()
    frame["SMA_5"] = close.rolling(5).mean()
    frame["EMA_5"] = close.ewm(span=5, adjust=False).mean()
    frame["Volatility_5"] = frame["Return_1"].rolling(5).std()
    frame["HL_Spread"] = (frame["High"] - frame["Low"]) / close
    frame["OC_Change"] = (frame["Close"] - frame["Open"]) / frame["Open"]
    frame["Volume_Change"] = frame["Volume"].pct_change()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index()
    if len(frame) < 45:
        raise ValueError("Too few sessions remain after feature engineering.")
    return frame


def split_and_window(
    data: pd.DataFrame, config: PipelineConfig
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], StandardScaler, StandardScaler, list[str], pd.Series]:
    """Scale with training data only and create non-overlapping target partitions."""
    frame = add_features(data)
    feature_columns = [column for column in frame.columns if column not in ("Date", config.target_column)]
    feature_columns.insert(0, config.target_column)
    total_rows = len(frame)
    train_end = int(total_rows * config.train_fraction)
    validation_end = train_end + int(total_rows * config.validation_fraction)
    if train_end <= config.input_days + config.forecast_days or validation_end >= total_rows:
        raise ValueError("Dataset is too short for the requested split and forecast windows.")

    feature_scaler = StandardScaler().fit(frame.loc[: train_end - 1, feature_columns])
    target_scaler = StandardScaler().fit(frame.loc[: train_end - 1, [config.target_column]])
    features = feature_scaler.transform(frame[feature_columns])
    targets = target_scaler.transform(frame[[config.target_column]]).ravel()

    boundaries = {"train": (0, train_end), "validation": (train_end, validation_end), "test": (validation_end, total_rows)}
    windows: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        name: ([], []) for name in boundaries
    }
    for target_start in range(config.input_days, total_rows - config.forecast_days + 1):
        target_end = target_start + config.forecast_days
        for name, (partition_start, partition_end) in boundaries.items():
            if target_start >= partition_start and target_end <= partition_end:
                windows[name][0].append(features[target_start - config.input_days : target_start])
                windows[name][1].append(targets[target_start:target_end])
                break

    prepared = {
        name: (np.asarray(inputs, dtype=np.float32), np.asarray(outputs, dtype=np.float32))
        for name, (inputs, outputs) in windows.items()
    }
    if any(len(inputs) == 0 for inputs, _ in prepared.values()):
        raise ValueError("A partition has no complete forecast windows; provide more data.")
    return prepared, feature_scaler, target_scaler, feature_columns, frame["Date"]


def build_gru(input_shape: tuple[int, int], forecast_days: int):
    """Create the intentionally small baseline model."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.GRU(32, activation="tanh"),
            tf.keras.layers.Dense(forecast_days),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train(data: pd.DataFrame, output_dir: Path, config: PipelineConfig, epochs: int = 30) -> dict[str, float]:
    """Train, evaluate, and persist one GRU baseline plus preprocessing artifacts."""
    import joblib
    import tensorflow as tf

    output_dir.mkdir(parents=True, exist_ok=True)
    sets, feature_scaler, target_scaler, feature_columns, _ = split_and_window(data, config)
    train_x, train_y = sets["train"]
    validation_x, validation_y = sets["validation"]
    test_x, test_y = sets["test"]
    model = build_gru(train_x.shape[1:], config.forecast_days)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    model.fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
        epochs=epochs,
        batch_size=min(32, len(train_x)),
        verbose=2,
        callbacks=callbacks,
        shuffle=False,
    )
    predictions = target_scaler.inverse_transform(model.predict(test_x, verbose=0))
    actual = target_scaler.inverse_transform(test_y)
    metrics = {
        "mae": float(mean_absolute_error(actual.ravel(), predictions.ravel())),
        "rmse": float(mean_squared_error(actual.ravel(), predictions.ravel()) ** 0.5),
        "test_windows": int(len(test_x)),
    }
    model.save(output_dir / "gru_model.keras")
    joblib.dump(feature_scaler, output_dir / "feature_scaler.joblib")
    joblib.dump(target_scaler, output_dir / "target_scaler.joblib")
    (output_dir / "metadata.json").write_text(
        json.dumps({"config": asdict(config), "feature_columns": feature_columns, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame({"actual": actual.ravel(), "predicted": predictions.ravel()}).to_csv(output_dir / "test_predictions.csv", index=False)
    return metrics


def predict_latest(data: pd.DataFrame, artifact_dir: Path) -> list[float]:
    """Forecast the next configured trading sessions from the newest available history."""
    import joblib
    import tensorflow as tf

    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    config = PipelineConfig(**metadata["config"])
    frame = add_features(data)
    features = frame[metadata["feature_columns"]].tail(config.input_days)
    if len(features) != config.input_days:
        raise ValueError("Not enough recent data to construct an inference window.")
    feature_scaler = joblib.load(artifact_dir / "feature_scaler.joblib")
    target_scaler = joblib.load(artifact_dir / "target_scaler.joblib")
    model = tf.keras.models.load_model(artifact_dir / "gru_model.keras")
    scaled_window = feature_scaler.transform(features).astype(np.float32)[None, :, :]
    return target_scaler.inverse_transform(model.predict(scaled_window, verbose=0))[0].tolist()


def save_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_ohlcv(data).to_csv(path, index=False)
