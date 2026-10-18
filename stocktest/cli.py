"""Command-line entry point for the minimal forecasting pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from forecasting import PipelineConfig, download_tadawul_history, predict_latest, save_csv, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Saudi stock forecasting pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download daily Tadawul OHLCV history")
    download.add_argument("--ticker", default="1120.SR", help="Yahoo Finance ticker, e.g. 1120.SR")
    download.add_argument("--start", default="2021-01-01")
    download.add_argument("--end")
    download.add_argument("--output", default="data/1120.SR.csv")
    training = commands.add_parser("train", help="Train and save a GRU baseline")
    training.add_argument("--input", required=True)
    training.add_argument("--output-dir", default="artifacts")
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--input-days", type=int, default=5)
    training.add_argument("--forecast-days", type=int, default=5)
    prediction = commands.add_parser("predict", help="Forecast from the newest rows in a CSV")
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--artifact-dir", default="artifacts")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "download":
        save_csv(download_tadawul_history(arguments.ticker, arguments.start, arguments.end), Path(arguments.output))
        print(f"Saved {arguments.output}")
    elif arguments.command == "train":
        config = PipelineConfig(input_days=arguments.input_days, forecast_days=arguments.forecast_days)
        metrics = train(pd.read_csv(arguments.input), Path(arguments.output_dir), config, arguments.epochs)
        print(json.dumps(metrics, indent=2))
    else:
        forecast = predict_latest(pd.read_csv(arguments.input), Path(arguments.artifact_dir))
        print(json.dumps({"next_close_forecast": forecast}, indent=2))


if __name__ == "__main__":
    main()
