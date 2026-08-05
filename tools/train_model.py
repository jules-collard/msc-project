import argparse
import json

from scripts.data_readers import batch_read_shot_data
from scripts.models.data import DataSplitter, prepare_data
from scripts.models.features import pre_shot_features
from scripts.models.training import ModelTrainer


def main():

    parser = argparse.ArgumentParser(
        prog="model_experiments",
        description="Run model experiments with hyperparameter tuning using Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "framework",
        type=str,
        help="Framework to use for training."
    )

    parser.add_argument(
        "strategy",
        type=str,
        help="Strategy to use for training."
    )

    parser.add_argument(
        "param_file",
        type=str,
        help="Path to the parameter file."
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="Output file to save model object."
    )

    parser.add_argument(
        "--data_pattern",
        type=str,
        default="/output/shot_data/20242025/*.parquet",
        help="Pattern to read data files."
    )

    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Path to the train/test split file."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--calibrate",
        action='store_true',
        help="Enable calibration of the model."
    )

    args = parser.parse_args()

    data = batch_read_shot_data(args.data_pattern).pipe(prepare_data)
    splitter = DataSplitter(data.collect(), pre_shot_features, "goal", split_path=args.split_path, seed=args.seed)
    X_train, y_train, X_val, y_val, _ = splitter.get_split_data()

    with open(args.param_file, 'r') as f:
        params = json.load(f)

    trainer = ModelTrainer(X_train, y_train, args.framework, args.strategy, X_val, y_val, calibrate=args.calibrate, **params)
    trainer.train()
    trainer.save(args.output_file)


if __name__ == "__main__":
    main()