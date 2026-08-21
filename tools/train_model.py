import argparse
import json

import polars as pl

from scripts.data_readers import batch_read_shot_data
from scripts.models.data import DataSplitter, prepare_data, post_shot_filter
from scripts.models.features import get_features
from scripts.models.training import ModelTrainer


def main():

    parser = argparse.ArgumentParser(
        prog="train_model",
        description="Train a model using the specified framework and strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "framework",
        type=str,
        choices=['xgboost', 'lightgbm'],
        help="Framework to use for training."
    )

    parser.add_argument(
        "strategy",
        type=str,
        choices=['WCE', 'RO', 'RU'],
        help="Strategy to use for training."
    )

    parser.add_argument(
        "feature_set",
        type=str,
        choices=["pre_shot", "pre_shot_pruned", "pre_shot_minimal", "pre_shot_speed", "post_shot_full", "post_shot_minimal", "post_shot_xg"],
        help="Feature set to use for training."
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
        "--data-pattern",
        type=str,
        default="/output/shot_data/20242025/*.parquet",
        help="Pattern to read data files."
    )

    parser.add_argument(
        "--xg-data-pattern",
        type=str,
        default=None,
        help="Pattern/file to read xG predictions - only used when feature_set is 'post_shot_xg'."
    )

    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Path to the train/test split file. If not included, model is trained on entire dataset."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--uncalibrated",
        action='store_true',
        help="Disable loss correction calibration for the model."
    )

    args = parser.parse_args()


    data = batch_read_shot_data(args.data_pattern).pipe(prepare_data).collect()
    features = get_features(args.feature_set)

    if "pre_shot" in features and args.xg_data_pattern is None:
        parser.error("--xg-data-pattern is required")
    elif args.xg_data_pattern is not None and "pre_shot" not in features:
        parser.error("--xg-data-pattern should only be used when feature_set is post_shot_xg")

    if args.feature_set.startswith("post_shot"):
        data = data.pipe(post_shot_filter)

    if "pre_shot" in features:
        xg_data = pl.scan_parquet(args.xg_data_pattern).collect()
        data = (
            data
            .pipe(post_shot_filter)
            .join(xg_data, on=["game_id", "period", "shot_id"], how="left", validate="1:1")
        )

    if args.split_path is not None:
        print(f"Using split file: {args.split_path}")
        splitter = DataSplitter(data, features, "goal", split_path=args.split_path, seed=args.seed)
        X_train, y_train, X_val, y_val, _groups = splitter.get_split_data()
    else:
        X_train, y_train = DataSplitter.extract_features_and_target(data, features, "goal")
        X_val, y_val = None, None

    with open(args.param_file, 'r') as f:
        params = json.load(f)

    trainer = ModelTrainer(X_train, y_train, args.framework, args.strategy, loss_correct=not args.uncalibrated, X_val=X_val, y_val=y_val, **params)
    trainer.train()
    trainer.save(args.output_file)


if __name__ == "__main__":
    main()