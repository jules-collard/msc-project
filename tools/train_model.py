import argparse
import json

from scripts.data_readers import batch_read_shot_data
from scripts.models.data import DataSplitter, prepare_data, post_shot_filter
from scripts.models.features import pre_shot_features, pre_shot_features_pruned, post_shot_features_full
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
        choices=["pre_shot", "pre_shot_pruned"],
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
        "--data_pattern",
        type=str,
        default="/output/shot_data/20242025/*.parquet",
        help="Pattern to read data files."
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

    match args.feature_set:
        case "pre_shot":
            features = pre_shot_features
        case "pre_shot_pruned":
            features = pre_shot_features_pruned
        case "post_shot_full":
            features = post_shot_features_full
            data = data.pipe(post_shot_filter)
        case _:
            raise ValueError(f"Unknown feature set: {args.feature_set}")

    if args.split_path is not None:
        print(f"Using split file: {args.split_path}")
        splitter = DataSplitter(data, features, "goal", split_path=args.split_path, seed=args.seed)
        X_train, y_train, _X_val, _y_val, _groups = splitter.get_split_data()
    else:
        X_train, y_train = DataSplitter.extract_features_and_target(data, features, "goal")

    with open(args.param_file, 'r') as f:
        params = json.load(f)

    trainer = ModelTrainer(X_train, y_train, args.framework, args.strategy, loss_correct=not args.uncalibrated, **params)
    trainer.train()
    trainer.save(args.output_file)


if __name__ == "__main__":
    main()