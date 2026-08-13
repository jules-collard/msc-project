import argparse

import polars as pl
import optuna

from scripts.data_readers import batch_read_shot_data
from scripts.models.experiments import ExperimentRunner
from scripts.models.data import prepare_data, post_shot_filter
from scripts.models.features import pre_shot_features, pre_shot_features_pruned, post_shot_features_full


def main():

    parser = argparse.ArgumentParser(
        prog="model_experiments",
        description="Run model experiments with hyperparameter tuning using Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="Output file to save experiment results."
    )

    parser.add_argument(
        "feature_set",
        type=str,
        choices=["pre_shot", "pre_shot_pruned", "post_shot_full"],
        help="Feature set to use for training."
    )

    parser.add_argument(
        "--data_pattern",
        type=str,
        default="/output/shot_data/20242025/*.parquet",
        help="Pattern to read data files."
    )

    parser.add_argument(
        "--split_path",
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
        "--frameworks",
        nargs='+',
        default=['lightgbm', 'xgboost'],
        choices=['lightgbm-dart', 'xgboost-dart', 'lightgbm', 'xgboost'],
        help="List of frameworks to test."
    )

    parser.add_argument(
        "--strategies",
        nargs='+',
        default=['RO', 'RU', 'WCE'],
        choices=['RO', 'RU', 'WCE'],
        help="List of imbalance strategies to test."
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of trials for hyperparameter tuning."
    )

    parser.add_argument(
        "--multivariate",
        action='store_true',
        help="Use multivariate sampling for hyperparameter tuning."
    )

    parser.add_argument(
        "--info_log",
        action='store_true',
        help="Enable info logging."
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=980,
        help="Number of trees to build"
    )

    args = parser.parse_args()

    data = batch_read_shot_data(args.data_pattern).pipe(prepare_data)

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
    
    experiment = ExperimentRunner(data.collect(), features, "goal", split_path=args.split_path, n_estimators=args.n_estimators, seed=args.seed)

    if args.info_log:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = experiment.run_all(frameworks=args.frameworks, strategies=args.strategies, n_trials=args.n_trials, multivariate=args.multivariate)
    pl.from_dicts(results).write_parquet(args.output_file)

if __name__ == "__main__":
    main()