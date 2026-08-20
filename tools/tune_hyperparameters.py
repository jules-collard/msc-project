import argparse
import json

import polars as pl
import optuna

from scripts.data_readers import batch_read_shot_data
from scripts.models.experiments import ExperimentRunner
from scripts.models.data import prepare_data, post_shot_filter
from scripts.models.features import pre_shot_features, pre_shot_features_pruned, pre_shot_features_minimal, pre_shot_features_speed, post_shot_features_full, post_shot_features_minimal, post_shot_features_xg


def main():

    parser = argparse.ArgumentParser(
        prog="model_experiments",
        description="Run model experiments with hyperparameter tuning using Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


    parser.add_argument(
        "feature_set",
        type=str,
        choices=["pre_shot", "pre_shot_pruned", "pre_shot_minimal", "pre_shot_speed", "post_shot_full", "post_shot_minimal", "post_shot_xg"],
        help="Feature set to use for training."
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output (parquet) file to save experiment results."
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
        help="Pattern/file to read xG predictions - only used when feature_set is 'post_shot_minimal'."
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
        "--n-trials",
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
        "--info-log",
        action='store_true',
        help="Enable info logging."
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=980,
        help="Number of trees to build"
    )

    parser.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="Path to a JSON file to save best parameters"
    )

    args = parser.parse_args()

    if args.feature_set == "post_shot_xg" and args.xg_data_pattern is None:
        parser.error("--post-shot-minimal-config is required when feature_set is post_shot_minimal")
    elif args.xg_data_pattern is not None and args.feature_set != "post_shot_xg":
        parser.error("--post-shot-minimal-config should only be used when feature_set is post_shot_minimal")

    data = batch_read_shot_data(args.data_pattern).pipe(prepare_data)

    match args.feature_set:
        case "pre_shot":
            features = pre_shot_features
        case "pre_shot_pruned":
            features = pre_shot_features_pruned
        case "pre_shot_minimal":
            features = pre_shot_features_minimal
        case "pre_shot_speed":
            features = pre_shot_features_speed
        case "post_shot_full":
            features = post_shot_features_full
            data = data.pipe(post_shot_filter)
        case "post_shot_minimal":
            features = post_shot_features_minimal
            data = data.pipe(post_shot_filter)
        case "post_shot_xg":
            features = post_shot_features_xg
            xg_data = pl.scan_parquet(args.xg_data_pattern)
            data = (
                data
                .pipe(post_shot_filter)
                .join(xg_data, on=["game_id", "period", "shot_id"], how="left", validate="1:1")
            )
        case _:
            raise NotImplementedError(f"Unknown feature set: {args.feature_set}")
    
    experiment = ExperimentRunner(data.collect(), features, "goal", split_path=args.split_path, n_estimators=args.n_estimators, seed=args.seed)

    if args.info_log:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = experiment.run_all(frameworks=args.frameworks, strategies=args.strategies, n_trials=args.n_trials, multivariate=args.multivariate)
    
    if args.output_file is not None:
        pl.from_dicts(results).write_parquet(args.output_file)
        print(f"Experiment results saved to {args.output_file}")

    if args.params_file and experiment.best_params:
        with open(args.params_file, 'w') as f:
            json.dump(experiment.best_params, f, indent=4)
        print(f"Best parameters saved to {args.params_file}")

if __name__ == "__main__":
    main()