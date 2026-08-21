import argparse
import joblib
import polars as pl
from polars import col as c

from scripts.data_readers import batch_read_shot_data
from scripts.models.data import DataSplitter, prepare_data, post_shot_filter, polars_to_pandas
from scripts.models.features import get_features
def main():

    parser = argparse.ArgumentParser(
        prog="predict",
        description="Generate prediction using the specified model object and data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model_file",
        type=str,
        help="Path to the model object file."
    )

    parser.add_argument(
        "feature_set",
        type=str,
        choices=["pre_shot", "pre_shot_pruned", "pre_shot_minimal", "pre_shot_speed", "post_shot_full", "post_shot_minimal"],
        help="Feature set to use for prediction."
    )

    parser.add_argument(
        "data_pattern",
        type=str,
        help="Pattern to read data files."
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="Output parquet file to save predictions."
    )

    parser.add_argument(
        "--col-name",
        type=str,
        default="expected_goals",
        help="Column name for the prediction output."
    )

    args = parser.parse_args()

    data = batch_read_shot_data(args.data_pattern).pipe(prepare_data).collect()
    features = get_features(args.feature_set)

    # if "pre_shot" in features and args.xg_data_pattern is None:
    #     parser.error("--xg-data-pattern is required")
    # elif args.xg_data_pattern is not None and "pre_xg" not in features:
    #     parser.error("--xg-data-pattern should only be used when feature_set is post_shot_xg")

    if args.feature_set.startswith("post_shot"):
        data = data.pipe(post_shot_filter)

    if "pre_shot" in features:
        xg_data = pl.scan_parquet(args.xg_data_pattern).collect()
        data = (
            data
            .pipe(post_shot_filter)
            .join(xg_data, on=["game_id", "period", "shot_id"], how="left", validate="1:1")
        )

    X, _ = DataSplitter.extract_features_and_target(data, features, "goal")
    ids = data.select(c('game_id', 'period', 'shot_id'))

    clf = joblib.load(args.model_file)
    preds = clf.predict_proba(polars_to_pandas(X))[:,1]

    results = ids.with_columns(predictions = preds).rename({"predictions": args.col_name})
    results.write_parquet(args.output_file)
    print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()