import argparse
import joblib
from pathlib import Path
from datetime import date, timedelta
from warnings import warn

import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
import shap
import polars as pl
from polars import col as c

from scripts.models.features import feature_sets, post_shot_feature_groups
from scripts.models.data import prepare_data, post_shot_filter, one_hot_encode_shot_type, polars_to_pandas
from scripts.data_readers import batch_read_shot_data, read_game_id_mapping

def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected date in YYYY-MM-DD format"
        ) from exc

def _get_feature_clusters(feature_set, feature_groups):
    feature_to_idx = {name: idx for idx, name in enumerate(feature_set)}

    # Initialize the NxN distance matrix with 1s (max distance) and 0 on the diagonal
    n_features = len(feature_set)
    dist_matrix = np.ones((n_features, n_features))
    np.fill_diagonal(dist_matrix, 0)

    # Force features in the same group to have a distance of 0
    for group in feature_groups.values():
        # Convert feature names to their corresponding column indices
        indices = [feature_to_idx[feat] for feat in group]

        # Set pairwise distances for all features in this group to 0
        for i in indices:
            for j in indices:
                dist_matrix[i, j] = 0

    condensed_dist = squareform(dist_matrix)
    custom_linkage = linkage(condensed_dist, method="complete")
    return custom_linkage

def _clean_shap_values(shap_values, id_df, feature_set, feature_groups) -> pl.DataFrame:
    shap_values_df = (
        pl.DataFrame(shap_values.values, schema=feature_set)
        .with_columns(base_value=shap_values.base_values)
        .with_columns([
            pl.sum_horizontal(group).alias(name) for name, group in feature_groups.items()
        ]).select('base_value', *feature_groups.keys())
    )

    return pl.concat([id_df, shap_values_df], how="horizontal", strict=True)

def _validate_args(args):
    if args.feature_set != "post_shot_one_hot":
        raise NotImplementedError("SHAP calculation is currently only implemented for the 'post_shot_one_hot' feature set.")

    if not args.game_mapping and any([args.start_date, args.end_date, args.team, args.stage]):
        raise ValueError("If --game-mapping is not provided, --start-date, --end-date, --team, and --stage cannot be specified.")
    if args.game_mapping and not any([args.start_date, args.end_date, args.team, args.stage]):
        warn("If --game-mapping is provided, but none of --start-date, --end-date, --team, or --stage are specified, all games in the mapping will be processed.")

    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise ValueError("--start-date must be on or before --end-date.")
    

def _add_pipeline(data: pl.DataFrame | pl.LazyFrame, feature_set) -> pl.DataFrame | pl.LazyFrame:
    if feature_set.startswith("post_shot"):
        data = data.pipe(post_shot_filter)

    if "one_hot" in feature_set:
        data = data.pipe(one_hot_encode_shot_type)

    return data

def _save_data(data: pl.DataFrame, output_dir: str):
    print("Saving results...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for game_id in data.select(c('game_id')).unique().to_series().to_list():
        (
            data.filter(c('game_id') == game_id)
            .write_parquet(output_dir / f"{game_id}_shap.parquet")
        )

def _chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def _get_game_ids(args):
    if args.game_mapping is None:
        return None

    games = read_game_id_mapping(args.game_mapping)

    if args.start_date:
        games = games.filter(
            c('GameDate') >= args.start_date
        )

    if args.end_date:
        games = games.filter(
            c('GameDate') <= args.end_date
        )

    if args.team:
        games = games.filter(
            c('HomeTeam').eq(args.team).or_(c('AwayTeam').eq(args.team))
        )

    if args.stage:
        games = games.filter(
            c('Stage').eq(args.stage)
        )

    return games.select(c('SportlogiqGameID').cast(pl.String)).to_series().to_list()

def main():

    parser = argparse.ArgumentParser(
        prog="calculate_shap",
        description="Calculate SHAP values for a given model and dataset, saving results to a specified output directory.",
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
        choices=feature_sets.keys(),
        help="Feature set to use for SHAP value calculation."
    )

    parser.add_argument(
        "training_data_pattern",
        type=str,
        help="Pattern to read training data files (used for SHAP background dataset)."
    )

    parser.add_argument(
        "predict_data_pattern",
        type=str,
        help="Pattern to read data files for which SHAP values will be calculated."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to save SHAP values."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to use for SHAP value calculation."
    )

    parser.add_argument(
        "--game-mapping",
        type=str,
        help="Path to the mapping file containing game ID mappings."
    )

    parser.add_argument(
        "--start-date",
        type=_iso_date,
        default=None,
        help="Start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end-date",
        type=_iso_date,
        default=None,
        help="End date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--team",
        type=str,
        default=None,
        help="Team name to filter the data for SHAP value calculation."
    )

    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["regular", "playoffs"],
        help="Stage of the season to filter the data for SHAP value calculation."
    )

    args = parser.parse_args()

    _validate_args(args)

    model = joblib.load(args.model_file)
    def prob_wrapper(X):
        return model.predict_proba(X)[:, 1]

    data_train = batch_read_shot_data(args.training_data_pattern).pipe(prepare_data)
    data_predict = batch_read_shot_data(args.predict_data_pattern).pipe(prepare_data)
    data_train = _add_pipeline(data_train, args.feature_set)
    data_predict = _add_pipeline(data_predict, args.feature_set)

    features = feature_sets[args.feature_set]

    X_train = data_train.select(features).collect().sample(args.num_samples, seed=args.seed)

    game_ids = _get_game_ids(args)
    data_predict = (
        data_predict
        .select("game_id", "period", "shot_id", *features)
        .filter(c('game_id').is_in(game_ids))
        .collect()
    )

    linkage = _get_feature_clusters(features, post_shot_feature_groups)
    masker = shap.maskers.Partition(polars_to_pandas(X_train), max_samples=args.num_samples, clustering=linkage)
    explainer = shap.Explainer(prob_wrapper, masker=masker, seed=args.seed)

    for chunk in _chunker(game_ids, 15):
        print(f"Calculating SHAP values for {chunk}...")

        chunk_data = (
            data_predict
            .filter(c("game_id").is_in(chunk))
            .select("game_id", "period", "shot_id", *features)
        )

        id_df = chunk_data.select("game_id", "period", "shot_id")
        feature_data = chunk_data.select(features)

        shap_values = explainer(polars_to_pandas(feature_data))

        shap_values_df = _clean_shap_values(
            shap_values,
            id_df,
            features,
            post_shot_feature_groups,
        )

        _save_data(shap_values_df, args.output_dir)

        del chunk_data, feature_data, id_df, shap_values, shap_values_df


if __name__ == "__main__":
    main()
