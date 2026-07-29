import argparse
from datetime import date
from pathlib import Path

from polars import col as c
import polars.selectors as cs

from scripts.data_readers import read_game_id_mapping, read_player_id_mapping, batch_read_events, batch_read_puck_tracking, batch_read_entity_tracking, batch_read_rosters
from scripts.post_shot.features import PostShotData
from scripts.pre_shot.features import PreShotData


def iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected date in YYYY-MM-DD format"
        ) from exc

def main():
    parser = argparse.ArgumentParser(
        prog="post_shot_data",
        description="Run complete data pipeline, synthesising event, puck and player tracking data to derive pre- and post-shot features, saving results to parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "mapping_file",
        type=str,
        help="Path to the mapping file containing game ID mappings."
    )

    parser.add_argument(
        "player_mapping_file",
        type=str,
        help="Path to the mapping file containing player ID mappings."
    )

    parser.add_argument(
        "--start_date",
        type=iso_date,
        default=None,
        help="Start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end_date",
        type=iso_date,
        default=None,
        help="End date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="/output/shot_data",
        help="Output directory to store the resulting parquet files.",
    )

    args = parser.parse_args()

    games = read_game_id_mapping(args.mapping_file)
    player_mapping = read_player_id_mapping(args.player_mapping_file)

    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start_date must be on or before --end_date.")

    if args.start_date:
        games = games.filter(
            c('GameDate') >= args.start_date
        )

    if args.end_date:
        games = games.filter(
            c('GameDate') <= args.end_date
        )

    sportlogiq_ids = games.select(c('SportlogiqGameID')).to_series().to_list()
    SMT_ids = games.select(c('SMTGameID')).to_series().to_list()

    print("Reading rosters...")
    player_info = batch_read_rosters([f"/data/sportlogiq/*/games/{id}/*_gameroster.json" for id in sportlogiq_ids])

    print("Reading events...")
    events = batch_read_events([f"/data/sportlogiq/*/games/{id}/*_sapifullevents.json" for id in sportlogiq_ids])

    print("Reading player tracking...")
    player_tracking = batch_read_entity_tracking(
        [f"/data/smtoasis/*/games/{id}/*_entity_tracking_processed_measurements.parquet" for id in SMT_ids],
        mapping=games.lazy()
    )

    print("Reading puck tracking...")
    puck_tracking = batch_read_puck_tracking(
        [f"/data/smtoasis/*/games/{id}/*_puck_tracking_raw_measurements.parquet" for id in SMT_ids],
        mapping=games.lazy()
    )

    print("Processing post-shot data...")
    post_shot_data = PostShotData(events, puck_tracking, player_info)
    post_shots = post_shot_data.full_output().collect()

    print("Processing pre-shot data...")
    pre_shot_data = PreShotData(post_shots.lazy(), player_tracking, player_mapping)
    pre_shots = pre_shot_data.full_output().collect()

    combined = post_shots.join(
        pre_shots,
        on=['game_id', 'period', 'shot_id'],
        how='left',
        suffix='_dup'
    ).drop(cs.ends_with('_dup'))

    print("Saving results...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for game_id in combined.select(c('game_id')).unique().to_series().to_list():
        (
            combined.filter(c('game_id') == game_id)
            .write_parquet(output_dir / f"{game_id}_shot_data.parquet")
        )

if __name__ == "__main__":
    main()