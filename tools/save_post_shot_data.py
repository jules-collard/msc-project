import argparse
from datetime import date
import polars as pl
from polars import col as c

from scripts.data_readers import read_game_id_mapping, batch_read_events, batch_read_puck_tracking
from scripts.post_shot.features import PostShotData

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
        description="Run shot detection algorithm, derive post-shot features, and save results to Parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "mapping_file",
        type=str,
        help="Path to the mapping file containing game ID mappings."
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
        "-o", "--output_file",
        type=str,
        default="/output/post_shot_data.parquet",
        help="Path to the output Parquet file."
    )

    args = parser.parse_args()

    games = read_game_id_mapping(args.mapping_file)

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

    print("Reading events...")
    events = batch_read_events([f"/data/sportlogiq/*/games/{id}/*_sapifullevents.json" for id in sportlogiq_ids])

    print("Reading puck tracking...")
    puck_tracking = batch_read_puck_tracking(
        [f"/data/smtoasis/*/games/{id}/*_puck_tracking_raw_measurements*.parquet" for id in SMT_ids],
        mapping=games.lazy()
    )

    post_shot_data = PostShotData(events, puck_tracking)

    print("Saving results...")
    post_shot_data.full_output().collect().write_parquet(args.output_file)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()