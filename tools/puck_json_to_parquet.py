import argparse
import glob
from pathlib import Path

import ijson
import polars as pl
from polars import col as c

def parse_tracking_data(file_path):
    """
    Generator that streams nested tracking data from a JSON file.
    """
    with open(file_path, "rb") as f:
        # Only yield puck tracking data (EntityId == '1')
        for o in ijson.items(f, 'item.TrackingData.item', use_float=True):
            if o.get('EntityId') == '1':
                yield o

def process_file(input_path, output_dir=None, delete_input=False, clean=False):
    """
    Reads a single JSON file, converts to DataFrame, and saves as Parquet.
    """
    path_obj = Path(input_path)

    df = pl.from_dicts(parse_tracking_data(input_path), infer_schema_length=None).lazy()

    if clean:
        df = (
            df
            .with_columns(
                c('Location').struct.rename_fields(['x', 'y', 'z']).struct.unnest(), 
                c('Velocity').struct.rename_fields(['vx', 'vy', 'vz']).struct.unnest(),
                c('Acceleration').struct.rename_fields(['ax', 'ay', 'az']).struct.unnest(),
            ).with_columns(
                c('z', 'vz', 'az').fill_null(0) # missing z values represent 0 (i.e. puck on ice)
            ).drop(c('Location', 'Velocity', 'Acceleration', 'OnPlayingSurface', 'PayloadData', 'EntityOfficialId',
                    'Landmarks3D', 'MetaTag1', 'LocationLTC', 'MeasurementId', 'LocationConfidence'))
            .rename({'LocationUTC':'ts', 'ClockState':'clock_state', 'EntityId':'entity_id'})
        )
    
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_path = path_obj.parent

    output_path = output_dir_path / f"{path_obj.stem}.parquet"

    df.sink_parquet(output_path)

    print(f"Successfully converted: {path_obj.name} -> {output_path.name}")

    if delete_input:
        path_obj.unlink()
        print(f"Deleted original file: {path_obj.name}")

def main():
    parser = argparse.ArgumentParser(
        prog="puck_json_to_parquet",
        description="Extract puck tracking from SMT JSON tracking data and save to Parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Define CLI arguments
    parser.add_argument(
        "input_files", 
        type=str, 
        help="Filename(s) or Glob pattern for input JSON files (e.g., 'game_id/HOCKEY_NHL_*.json')"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="Directory to save the resulting Parquet files. Defaults to the input file's directory."
    )
    parser.add_argument(
        "-d", "--delete",
        action="store_true",
        help="Delete the input JSON file after successful conversion."
    )
    parser.add_argument(
        "-c", "--clean",
        action="store_true",
        help="Clean the output DataFrame to only include essential puck tracking columns."
    )
    
    args = parser.parse_args()
    matched_files = glob.glob(args.input_files, recursive=True)
    
    if not matched_files:
        print(f"No files found matching pattern: {args.input_files}")
        return
        
    print(f"Found {len(matched_files)} files. Starting conversion...")
    
    for file_path in matched_files:
        process_file(file_path, args.output_dir, args.delete, args.clean)
        
    print("Conversion complete.")

if __name__ == "__main__":
    main()