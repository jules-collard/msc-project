#!/bin/bash
set -e

# Define the start and end dates for each month
dates=(
  "2025-10-01 2025-10-31"
  "2025-11-01 2025-11-30"
  "2025-12-01 2025-12-31"
  "2026-01-01 2026-01-31"
  "2026-02-01 2026-02-28"
  "2026-03-01 2026-03-31"
  "2026-04-01 2026-04-30"
  "2026-05-01 2026-05-31"
  "2026-06-01 2026-06-30"
)

# Loop through and run the command, stopping on any failure (set -e equivalent)
for d in "${dates[@]}"; do
  set -- $d
  PYTHONPATH=scripts/ uv run python -m tools.data_pipeline \
    mappings/NHL_20242025_20252026_game_smt_sportlogiq_id_map.csv \
    mappings/NHL_20242025_20252026_player_sportlogiq_id_map.csv \
    --start_date "$1" --end_date "$2" \
    --output_dir /output/shot_data/20252026/
done