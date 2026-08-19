#!/bin/bash

# Define the start and end dates for each month
dates=(
  "2024-10-01 2024-10-31"
  "2024-11-01 2024-11-30"
  "2024-12-01 2024-12-31"
  "2025-01-01 2025-01-31"
  "2025-02-01 2025-02-28"
  "2025-03-01 2025-03-31"
  "2025-04-01 2025-04-30"
  "2025-05-01 2025-05-31"
  "2025-06-01 2025-06-30"
)

# Loop through and run the command, stopping on any failure (set -e equivalent)
for d in "${dates[@]}"; do
  set -- $d
  PYTHONPATH=scripts/ uv run python -m tools.data_pipeline \
    mappings/NHL_20242025_20252026_game_smt_sportlogiq_id_map.csv \
    mappings/NHL_20242025_20252026_player_sportlogiq_id_map.csv \
    --start_date "$1" --end_date "$2" \
    --output_dir /output/shot_data/20242025-clean/ || break
done