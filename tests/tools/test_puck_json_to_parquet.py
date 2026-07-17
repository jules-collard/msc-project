import json
import sys
from pathlib import Path
from unittest.mock import patch

import polars as pl

# Import the functions from your script
from tools.puck_json_to_parquet import parse_tracking_data, process_file, main

# Define a mock JSON structure that matches the 'item.TrackingData.item' ijson path
MOCK_JSON_DATA = [
    {
        "TrackingData": [
            {"EntityId": "1", "x_coord": 10.5, "y_coord": -5.2, "speed": 15.0},  # Puck
            {"EntityId": "2", "x_coord": 0.0, "y_coord": 0.0, "speed": 10.0}     # Player
        ]
    },
    {
        "TrackingData": [
            {"EntityId": "1", "x_coord": 11.0, "y_coord": -4.0, "speed": 16.5},  # Puck
            {"EntityId": "5", "x_coord": -5.0, "y_coord": 2.0, "speed": 12.0}    # Player
        ]
    }
]

def test_parse_tracking_data(tmp_path):
    """Test that the generator only yields records where EntityId == '1'."""
    # Setup mock file
    json_file = tmp_path / "test_data.json"
    json_file.write_text(json.dumps(MOCK_JSON_DATA))

    # Execute generator
    results = list(parse_tracking_data(str(json_file)))

    # Assertions
    assert len(results) == 2, "Should only extract the 2 puck records"
    assert all(r.get("EntityId") == "1" for r in results)
    assert results[0]["x_coord"] == 10.5
    assert results[1]["speed"] == 16.5

def test_process_file_default_output(tmp_path):
    """Test saving the Parquet file to the same directory as the input."""
    json_file = tmp_path / "game_2023.json"
    json_file.write_text(json.dumps(MOCK_JSON_DATA))

    # Run processing with no output directory
    process_file(str(json_file))

    expected_parquet = tmp_path / "game_2023.parquet"
    assert expected_parquet.exists(), "Parquet file should be created in the input directory"

    # Verify the contents of the generated Parquet file
    df = pl.read_parquet(expected_parquet)
    assert df.shape == (2, 4), "DataFrame should have 2 rows and 4 columns"
    assert df["EntityId"].to_list() == ["1", "1"]
    assert df["x_coord"].to_list() == [10.5, 11.0]

def test_process_file_custom_output(tmp_path):
    """Test saving the Parquet file to a explicitly provided output directory."""
    input_dir = tmp_path / "raw_data"
    input_dir.mkdir()
    json_file = input_dir / "game_2024.json"
    json_file.write_text(json.dumps(MOCK_JSON_DATA))

    output_dir = tmp_path / "processed_data"
    # Do not create output_dir here; let the script create it (tests mkdir(parents=True))

    process_file(str(json_file), output_dir=str(output_dir))

    expected_parquet = output_dir / "game_2024.parquet"
    assert expected_parquet.exists(), "Parquet file should be created in the target directory"
    assert not (input_dir / "game_2024.parquet").exists(), "Should not create file in input directory"

def test_main_cli_execution(tmp_path):
    """Test the command line argument parsing and full script execution."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # Create two input files to test the globbing
    (input_dir / "HOCKEY_NHL_1.json").write_text(json.dumps(MOCK_JSON_DATA))
    (input_dir / "HOCKEY_NHL_2.json").write_text(json.dumps(MOCK_JSON_DATA))

    output_dir = tmp_path / "output"

    # Mock the command line arguments passed to sys.argv
    test_args = [
        "converter.py", 
        str(input_dir / "*.json"), 
        "--output_dir", 
        str(output_dir)
    ]
    
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify both files were converted and saved to the output directory
    assert (output_dir / "HOCKEY_NHL_1.parquet").exists()
    assert (output_dir / "HOCKEY_NHL_2.parquet").exists()

def test_process_file_deletes_input(tmp_path):
    """Test that the input file is deleted when delete_input=True."""
    json_file = tmp_path / "game_delete_test.json"
    json_file.write_text(json.dumps(MOCK_JSON_DATA))
    
    # Verify file exists before processing
    assert json_file.exists()

    # Run processing with delete flag
    process_file(str(json_file), delete_input=True)

    expected_parquet = tmp_path / "game_delete_test.parquet"
    
    # Assertions
    assert expected_parquet.exists(), "Parquet file should be created"
    assert not json_file.exists(), "Original JSON file should be deleted"

def test_process_file_keeps_input_by_default(tmp_path):
    """Test that the input file is kept when delete_input=False (default)."""
    json_file = tmp_path / "game_keep_test.json"
    json_file.write_text(json.dumps(MOCK_JSON_DATA))

    # Run processing without delete flag
    process_file(str(json_file))

    expected_parquet = tmp_path / "game_keep_test.parquet"
    
    # Assertions
    assert expected_parquet.exists(), "Parquet file should be created"
    assert json_file.exists(), "Original JSON file should be kept"

def test_main_cli_delete_flag(tmp_path):
    """Test that the --delete flag is correctly parsed and executed."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    file1 = input_dir / "HOCKEY_NHL_3.json"
    file2 = input_dir / "HOCKEY_NHL_4.json"
    
    file1.write_text(json.dumps(MOCK_JSON_DATA))
    file2.write_text(json.dumps(MOCK_JSON_DATA))

    output_dir = tmp_path / "output"

    # Mock the command line arguments including the --delete flag
    test_args = [
        "converter.py", 
        str(input_dir / "*.json"), 
        "--output_dir", 
        str(output_dir),
        "--delete"
    ]
    
    with patch.object(sys, 'argv', test_args):
        main()

    # Verify conversions succeeded
    assert (output_dir / "HOCKEY_NHL_3.parquet").exists()
    assert (output_dir / "HOCKEY_NHL_4.parquet").exists()
    
    # Verify original files were deleted
    assert not file1.exists(), "File 1 should have been deleted by the CLI --delete flag"
    assert not file2.exists(), "File 2 should have been deleted by the CLI --delete flag"