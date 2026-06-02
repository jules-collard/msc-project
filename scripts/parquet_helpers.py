"""
    Parquet Helpers for SMT Oasis Data
    This module provides helper classes and functions for working with Parquet
    files in the SMT Oasis data context.
"""
import io
import json
from typing import Dict, List, Any

import pyarrow as pa
import pyarrow.parquet as pq


class EntityTrackingReader:
    """
    Reader class for entity tracking data stored in Parquet format.

    This class provides methods to access and manipulate entity tracking data
    that has been serialized as a Parquet file. It can extract both data and
    metadata, and can reconstruct the original nested format.
    """

    def __init__(self, binary_data: bytes):
        """
        Initialize the EntityTrackingReader with binary Parquet data.

        Args:
            binary_data: The binary Parquet data to read
        """
        self.buffer = pa.py_buffer(binary_data)
        self.reader = pq.ParquetFile(self.buffer)
        self.table = self.reader.read()
        self._extract_metadata()

    def _extract_metadata(self):
        """
        Extract metadata from the Parquet file's metadata section.
        """
        # Read key-value metadata from the Parquet file
        metadata = self.reader.metadata.metadata
        if metadata is None or b'habshub_metadata' not in metadata:
            raise ValueError("No habshub_metadata found in Parquet file")

        # Decode and parse JSON metadata
        self.habshub_metadata = json.loads(
            metadata[b'habshub_metadata'].decode('utf-8')
        )

        # Extract specific metadata components
        self.meta_stats = self.habshub_metadata.get('meta_stats', {})
        self.data_stats = self.habshub_metadata.get('data_stats', {})
        self.filter_stats = self.habshub_metadata.get('filter_stats', {})
        self.measurement_defs = self.habshub_metadata.get(
            'entity_processed_measurements_defs',
            []
        )

    def get_table(self) -> pa.Table:
        """
        Get the PyArrow table containing all entity tracking data.

        Returns:
            The PyArrow table
        """
        return self.table

    def filter_entity(self, entity_id: str) -> pa.Table:
        """
        Filter the table to only include rows for a specific entity.

        Args:
            entity_id: The entity ID to filter by

        Returns:
            A PyArrow table filtered by entity_id
        """
        mask = pa.compute.equal(self.table['entity_id'], entity_id)
        return self.table.filter(mask)

    def get_entities(self) -> List[str]:
        """
        Get a list of all entity IDs in the dataset.

        Returns:
            A list of unique entity IDs
        """
        return self.table['entity_id'].unique().to_pylist()

    def to_dict_format(self) -> Dict[str, Any]:
        """
        Convert the Parquet data back to the original nested dictionary format.

        Returns:
            A dictionary containing the entity tracking data in the original
            format
        """
        # Use PyArrow directly without pandas
        entities = {}

        # Get unique entity_ids
        entity_ids = self.get_entities()

        for entity_id in entity_ids:
            # Filter table for this entity
            entity_table = self.filter_entity(entity_id)

            # Get the official_id for this entity (all rows have same value)
            official_id = entity_table['entity_official_id'][0].as_py()

            # Create measurements list
            measurements = []

            # Convert to dict of arrays first
            data_by_columns = {
                col: entity_table[col].to_pylist()
                for col in entity_table.column_names
            }

            # Create a list of (index, timestamp) tuples for sorting
            indices_with_ts = [
                (i, ts)
                for i, ts in enumerate(data_by_columns['ts'])
            ]
            indices_with_ts.sort(key=lambda x: x[1])  # Sort by timestamp

            # Create sorted measurements using the sorted indices
            for idx, _ in indices_with_ts:
                # Extract values for each field in the correct order
                measurement_values = [
                    data_by_columns[field][idx]
                    for field in self.measurement_defs
                ]
                measurements.append(tuple(measurement_values))

            entities[entity_id] = {
                'entity_id': entity_id,
                'entity_official_id': official_id,
                'measurements': measurements
            }

        # Reconstruct the full original format
        return {
            'entity_processed_measurements': entities,
            'entity_processed_measurements_defs': self.measurement_defs,
            'meta_stats': self.meta_stats,
            'data_stats': self.data_stats,
            'filter_stats': self.filter_stats
        }

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get the metadata from the Parquet file.

        Returns:
            A dictionary containing all metadata
        """
        return self.habshub_metadata


def read_entity_tracking_parquet(binary_data: bytes) -> Dict[str, Any]:
    """
    Convenience function to read entity tracking data from a Parquet file
    and return it in the original format.

    Args:
        binary_data: The binary Parquet data to read

    Returns:
        The entity tracking data in the original dictionary format
    """
    reader = EntityTrackingReader(binary_data)
    return reader.to_dict_format()


def create_entity_tracking_parquet(data: Dict[str, Any]) -> bytes:
    """
    Convert entity tracking data from original format to Parquet binary format.

    Args:
        data: The entity tracking data in the original dictionary format

    Returns:
        Binary Parquet data
    """
    entity_data = data['entity_processed_measurements']
    measurement_defs = data['entity_processed_measurements_defs']
    metadata = {
        'meta_stats': data['meta_stats'],
        'data_stats': data['data_stats'],
        'filter_stats': data['filter_stats'],
        'entity_processed_measurements_defs': measurement_defs
    }

    # Create lists for each column
    columns = {
        'entity_id': [],
        'entity_official_id': [],
    }

    # Initialize columns for each measurement field
    for field in measurement_defs:
        columns[field] = []

    # Flatten the nested entity data
    for entity_id, entity_info in entity_data.items():
        official_id = entity_info['entity_official_id']
        for measurement in entity_info['measurements']:
            columns['entity_id'].append(entity_id)
            columns['entity_official_id'].append(official_id)

            # Add each measurement field
            for i, field in enumerate(measurement_defs):
                columns[field].append(measurement[i])

    # Create PyArrow arrays for each column
    arrays = []
    fields = []

    # Entity IDs (string type)
    arrays.append(pa.array(columns['entity_id'], type=pa.string()))
    fields.append(pa.field('entity_id', pa.string()))

    arrays.append(pa.array(columns['entity_official_id'], type=pa.string()))
    fields.append(pa.field('entity_official_id', pa.string()))

    # Add measurement fields with appropriate types
    for field in measurement_defs:
        if field in ('period', 'segment_idx', 'clock_state'):
            # Integer fields
            arrays.append(pa.array(columns[field], type=pa.int32()))
            fields.append(pa.field(field, pa.int32()))
        else:
            # Float fields (timestamps, positions, velocities, etc.)
            arrays.append(pa.array(columns[field], type=pa.float64()))
            fields.append(pa.field(field, pa.float64()))

    # Create table and write to buffer
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))

    # Create a buffer and write the parquet data to it
    buf = io.BytesIO()

    # Set metadata on the schema instead of passing to write_table
    schema_with_metadata = table.schema.with_metadata({
        b'habshub_metadata': json.dumps(metadata).encode('utf-8')
    })
    table_with_metadata = table.cast(schema_with_metadata)

    pq.write_table(
        table=table_with_metadata,
        where=buf,
        # Compression parameters
        compression='zstd',
        compression_level=3,
        use_dictionary=True
    )

    # Return the buffer contents as bytes
    buf.seek(0)
    return buf.read()
