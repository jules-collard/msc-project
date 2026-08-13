import polars as pl
import pandas as pd
from polars import col as c
import numpy as np
from sklearn.model_selection import train_test_split

from models.types import ShotType, Handedness


class DataSplitter:

    def __init__(
        self,
        data:pl.DataFrame,
        feature_cols: list[str],
        target_col: str,
        group_col: str = 'game_id',
        split_path: str | None = None,
        seed: int | None = None
    ):
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.game_ids = data.select(group_col).unique().to_numpy().flatten()
        self.seed = seed

        if split_path is not None:
            self.load_split(split_path)

    def split(self, test_size: float = 0.3) -> None:
        """
        Split the data into training and test/validation sets based on the specified test size.
        The split is done in a way that ensures that all samples from the same game are kept together.
        """
        assert 0 < test_size < 1

        train_ids, test_ids = train_test_split(self.game_ids, test_size=test_size, random_state=self.seed)

        self.train_ids = train_ids
        self.test_ids = test_ids

    def get_split_data(self) -> tuple[pl.DataFrame, np.ndarray, pl.DataFrame, np.ndarray, np.ndarray]:
        """
        Get the training and test data based on the previously split game IDs.
        """
        if self.train_ids is None or self.test_ids is None:
            print("No train/test split found. Performing a new split with default parameters.")
            self.split()

        train_data = self.data.filter(c('game_id').is_in(self.train_ids))
        test_data = self.data.filter(c('game_id').is_in(self.test_ids))

        X_train, y_train = self.extract_features_and_target(train_data, self.feature_cols, self.target_col)
        X_test, y_test = self.extract_features_and_target(test_data, self.feature_cols, self.target_col)

        groups = train_data.select('game_id').to_numpy().flatten()

        return X_train, y_train, X_test, y_test, groups

    def load_split(self, path: str) -> None:
        """
        Load the split data from a file.
        """
        print(f"Loading train/test split from {path}...")
        data = np.load(path, allow_pickle=True)
        self.train_ids = data['train_ids']
        self.test_ids = data['test_ids']

    def save_split(self, path: str) -> None:
        """
        Save the current split of game IDs to a file.
        """
        np.savez(path, train_ids=self.train_ids, test_ids=self.test_ids)

    def get_ids(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        train_ids_df = self.data.filter(c('game_id').is_in(self.train_ids)).select(c('game_id', 'period', 'shot_id'))
        test_ids_df = self.data.filter(c('game_id').is_in(self.test_ids)).select(c('game_id', 'period', 'shot_id'))
        return train_ids_df, test_ids_df

    @staticmethod
    def extract_features_and_target(data: pl.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pl.DataFrame, np.ndarray]:
        """
        Extract features and target from the given DataFrame.
        """
        X = data.select(feature_cols)
        y = data.select(target_col).to_numpy().flatten()
        return X, y

def polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    return df.with_columns(pl.selectors.by_dtype(pl.Boolean).cast(pl.Int16)).to_pandas()

def prepare_data(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return (
        data
        .with_columns(
            c('shot_x').fill_null(c('x_adj_coord')),
            c('shot_y').fill_null(c('y_adj_coord')),
            c('shot_type').cast(ShotType),
            c('goalie_handedness', 'shooter_handedness').cast(Handedness),
            pl.when(
                c('goalie_handedness') == 'R'
            ).then(
                -c('goalline_y')
            ).otherwise(c('goalline_y'))
            .alias('goalline_y_norm')
        ).filter(
            c('opposing_team_goalie_on_ice_ref').is_not_null(), # No empty nets
            c('shot_x') >= 0,
        )
    )

def post_shot_filter(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return (
        data
        .filter(
            c('shot_speed').is_not_null(),
            c('goalline_y_norm').is_not_null(),
            c('goalline_z').is_not_null(),
            c('shot_x') < 89, # No shots from behind the net
            c('type').str.contains('blocked').not_() # Only unblocked shots for post-shot model
        )
    )