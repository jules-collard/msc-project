import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold


class DataSplitter:

    def __init__(self, data: pl.DataFrame, feature_cols: list[str], target_col: str, group_col: str = 'game_id'):
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.game_ids = data.select(group_col).unique().to_numpy().flatten()

    def get_split(self, val_size: float = 0.2, test_size: float = 0.2, random_state: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training, validation and test sets based on the specified validation size.
        The split is done in a way that ensures that all samples from the same game are kept together.
        """
        assert 0 < val_size < 1
        assert 0 < test_size < 1
        assert val_size + test_size < 1

        train_ids, not_train_ids = train_test_split(self.game_ids, test_size=val_size + test_size, random_state=random_state)
        val_ids, test_ids = train_test_split(not_train_ids, test_size=test_size / (val_size + test_size), random_state=random_state)

        train_data = self.data.filter(pl.col('game_id').is_in(train_ids))
        val_data = self.data.filter(pl.col('game_id').is_in(val_ids))
        test_data = self.data.filter(pl.col('game_id').is_in(test_ids))

        X_train, y_train = self._extract_features_and_target(train_data)
        X_val, y_val = self._extract_features_and_target(val_data)
        X_test, y_test = self._extract_features_and_target(test_data)

        groups = train_data.select('game_id').to_numpy().flatten()

        return X_train, y_train, X_val, y_val, X_test, y_test, groups
    
    def _extract_features_and_target(self, data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract features and target from the given DataFrame.
        """
        X = data.select(self.feature_cols).to_numpy()
        y = data.select(self.target_col).to_numpy().flatten()
        return X, y