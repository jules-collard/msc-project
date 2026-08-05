import numpy as np
import xgboost as xgb
import joblib

from models.types import Framework, ImbalanceStrategy
from models.calibration import Calibrator

class ModelTrainer:

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        framework: Framework,
        strategy: ImbalanceStrategy,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        calibrate: bool = True,
        seed: int | None = None,
        feature_names: list[str] | None = None,
        **params
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.framework = framework
        self.strategy = strategy
        self.calibrate = calibrate
        self.seed = seed
        self.feature_names = feature_names
        self.params = params

        self.clf = None

        if self.calibrate and (X_val is None or y_val is None):
            raise ValueError("Validation data is required when calibration is enabled")

    def train(self):
        if self.framework == 'xgboost':
            return self.train_xgboost()
        else:
            raise NotImplementedError("Framework not implemented")

    def train_xgboost(self):
        if self.strategy == 'WCE':
            wce_weight = self.params.pop('wce_weight')
            self.params['scale_pos_weight'] = wce_weight
        else:
            raise NotImplementedError("Strategy not implemented")

        clf = xgb.XGBClassifier(objective='binary:logistic', random_state=self.seed, **self.params)
        print(f"Training {self.framework} model with strategy {self.strategy} and parameters: {self.params}")
        clf.fit(self.X_train, self.y_train)

        if self.feature_names is not None:
            clf.get_booster().feature_names = self.feature_names
        if self.calibrate:
            print("Calibrating model...")
            clf = Calibrator.calibrate(clf, self.X_val, self.y_val)

        self.clf = clf

        return clf

    def save(self, path):
        if self.clf is None:
            raise ValueError("No model to save")

        joblib.dump(self.clf, path)
        print("Model saved to ", path)

    @staticmethod
    def load(path):
        return joblib.load(path)
        