import numpy as np
import xgboost as xgb
import joblib

from models.types import Framework, ImbalanceStrategy
from models.calibration import Calibrator, LossCalibratedClassifier

class ModelTrainer:

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        framework: Framework,
        strategy: ImbalanceStrategy,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        loss_correct: bool = False,
        calibrate: bool = True,
        seed: int | None = None,
        **params
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.framework = framework
        self.strategy = strategy
        self.loss_correct = loss_correct
        self.calibrate = calibrate
        self.seed = seed
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

        if self.loss_correct:
            clf = LossCalibratedClassifier(clf)
        
        print(f"Training {self.framework} model with strategy {self.strategy}.")
        print(f"Parameters: {self.params}")
        if self.loss_correct:
            print("Loss correction enabled.")
        if self.calibrate:
            print("Calibration enabled.")

        clf.fit(self.X_train, self.y_train)

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
        