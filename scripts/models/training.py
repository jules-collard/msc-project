import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb
import joblib

from models.types import Framework, ImbalanceStrategy
from models.calibration import LossCalibratedClassifier
from models.data import polars_to_pandas

class ModelTrainer:

    def __init__(
        self,
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        framework: Framework,
        strategy: ImbalanceStrategy,
        loss_correct: bool = True,
        seed: int | None = None,
        **params
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.framework = framework
        self.strategy = strategy
        self.loss_correct = loss_correct
        self.seed = seed
        self.params = params

        self.clf = None

    def train(self):
        if self.framework == 'xgboost':
            return self.train_xgboost()
        elif self.framework == 'lightgbm':
            return self.train_lightgbm()
        else:
            raise NotImplementedError("Framework not implemented")

    def train_xgboost(self):
        if self.strategy == 'WCE':
            wce_weight = self.params.pop('wce_weight')
            self.params['scale_pos_weight'] = wce_weight
        else:
            raise NotImplementedError("Strategy not implemented")

        clf = xgb.XGBClassifier(objective='binary:logistic', enable_categorical=True, random_state=self.seed, **self.params)

        if self.loss_correct:
            clf = LossCalibratedClassifier(clf)
            print("Loss correction enabled.")
        
        print(f"Training {self.framework} model with strategy {self.strategy}.")
        print(f"Parameters: {self.params}")            

        self.clf = clf.fit(self.X_train, self.y_train)

        return clf

    def train_lightgbm(self):
        if self.strategy == 'WCE':
            wce_weight = self.params.pop('wce_weight')
            self.params['scale_pos_weight'] = wce_weight
        else:
            raise NotImplementedError("Strategy not implemented")

        clf = lgb.LGBMClassifier(objective='binary', random_state=self.seed, **self.params)

        if self.loss_correct:
            clf = LossCalibratedClassifier(clf)
            print("Loss correction enabled.")
        
        print(f"Training {self.framework} model with strategy {self.strategy}.")
        print(f"Parameters: {self.params}")            

        X_train_ = polars_to_pandas(self.X_train)
        self.clf = clf.fit(X_train_, self.y_train)

        return clf

    def save(self, path):
        if self.clf is None:
            raise ValueError("No model to save")

        joblib.dump(self.clf, path)
        print("Model saved to ", path)

    @staticmethod
    def load(path):
        return joblib.load(path)
        