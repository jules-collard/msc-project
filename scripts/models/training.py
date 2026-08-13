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
        n_estimators: int = 1500,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        **params
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.framework = framework
        self.strategy = strategy
        self.loss_correct = loss_correct
        self.seed = seed
        self.n_estimators = n_estimators
        self.params = params
        self.X_val = X_val
        self.y_val = y_val

        self.clf = None

    def train(self):
        if 'n_estimators' not in self.params.keys():
            self.params['n_estimators'] = self.n_estimators  # Default value if not provided (i.e. early stopping)

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

        if self.X_val is not None and self.y_val is not None:
            clf = xgb.XGBClassifier(
                objective='binary:logistic',
                enable_categorical=True,
                random_state=self.seed,
                eval_metric='aucpr',
                early_stopping_rounds=50,
                **self.params
            )
        else:
            clf = xgb.XGBClassifier(objective='binary:logistic', enable_categorical=True, random_state=self.seed, **self.params)

        if self.loss_correct:
            clf = LossCalibratedClassifier(clf)
            print("Loss correction enabled.")
        
        print(f"Training {self.framework} model with strategy {self.strategy}.")
        print(f"Parameters: {self.params}")            

        if self.X_val is not None and self.y_val is not None:
            self.clf = clf.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=True)
        else:
            self.clf = clf.fit(self.X_train, self.y_train)

        return clf

    def train_lightgbm(self):
        if self.strategy == 'WCE':
            wce_weight = self.params.pop('wce_weight')
            self.params['scale_pos_weight'] = wce_weight
        else:
            raise NotImplementedError("Strategy not implemented")

        clf = lgb.LGBMClassifier(objective='binary', random_state=self.seed, verbosity=-1, **self.params)

        if self.loss_correct:
            clf = LossCalibratedClassifier(clf)
            print("Loss correction enabled.")
        
        print(f"Training {self.framework} model with strategy {self.strategy}.")
        print(f"Parameters: {self.params}")            

        X_train_ = polars_to_pandas(self.X_train)

        if self.X_val is not None and self.y_val is not None:
            X_val_ = polars_to_pandas(self.X_val)
            early_stopper = lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=True)
            self.clf = clf.fit(X_train_, self.y_train, eval_X=X_val_, eval_y=self.y_val, eval_metric= "average_precision", callbacks=[early_stopper])
        else:
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
        