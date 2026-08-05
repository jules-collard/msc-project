import polars as pl
import optuna
from sklearn.model_selection import GroupKFold

from models.data import DataSplitter
from models.tuning import OptunaObjective
from models.pipelines import PipelineBuilder
from models.calibration import Calibrator
from models.types import Framework, ImbalanceStrategy


class ExperimentRunner:
    def __init__(
        self,
        data: pl.DataFrame,
        feature_cols: list[str],
        target_col: str,
        seed: int | None = None,
        **kwargs
    ):
        self.data_splitter = DataSplitter(data, feature_cols, target_col, seed=seed, **kwargs)
        self.cv = GroupKFold(n_splits=3, shuffle=True, random_state=seed)
        self.best_model_info = None
        self.results = []
        self.seed = seed

        self.X_train, self.y_train, self.X_val, self.y_val, self.groups_train = self.data_splitter.get_split_data()

    def run_all(
        self,
        frameworks: list[Framework] = ['xgboost-dart', 'lightgbm-dart', 'lightgbm', 'xgboost'],
        strategies: list[ImbalanceStrategy] = ['RO', 'RU', 'WCE'],
        n_trials: int = 50,
        multivariate: bool = False
    ):
        best_score = 0
        self.results = []
        
        for framework in frameworks:
            for strategy in strategies:
                print(f"Training {framework} with {strategy}...")

                # Find optimal hyperparameters for given setup
                sampler = optuna.samplers.TPESampler(multivariate=multivariate, seed=self.seed)
                objective = OptunaObjective(self.X_train, self.y_train, self.groups_train, self.cv, framework, strategy, seed=self.seed)
                study = optuna.create_study(direction='maximize', sampler=sampler)
                study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)

                info = {
                    'framework': framework,
                    'strategy': strategy,
                    'best_pr_auc': study.best_value,
                    'best_roc_auc': study.best_trial.user_attrs.get("roc_auc", None),
                    'best_params': study.best_params
                }

                if study.best_value > best_score:
                    best_score = study.best_value
                    self.best_model_info = info
                
                self.results.append(info)
                    
        return self.results