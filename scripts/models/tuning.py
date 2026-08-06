import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, make_scorer, roc_auc_score
from sklearn.model_selection import cross_validate, GroupKFold
from optuna.trial import Trial

from models.types import Framework, ImbalanceStrategy
from models.pipelines import PipelineBuilder


class OptunaObjective:
    def __init__(self, X, y, groups, cv_splitter, framework, imbalance_strategy, seed: int | None = None):
        self.X: pl.DataFrame = X
        self.y: np.ndarray = y
        self.groups: np.ndarray = groups
        self.cv: GroupKFold = cv_splitter
        self.framework: Framework = framework
        self.imbalance_strategy: ImbalanceStrategy = imbalance_strategy
        self.seed = seed

    def __call__(self, trial: Trial) -> float:
        params = self._get_param_space(trial, self.framework)

        if self.imbalance_strategy in ['RU', 'RO']:
            params['sampling_strategy'] = trial.suggest_float('sampling_strategy', 0.2, 1.0) # 20% to 50% split
        elif self.imbalance_strategy == 'WCE':
            params['wce_weight'] = trial.suggest_float('wce_weight', 1.0, 5.0)

        pipeline = PipelineBuilder.build(self.framework, self.imbalance_strategy, params, seed=self.seed)
        
        pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')
        roc_auc_scorer = make_scorer(roc_auc_score, response_method='predict_proba')
        cv_results = cross_validate(
            pipeline, self.X, self.y, 
            groups=self.groups, 
            cv=self.cv, 
            scoring={'pr_auc': pr_auc_scorer, 'roc_auc': roc_auc_scorer},
            error_score='raise'
        )

        mean_pr_auc = cv_results['test_pr_auc'].mean()
        mean_roc_auc = cv_results['test_roc_auc'].mean()

        trial.set_user_attr("roc_auc", mean_roc_auc) # Strore ROC AUC for later analysis
        
        return mean_pr_auc
        
    def _get_param_space(self, trial: Trial, framework: Framework) -> dict:
        if framework == 'xgboost':
            return self._get_xgb_param_space(trial)
        elif framework == 'lightgbm':
            return self._get_lgb_param_space(trial)
        elif framework == 'xgboost-dart':
            return self._get_xgb_dart_param_space(trial)
        elif framework == 'lightgbm-dart':
            return self._get_lgb_dart_param_space(trial)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def _get_xgb_param_space(self, trial: Trial, max_estimators=1000, min_lr=1e-3) -> dict:
        params = {
            # Core Structure
            'n_estimators': trial.suggest_int('n_estimators', 100, max_estimators),
            'learning_rate': trial.suggest_float('learning_rate', min_lr, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            
            # Regularization & Overfitting Prevention
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True), # Min. gain to split
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True), # L2 regularisation
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True), # L1 regularisation
            
            # Stochastic Elements
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),

            'verbosity': 0,
            'n_jobs': 6
        }
        return params

    def _get_xgb_dart_param_space(self, trial: Trial) -> dict:
        params = self._get_xgb_param_space(trial, max_estimators=500, min_lr=3e-3)
        params['rate_drop'] = trial.suggest_float('rate_drop', 0.05, 0.3)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.2, 0.8)
        return params

    def _get_lgb_param_space(self, trial, min_lr=1e-3):
        params = {
            # Core Structure
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', min_lr, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            
            # Regularization & Overfitting Prevention
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 150),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-4, 1, log=True),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 10.0, log=True),
            
            # Stochastic Elements
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 8),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.8, 1.0),

            'verbosity': -1,
        }
        return params

    def _get_lgb_dart_param_space(self, trial: Trial) -> dict:
        params: dict = self._get_lgb_param_space(trial, min_lr=3e-3)
        params['boosting'] = 'dart'
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.05, 0.3)
        return params