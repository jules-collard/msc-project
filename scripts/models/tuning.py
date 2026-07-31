import numpy as np
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import cross_val_score, GroupKFold
from optuna.trial import Trial

from models.types import Framework, ImbalanceStrategy
from models.pipelines import PipelineBuilder


class OptunaObjective:
    def __init__(self, X, y, groups, cv_splitter, framework, imbalance_strategy):
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.groups: np.ndarray = groups
        self.cv: GroupKFold = cv_splitter
        self.framework: Framework = framework
        self.imbalance_strategy: ImbalanceStrategy = imbalance_strategy

    def __call__(self, trial: Trial) -> float:
        params = self._get_param_space(trial, self.framework)

        if self.imbalance_strategy in ['SMOTE', 'SMOTENN', 'RU']:
            params['sampling_strategy'] = trial.suggest_float('sampling_strategy', 0.2, 1.0) # 20% to 50% split
        elif self.imbalance_strategy == 'WCE':
            params['wce_weight'] = trial.suggest_float('wce_weight', 1, 10)

        pipeline = PipelineBuilder.build(self.framework, self.imbalance_strategy, params)
        
        # 4. Score using Cross-Validation (Average Precision = PR-AUC)
        pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')
        scores = cross_val_score(
            pipeline, self.X, self.y, 
            groups=self.groups, 
            cv=self.cv, 
            scoring=pr_auc_scorer,
            n_jobs=-1 # Run folds in parallel
        )
        
        return scores.mean()
        
    def _get_param_space(self, trial: Trial, framework: Framework) -> dict:
        if framework == 'xgboost':
            return self._get_xgb_param_space(trial)
        elif framework == 'lightgbm':
            return self._get_lgb_param_space(trial)
        elif framework == 'logistic':
            return self._get_logistic_param_space(trial)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def _get_xgb_param_space(self, trial: Trial) -> dict:
        params = {
            # Core Structure
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            
            # Regularization & Overfitting Prevention
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            
            # Stochastic Elements
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),

            'verbosity': 0,
        }
        return params

    def _get_lgb_param_space(self, trial):
        params = {
            # Core Structure
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            
            # Regularization & Overfitting Prevention
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
            
            # Stochastic Elements
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),

            'verbosity': -1,
        }
        return params

    def _get_logistic_param_space(self, trial: Trial):
        return {
            'l1_ratio': 0.0 # L2 regularisation
        }