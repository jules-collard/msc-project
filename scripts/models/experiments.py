import polars as pl
import optuna
from sklearn.model_selection import GroupKFold

from models.data import DataSplitter
from models.tuning import OptunaObjective
from models.pipelines import PipelineBuilder


class ExperimentRunner:
    def __init__(
        self,
        data: pl.DataFrame,
        feature_cols: list[str],
        target_col: str,
        **kwargs
    ):
        self.data_splitter = DataSplitter(data, feature_cols, target_col, **kwargs)
        self.cv = GroupKFold(n_splits=3, shuffle=True, random_state=78)
        self.best_model_info = None

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.groups_train = self.data_splitter.get_split_data()

    def run_all(self):
        frameworks = ['logistic', 'xgboost', 'lightgbm']
        strategies = ['SMOTE', 'RU', 'WCE', 'None']
        
        best_score = 0
        results = []
        
        for framework in frameworks:
            for strategy in strategies:
                print(f"Training {framework} with {strategy}...")

                # Find optimal hyperparameters for given setup
                objective = OptunaObjective(self.X_train, self.y_train, self.groups_train, self.cv, framework, strategy)
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=30)

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
                
                results.append(info)
                    
        return results

    def run_best(self):
        if not self.best_model_info:
            raise ValueError("No best model info available. Run run_all() first.")
        
        framework = self.best_model_info['framework']
        strategy = self.best_model_info['strategy']
        best_params = self.best_model_info['best_params']

        print(f"Retraining best model: {framework} with {strategy}...")
        pipeline = PipelineBuilder.build(framework, strategy, best_params)
        pipeline.fit(self.X_train, self.y_train)
        
        return pipeline  # Return the trained pipeline for further evaluation