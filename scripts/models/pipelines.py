import lightgbm as lgb
import xgboost as xgb
from imblearn.combine import SMOTENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression

from models.types import Framework, ImbalanceStrategy


class PipelineBuilder:
    @staticmethod
    def build(
        framework: Framework,
        imbalance_strategy: ImbalanceStrategy,
        params: dict
    ):
        steps = []
        
        # 1. Inject the Sampling Strategy
        if imbalance_strategy == 'SMOTE':
            steps.append(('sampler', SMOTE(sampling_strategy=params.pop('sampling_strategy'))))
        elif imbalance_strategy == 'SMOTENN':
            steps.append(('sampler', SMOTENN(sampling_strategy=params.pop('sampling_strategy'))))
        elif imbalance_strategy == 'RU':
            steps.append(('sampler', RandomUnderSampler(sampling_strategy=params.pop('sampling_strategy', 0.5))))
        elif imbalance_strategy == 'WCE':
            wce_weight = params.pop('wce_weight')

        if framework == 'LightGBM':
            clf = lgb.LGBMClassifier(objective='binary', scale_pos_weight=wce_weight, **params)
        elif framework == 'XGBoost':
            clf = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=wce_weight, **params)
        elif framework == 'LogisticRegression':
            clf = LogisticRegression(**params, class_weight={0: 1.0, 1: wce_weight})
            
        steps.append(('classifier', clf))
        
        return Pipeline(steps)