import lightgbm as lgb
import xgboost as xgb
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression

from models.types import Framework, ImbalanceStrategy


class PipelineBuilder:
    @staticmethod
    def build(
        framework: Framework,
        imbalance_strategy: ImbalanceStrategy,
        params: dict,
    ):
        steps = []
        
        # 1. Inject the Sampling Strategy
        # if imbalance_strategy == 'SMOTE':
        #     steps.append(('sampler', SMOTE(sampling_strategy=params.pop('sampling_strategy'))))
        # elif imbalance_strategy == 'SMOTENN':
        #     steps.append(('sampler', SMOTEENN(sampling_strategy=params.pop('sampling_strategy'))))
        if imbalance_strategy == 'RO':
            steps.append(('sampler', RandomOverSampler(sampling_strategy=params.pop('sampling_strategy', 0.5))))
        elif imbalance_strategy == 'RU':
            steps.append(('sampler', RandomUnderSampler(sampling_strategy=params.pop('sampling_strategy', 0.5))))
        elif imbalance_strategy == 'WCE':
            wce_weight = params.pop('wce_weight')
            if framework == 'lightgbm' or framework == 'xgboost':
                params['scale_pos_weight'] = wce_weight
            else:
                params['class_weight'] = {0: 1.0, 1: wce_weight}

        if framework == 'lightgbm':
            clf = lgb.LGBMClassifier(objective='binary', **params)
        elif framework == 'xgboost':
            clf = xgb.XGBClassifier(objective='binary:logistic', **params)
        elif framework == 'logistic':
            clf = LogisticRegression(**params)
            
        steps.append(('classifier', clf))
        
        return Pipeline(steps)