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
        
        if imbalance_strategy == 'RO':
            steps.append(('sampler', RandomOverSampler(sampling_strategy=params.pop('sampling_strategy', 0.5))))
        elif imbalance_strategy == 'RU':
            steps.append(('sampler', RandomUnderSampler(sampling_strategy=params.pop('sampling_strategy', 0.5))))
        elif imbalance_strategy == 'WCE':
            wce_weight = params.pop('wce_weight')
            params['scale_pos_weight'] = wce_weight

        if framework == 'lightgbm' or framework == 'lightgbm-dart':
            clf = lgb.LGBMClassifier(objective='binary', **params)
        elif framework == 'xgboost' or framework == 'xgboost-dart':
            clf = xgb.XGBClassifier(objective='binary:logistic', **params)
            
        steps.append(('classifier', clf))
        
        return Pipeline(steps)