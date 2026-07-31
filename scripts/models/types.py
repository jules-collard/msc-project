from typing import Literal

Framework = Literal['xgboost', 'lightgbm', 'logistic']
ImbalanceStrategy = Literal['SMOTE', 'SMOTENN', 'RU', 'WCE', 'None']