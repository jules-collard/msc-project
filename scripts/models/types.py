from typing import Literal

Framework = Literal['xgboost', 'lightgbm', 'logistic']
ImbalanceStrategy = Literal['RO', 'RU', 'WCE', 'None']