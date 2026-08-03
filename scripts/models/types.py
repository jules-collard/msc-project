from typing import Literal

Framework = Literal['xgboost', 'lightgbm', 'xgboost-dart', 'lightgbm-dart']
ImbalanceStrategy = Literal['RO', 'RU', 'WCE']