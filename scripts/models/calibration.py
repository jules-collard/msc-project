from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


class Calibrator:
    @staticmethod
    def calibrate(pipeline, X_val, y_val, method='isotonic'):
        """
        Calibrate the given pipeline using the validation data.
        """
        calibrated_clf = CalibratedClassifierCV(
            estimator=FrozenEstimator(pipeline),
            method=method,
            ensemble=False 
        )
        calibrated_clf.fit(X_val, y_val)
        return calibrated_clf