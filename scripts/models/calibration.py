import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.utils.validation import check_is_fitted


class Calibrator:
    @staticmethod
    def calibrate(pipeline, X_val, y_val, method='sigmoid'):
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


class LossCalibratedClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """
    A custom scikit-learn meta-estimator that wraps a binary classifier 
    and analytically corrects the output probabilities for class imbalance weights.
    """
    def __init__(self, estimator, weight=None):
        self.estimator = estimator
        self.weight = weight

    def fit(self, X, y, **fit_params):
        """
        Clones the base estimator, fits it to the training data, and sets attributes.
        """
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        self.classes_ = getattr(self.estimator_, "classes_")

        if self.weight is not None:
            self.weight_ = self.weight
        elif hasattr(self.estimator_, "scale_pos_weight"): # XGBoost and LightGBM
            self.weight_ = getattr(self.estimator_, "scale_pos_weight", 1.0)
        elif hasattr(self.estimator_, "class_weight_"): # Sklearn Logistic Regression
            cw = getattr(self.estimator_, "class_weight_")
            if isinstance(cw, dict):
                # Calculate ratio: weight of class 1 / weight of class 0
                weight_0 = cw.get(self.classes_[0], 1.0)
                weight_1 = cw.get(self.classes_[1], 1.0)
                self.weight_ = weight_1 / weight_0 if weight_0 != 0 else 1.0
            elif cw == 'balanced':
                # Calculate ratio based on actual sample counts in 'y'
                n_pos = np.sum(y == self.classes_[1])
                n_neg = np.sum(y == self.classes_[0])
                self.weight_ = n_neg / n_pos if n_pos > 0 else 1.0
        else:
            raise ValueError("Weight must be provided or the estimator must have 'scale_pos_weight' or 'class_weight_' attribute.")
        
        self.is_fitted_ = True
        
        return self

    def predict_proba(self, X):
        """
        Extracts raw probabilities and applies the mathematical prior correction.
        """
        check_is_fitted(self)
        
        raw_probs = self.estimator_.predict_proba(X)
        p_raw = raw_probs[:, 1]
        
        p_calibrated = p_raw / (p_raw + self.weight_ * (1 - p_raw))
        
        return np.vstack([1 - p_calibrated, p_calibrated]).T

    def predict(self, X):
        """
        Converts calibrated probabilities back into discrete class predictions (0 or 1).
        """
        calibrated_probs = self.predict_proba(X)
        return self.classes_[np.argmax(calibrated_probs, axis=1)]
