import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from scripts.models.calibration import LossCalibratedClassifier


def calibrated_positive_probability(raw_probability: np.ndarray, weight: float) -> np.ndarray:
    raw_probability = np.asarray(raw_probability, dtype=float)
    return raw_probability / (raw_probability + weight * (1.0 - raw_probability))


class FixedProbabilityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, probabilities, class_weight=None, scale_pos_weight=None):
        self.probabilities = probabilities
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        probabilities = np.asarray(self.probabilities, dtype=float)
        if probabilities.ndim == 0:
            probabilities = np.full(X.shape[0], probabilities, dtype=float)
        else:
            probabilities = np.resize(probabilities, X.shape[0])
        return np.column_stack([1.0 - probabilities, probabilities])


@pytest.mark.parametrize(
    "estimator_cls, estimator_kwargs",
    [
        (LogisticRegression, {"max_iter": 1000, "random_state": 0}),
        (RandomForestClassifier, {"n_estimators": 25, "random_state": 0}),
    ],
)
def test_predict_proba_applies_explicit_weight(estimator_cls, estimator_kwargs):
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        random_state=0,
    )

    base_estimator = estimator_cls(**estimator_kwargs)
    wrapped_estimator = LossCalibratedClassifier(estimator_cls(**estimator_kwargs), weight=3.0)

    base_estimator.fit(X, y)
    wrapped_estimator.fit(X, y)

    X_test = X[:15]
    raw_positive = base_estimator.predict_proba(X_test)[:, 1]
    expected_positive = calibrated_positive_probability(raw_positive, 3.0)

    wrapped_proba = wrapped_estimator.predict_proba(X_test)

    np.testing.assert_allclose(wrapped_proba[:, 1], expected_positive)
    np.testing.assert_allclose(wrapped_proba[:, 0], 1.0 - expected_positive)


def test_predict_proba_infers_weight_from_class_weight():
    X = np.zeros((4, 2))
    y = np.array([0, 1, 0, 1])

    estimator = FixedProbabilityClassifier(
        probabilities=[0.10, 0.25, 0.60, 0.90],
        class_weight=4.0,
    )
    wrapped_estimator = LossCalibratedClassifier(estimator)

    wrapped_estimator.fit(X, y)
    wrapped_proba = wrapped_estimator.predict_proba(X)

    expected_positive = calibrated_positive_probability(np.array([0.10, 0.25, 0.60, 0.90]), 4.0)

    np.testing.assert_allclose(wrapped_proba[:, 1], expected_positive)
    np.testing.assert_allclose(wrapped_proba[:, 0], 1.0 - expected_positive)


def test_predict_proba_infers_weight_from_scale_pos_weight():
    X = np.zeros((4, 2))
    y = np.array([0, 1, 0, 1])

    estimator = FixedProbabilityClassifier(
        probabilities=[0.05, 0.40, 0.70, 0.95],
        scale_pos_weight=5.0,
    )
    wrapped_estimator = LossCalibratedClassifier(estimator)

    wrapped_estimator.fit(X, y)
    wrapped_proba = wrapped_estimator.predict_proba(X)

    expected_positive = calibrated_positive_probability(np.array([0.05, 0.40, 0.70, 0.95]), 5.0)

    np.testing.assert_allclose(wrapped_proba[:, 1], expected_positive)
    np.testing.assert_allclose(wrapped_proba[:, 0], 1.0 - expected_positive)


def test_predict_uses_calibrated_probabilities():
    X = np.zeros((4, 2))
    y = np.array([0, 1, 0, 1])

    estimator = FixedProbabilityClassifier(probabilities=[0.20, 0.70, 0.51, 0.49])
    wrapped_estimator = LossCalibratedClassifier(estimator, weight=0.25)

    wrapped_estimator.fit(X, y)

    expected_positive = calibrated_positive_probability(np.array([0.20, 0.70, 0.51, 0.49]), 0.25)
    expected_pred = (expected_positive >= 0.5).astype(int)

    np.testing.assert_array_equal(wrapped_estimator.predict(X), expected_pred)