"""Input-selection adapters for frozen state estimators.

B2 uses elapsed time; B3 uses 14 motion descriptors.
The adapter preserves the estimator class ordering.
"""

import numpy as np


class FeatureAdapter:
    """Select frozen input columns without altering the simulator."""

    def __init__(self, model, columns):
        self.model = model
        self.columns = tuple(columns)
        self.classes_ = np.asarray(model.classes_)

        assert np.array_equal(self.classes_, [0, 1, 2])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)

        assert X.ndim == 2
        assert X.shape[1] == 15

        selected = X[:, self.columns]
        result = self.model.predict_proba(selected)

        assert result.shape == (X.shape[0], 3)
        return result
