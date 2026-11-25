"""Regla baseline interpretable basada en severidad inspeccionada y volumen vehicular."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class SeverityTrafficRuleBaseline(BaseEstimator, ClassifierMixin):
    """Clasificador baseline sin fuga basado en severidad reportada y tráfico diario."""

    classes_ = np.array(["baja", "media", "alta"])

    def fit(self, X: pd.DataFrame, y=None):
        traffic = pd.to_numeric(X["trafico_diario"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(traffic) >= 3:
            self.q33_, self.q67_ = np.percentile(traffic, [33.3, 66.7])
        else:
            self.q33_, self.q67_ = 3000.0, 6500.0
        self.median_sev_ = float(
            pd.to_numeric(X["severidad_inspeccionada"], errors="coerce").median()
        )
        if np.isnan(self.median_sev_):
            self.median_sev_ = 1.0
        self.classes_ = np.array(["baja", "media", "alta"])
        return self

    def _scores(self, X: pd.DataFrame) -> np.ndarray:
        traffic = pd.to_numeric(X["trafico_diario"], errors="coerce").fillna(self.q33_).to_numpy(dtype=float)
        sev = pd.to_numeric(X["severidad_inspeccionada"], errors="coerce").fillna(self.median_sev_).to_numpy(dtype=float)
        tier = np.where(traffic > self.q67_, 2.0, np.where(traffic > self.q33_, 1.0, 0.0))
        return sev + 0.8 * tier

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._scores(X)
        idx = np.digitize(scores, [1.9, 3.5])
        return self.classes_[idx]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._scores(X)
        centers = np.array([1.0, 2.7, 4.3])
        dists = -((scores[:, None] - centers[None, :]) ** 2)
        exp_d = np.exp(dists - np.max(dists, axis=1, keepdims=True))
        return exp_d / np.sum(exp_d, axis=1, keepdims=True)
