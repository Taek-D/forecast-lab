"""XGBoost 모델 래퍼 모듈.

XGBoost를 감싸서 학습, 예측, SHAP 분석을 제공한다.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


class XGBoostModel:
    """XGBoost 모델 래퍼.

    Args:
        params: XGBoost 하이퍼파라미터 딕셔너리. None이면 기본값 사용.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model_: xgb.XGBRegressor | None = None
        self.train_time_ = 0.0
        self.predict_time_ = 0.0
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "XGBoostModel":
        """XGBoost 모델을 학습한다.

        Args:
            X_train: 학습 피처 DataFrame.
            y_train: 학습 타겟 Series.
            X_val: 검증 피처 DataFrame (early stopping용).
            y_val: 검증 타겟 Series.

        Returns:
            self (학습된 모델).
        """
        self.feature_names_ = list(X_train.columns)
        self.model_ = xgb.XGBRegressor(**self.params)

        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]

        start = time.perf_counter()
        self.model_.fit(X_train, y_train, verbose=False, **fit_params)
        self.train_time_ = time.perf_counter() - start

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측값을 반환한다.

        Args:
            X: 피처 DataFrame.

        Returns:
            예측값 배열.
        """
        start = time.perf_counter()
        preds = self.model_.predict(X)
        self.predict_time_ = time.perf_counter() - start
        return preds

    def get_feature_importance(self) -> pd.DataFrame:
        """XGBoost 내장 피처 중요도를 반환한다.

        Returns:
            feature, importance 컬럼의 DataFrame (내림차순).
        """
        importance = self.model_.feature_importances_
        df = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "importance": importance,
            }
        )
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_shap_values(self, X: pd.DataFrame) -> shap.Explanation:
        """SHAP values를 계산한다.

        Args:
            X: 피처 DataFrame.

        Returns:
            shap.Explanation 객체.
        """
        explainer = shap.TreeExplainer(self.model_)
        shap_values = explainer(X)
        return shap_values

    def summary(self) -> dict[str, Any]:
        """모델 요약 정보를 반환한다."""
        return {
            "params": self.params,
            "n_features": len(self.feature_names_),
            "train_time_sec": f"{self.train_time_:.2f}",
            "predict_time_sec": f"{self.predict_time_:.4f}",
        }
