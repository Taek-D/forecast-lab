"""SARIMAX 모델 래퍼 모듈.

statsmodels SARIMAX를 감싸서 학습, 예측, 잔차 분석을 제공한다.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAModel:
    """SARIMAX 모델 래퍼.

    Args:
        order: (p, d, q) 비계절 차수.
        seasonal_order: (P, D, Q, m) 계절 차수.
        exog_cols: 외생변수 컬럼명 리스트. None이면 순수 SARIMA.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        exog_cols: List[str] | None = None,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_cols = exog_cols
        self.model_ = None
        self.result_ = None
        self.train_time_ = 0.0
        self.predict_time_ = 0.0
        self.family_ = ""

    def fit(self, train_df: pd.DataFrame, family: str) -> "SARIMAModel":
        """단일 상품군 시계열에 대해 SARIMAX 모델을 학습한다.

        Args:
            train_df: date, family, sales 컬럼이 있는 Train DataFrame.
            family: 학습할 상품군 이름.

        Returns:
            self (학습된 모델).
        """
        self.family_ = family
        sub = train_df[train_df["family"] == family].sort_values("date")
        ts = sub.set_index("date")["sales"].asfreq("D")
        ts = ts.interpolate(method="linear")

        exog = None
        if self.exog_cols:
            exog_df = sub.set_index("date")[self.exog_cols].asfreq("D")
            exog_df = exog_df.interpolate(method="linear").ffill().bfill()
            exog = exog_df

        self.model_ = SARIMAX(
            ts,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        start = time.perf_counter()
        self.result_ = self.model_.fit(disp=False, maxiter=200)
        self.train_time_ = time.perf_counter() - start

        return self

    def predict(
        self,
        steps: int,
        val_df: pd.DataFrame | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """미래 steps일에 대해 예측한다.

        Args:
            steps: 예측할 일수.
            val_df: 외생변수가 필요한 경우 Validation DataFrame.
            alpha: 신뢰구간 유의수준 (기본 0.05 → 95% CI).

        Returns:
            date, forecast, lower_ci, upper_ci 컬럼의 DataFrame.
        """
        exog_future = None
        if self.exog_cols and val_df is not None:
            sub = val_df[val_df["family"] == self.family_].sort_values("date")
            exog_future = sub.set_index("date")[self.exog_cols].iloc[:steps]
            exog_future = exog_future.interpolate(method="linear").ffill().bfill()

        start = time.perf_counter()
        forecast = self.result_.get_forecast(steps=steps, exog=exog_future, alpha=alpha)
        self.predict_time_ = time.perf_counter() - start

        pred_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=alpha)

        result_df = pd.DataFrame(
            {
                "date": pred_mean.index,
                "forecast": pred_mean.values,
                "lower_ci": conf_int.iloc[:, 0].values,
                "upper_ci": conf_int.iloc[:, 1].values,
            }
        )
        return result_df

    def get_residuals(self) -> pd.Series:
        """학습된 모델의 잔차를 반환한다."""
        return self.result_.resid

    def get_aic(self) -> float:
        """모델의 AIC를 반환한다."""
        return self.result_.aic

    def summary(self) -> str:
        """모델 요약 통계를 반환한다."""
        return str(self.result_.summary())
