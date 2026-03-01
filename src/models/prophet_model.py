"""Prophet 모델 래퍼 모듈.

Facebook Prophet을 감싸서 학습, 예측, Cross-Validation을 제공한다.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


class ProphetModel:
    """Prophet 모델 래퍼.

    Args:
        changepoint_prior_scale: 트렌드 유연성 (기본 0.05).
        seasonality_prior_scale: 계절성 강도 (기본 10.0).
        seasonality_mode: 'additive' 또는 'multiplicative'.
        regressors: 외생변수 컬럼명 리스트. None이면 순수 Prophet.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = "additive",
        regressors: List[str] | None = None,
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.regressors = regressors
        self.model_: Prophet | None = None
        self.train_df_: pd.DataFrame | None = None
        self.family_ = ""
        self.train_time_ = 0.0
        self.predict_time_ = 0.0

    def _build_model(self) -> Prophet:
        """Prophet 모델 인스턴스를 생성한다."""
        m = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        if self.regressors:
            for reg in self.regressors:
                m.add_regressor(reg)
        return m

    def _prepare_df(self, df: pd.DataFrame, family: str) -> pd.DataFrame:
        """Prophet 형식 (ds, y + regressors)으로 변환한다."""
        sub = df[df["family"] == family].sort_values("date").copy()
        prophet_df = pd.DataFrame({"ds": sub["date"], "y": sub["sales"]})
        if self.regressors:
            for reg in self.regressors:
                prophet_df[reg] = sub[reg].values
        prophet_df = prophet_df.reset_index(drop=True)
        return prophet_df

    def fit(self, train_df: pd.DataFrame, family: str) -> "ProphetModel":
        """단일 상품군 시계열에 대해 Prophet 모델을 학습한다.

        Args:
            train_df: date, family, sales 컬럼이 있는 Train DataFrame.
            family: 학습할 상품군 이름.

        Returns:
            self (학습된 모델).
        """
        self.family_ = family
        self.train_df_ = self._prepare_df(train_df, family)
        self.model_ = self._build_model()

        start = time.perf_counter()
        self.model_.fit(self.train_df_)
        self.train_time_ = time.perf_counter() - start

        return self

    def predict(
        self,
        periods: int,
        val_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """미래 periods일에 대해 예측한다.

        Args:
            periods: 예측할 일수.
            val_df: 외생변수가 필요한 경우 Validation DataFrame.

        Returns:
            ds, yhat, yhat_lower, yhat_upper 컬럼의 DataFrame.
        """
        start = time.perf_counter()

        if self.regressors and val_df is not None:
            future = self._prepare_df(val_df, self.family_).drop(columns=["y"])
            future = future.iloc[:periods]
        else:
            future = self.model_.make_future_dataframe(periods=periods)
            future = future.tail(periods).reset_index(drop=True)

        forecast = self.model_.predict(future)
        self.predict_time_ = time.perf_counter() - start

        result_df = pd.DataFrame(
            {
                "date": forecast["ds"].values,
                "forecast": forecast["yhat"].values,
                "lower_ci": forecast["yhat_lower"].values,
                "upper_ci": forecast["yhat_upper"].values,
            }
        )
        return result_df

    def cross_validate(
        self,
        initial: str = "730 days",
        period: str = "90 days",
        horizon: str = "181 days",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prophet 내장 Cross-Validation을 실행한다.

        Args:
            initial: 초기 학습 기간.
            period: CV 간격.
            horizon: 예측 기간.

        Returns:
            (cv_results, performance_metrics) 튜플.
        """
        cv_results = cross_validation(
            self.model_,
            initial=initial,
            period=period,
            horizon=horizon,
        )
        perf = performance_metrics(cv_results, rolling_window=0.1)
        return cv_results, perf

    def get_components(self) -> pd.DataFrame:
        """트렌드/계절성 분해를 반환한다."""
        future = self.model_.make_future_dataframe(periods=0)
        return self.model_.predict(future)

    def summary(self) -> str:
        """모델 요약 정보를 반환한다."""
        params = {
            "family": self.family_,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "seasonality_mode": self.seasonality_mode,
            "regressors": self.regressors or [],
            "train_time_sec": f"{self.train_time_:.2f}",
        }
        return str(params)
