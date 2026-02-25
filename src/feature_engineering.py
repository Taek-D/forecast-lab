"""피처 엔지니어링 모듈.

시계열 모델링에 필요한 Lag, Rolling, 날짜 기반 피처를 생성한다.
모든 피처는 상품군(family)별로 독립적으로 생성되어 미래 데이터 유출을 방지한다.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List


def create_lag_features(
    df: pd.DataFrame,
    target: str = "sales",
    lags: List[int] | None = None,
) -> pd.DataFrame:
    """상품군별 Lag 피처를 생성한다.

    Args:
        df: date, family, target 컬럼이 있는 DataFrame.
        target: 타겟 컬럼명.
        lags: Lag 일수 리스트. None이면 [7, 14, 28, 365].

    Returns:
        Lag 피처가 추가된 DataFrame.
    """
    if lags is None:
        lags = [7, 14, 28, 365]

    df = df.copy()
    df = df.sort_values(["family", "date"]).reset_index(drop=True)

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("family")[target].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    target: str = "sales",
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """상품군별 Rolling 통계 피처를 생성한다.

    Args:
        df: date, family, target 컬럼이 있는 DataFrame.
        target: 타겟 컬럼명.
        windows: 윈도우 크기 리스트. None이면 [7, 30].

    Returns:
        Rolling 피처가 추가된 DataFrame.
    """
    if windows is None:
        windows = [7, 30]

    df = df.copy()
    df = df.sort_values(["family", "date"]).reset_index(drop=True)

    for window in windows:
        # shift(1) 후 rolling: 현재 시점 미포함 (미래 유출 방지)
        shifted = df.groupby("family")[target].shift(1)

        df[f"rolling_mean_{window}"] = shifted.groupby(
            df["family"]
        ).transform(lambda x: x.rolling(window=window, min_periods=window).mean())

        df[f"rolling_std_{window}"] = shifted.groupby(
            df["family"]
        ).transform(lambda x: x.rolling(window=window, min_periods=window).std())

    return df


def create_ewm_features(
    df: pd.DataFrame,
    target: str = "sales",
    spans: List[int] | None = None,
) -> pd.DataFrame:
    """상품군별 지수가중이동평균(EWM) 피처를 생성한다.

    Args:
        df: date, family, target 컬럼이 있는 DataFrame.
        target: 타겟 컬럼명.
        spans: EWM span 리스트. None이면 [7, 30].

    Returns:
        EWM 피처가 추가된 DataFrame.
    """
    if spans is None:
        spans = [7, 30]

    df = df.copy()
    df = df.sort_values(["family", "date"]).reset_index(drop=True)

    for span in spans:
        shifted = df.groupby("family")[target].shift(1)
        df[f"ewm_mean_{span}"] = shifted.groupby(df["family"]).transform(
            lambda x: x.ewm(span=span, min_periods=span).mean()
        )

    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """모든 피처를 통합 생성한다.

    Lag, Rolling, EWM 피처를 순서대로 생성하여 반환한다.
    date, family, sales 컬럼이 필수이며, 날짜 피처는 data_loader에서
    이미 생성된 것을 가정한다.

    Args:
        df: 전처리된 DataFrame (data_loader.create_date_features 적용 후).

    Returns:
        모든 피처가 추가된 DataFrame.
    """
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_ewm_features(df)
    return df
