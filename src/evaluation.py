"""모델 평가 지표 계산 모듈.

MAPE, RMSE, MAE 등 시계열 예측 모델의 성능을 평가하고 비교하는 함수를 제공한다.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error를 계산한다.

    Args:
        y_true: 실제 값 배열.
        y_pred: 예측 값 배열.

    Returns:
        MAPE (%). y_true == 0인 경우는 제외한다.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error를 계산한다.

    Args:
        y_true: 실제 값 배열.
        y_pred: 예측 값 배열.

    Returns:
        RMSE 값.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error를 계산한다.

    Args:
        y_true: 실제 값 배열.
        y_pred: 예측 값 배열.

    Returns:
        MAE 값.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    family: str = "",
    train_time: float = 0.0,
    predict_time: float = 0.0,
) -> dict:
    """모델 예측 결과를 종합 평가한다.

    Args:
        y_true: 실제 값 배열.
        y_pred: 예측 값 배열.
        model_name: 모델 이름.
        family: 상품군 이름.
        train_time: 학습 소요 시간 (초).
        predict_time: 추론 소요 시간 (초).

    Returns:
        모델명, 상품군, MAPE, RMSE, MAE, 학습/추론 시간을 담은 딕셔너리.
    """
    return {
        "model": model_name,
        "family": family,
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """여러 모델의 평가 결과를 비교 테이블로 정리한다.

    Args:
        results: evaluate_model()의 반환값 리스트.

    Returns:
        모델별 지표 비교 DataFrame.
    """
    df = pd.DataFrame(results)
    df = df.sort_values(["family", "mape"]).reset_index(drop=True)
    return df


def time_function(func: Callable, *args, **kwargs) -> tuple:
    """함수 실행 시간을 측정한다.

    Args:
        func: 실행할 함수.
        *args: 함수 인자.
        **kwargs: 함수 키워드 인자.

    Returns:
        (함수 반환값, 실행 시간(초)) 튜플.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed
