"""데이터 로드 및 전처리 파이프라인.

Kaggle "Store Sales - Time Series Forecasting" 데이터를 로드하고,
Top 5 상품군 x 전체 매장 합산 시계열로 변환하는 파이프라인.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

# ─── 경로 설정 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ─── 시간 분할 기준 ─────────────────────────────────────────
TRAIN_END = "2016-12-31"
VAL_START = "2017-01-01"
VAL_END = "2017-06-30"
TEST_START = "2017-07-01"


def load_raw_data(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Kaggle 원본 데이터 파일들을 로드한다.

    Args:
        data_dir: raw 데이터 디렉토리 경로. None이면 기본 경로 사용.

    Returns:
        파일명을 키로 하는 DataFrame 딕셔너리.
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    data = {}
    data["train"] = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    data["stores"] = pd.read_csv(data_dir / "stores.csv")
    data["oil"] = pd.read_csv(data_dir / "oil.csv", parse_dates=["date"])
    data["holidays"] = pd.read_csv(
        data_dir / "holidays_events.csv", parse_dates=["date"]
    )
    data["transactions"] = pd.read_csv(
        data_dir / "transactions.csv", parse_dates=["date"]
    )

    return data


def get_top_families(train: pd.DataFrame, n: int = 5) -> List[str]:
    """총 매출 기준 Top N 상품군을 선정한다.

    Args:
        train: 원본 train DataFrame.
        n: 선정할 상품군 수.

    Returns:
        상품군 이름 리스트 (매출 내림차순).
    """
    family_sales = train.groupby("family")["sales"].sum().sort_values(ascending=False)
    return family_sales.head(n).index.tolist()


def aggregate_by_family(
    train: pd.DataFrame, families: List[str]
) -> pd.DataFrame:
    """선정된 상품군에 대해 전체 매장 합산 일별 매출을 계산한다.

    Args:
        train: 원본 train DataFrame.
        families: 대상 상품군 리스트.

    Returns:
        date, family, sales, onpromotion 컬럼을 가진 DataFrame.
    """
    filtered = train[train["family"].isin(families)].copy()
    agg = (
        filtered.groupby(["date", "family"])
        .agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
        .reset_index()
    )
    return agg.sort_values(["family", "date"]).reset_index(drop=True)


def prepare_oil_data(
    oil: pd.DataFrame, date_range: pd.DatetimeIndex | None = None
) -> pd.DataFrame:
    """유가 데이터를 전처리한다.

    결측치를 선형 보간으로 채우고, 필요 시 전체 날짜 범위로 확장한다.

    Args:
        oil: 원본 유가 DataFrame.
        date_range: 맞출 날짜 범위. None이면 oil 자체 범위 사용.

    Returns:
        date, oil_price 컬럼의 DataFrame.
    """
    oil = oil.copy()
    oil = oil.rename(columns={"dcoilwtico": "oil_price"})
    oil = oil.dropna(subset=["date"])

    if date_range is not None:
        full_dates = pd.DataFrame({"date": date_range})
        oil = full_dates.merge(oil, on="date", how="left")
    else:
        oil = oil.sort_values("date").reset_index(drop=True)

    oil["oil_price"] = (
        oil["oil_price"].interpolate(method="linear").ffill().bfill()
    )
    return oil


def prepare_holidays(holidays: pd.DataFrame) -> pd.DataFrame:
    """공휴일 데이터를 전처리한다.

    National 공휴일과 Event만 추출하고, transferred된 공휴일은 제거한다.

    Args:
        holidays: 원본 공휴일 DataFrame.

    Returns:
        date, is_holiday, holiday_type 컬럼의 DataFrame.
    """
    holidays = holidays.copy()

    # National 공휴일 + Event만 (Local/Regional은 매장별이므로 합산 시 노이즈)
    national = holidays[
        (holidays["locale"] == "National") | (holidays["type"] == "Event")
    ].copy()

    # transferred=True인 공휴일은 실제로 다른 날 쉬므로 제거
    national = national[national["transferred"] != True]  # noqa: E712

    # 날짜별 중복 제거 (같은 날 여러 이벤트)
    holiday_flags = (
        national.groupby("date")
        .agg(
            is_holiday=("type", "count"),
            holiday_type=("type", "first"),
        )
        .reset_index()
    )
    holiday_flags["is_holiday"] = 1

    return holiday_flags[["date", "is_holiday", "holiday_type"]]


def aggregate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """거래 건수를 전체 매장 합산으로 집계한다.

    Args:
        transactions: 원본 거래 DataFrame.

    Returns:
        date, total_transactions 컬럼의 DataFrame.
    """
    agg = (
        transactions.groupby("date")["transactions"]
        .sum()
        .reset_index()
        .rename(columns={"transactions": "total_transactions"})
    )
    return agg


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """날짜 기반 피처를 생성한다.

    Args:
        df: date 컬럼이 있는 DataFrame.

    Returns:
        날짜 피처가 추가된 DataFrame.
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon, 6=Sun
    df["day_name"] = df["date"].dt.day_name()
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    return df


def merge_external_data(
    sales: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """매출 데이터에 외생변수를 병합한다.

    Args:
        sales: 집계된 매출 DataFrame (date, family, sales, onpromotion).
        oil: 전처리된 유가 DataFrame.
        holidays: 전처리된 공휴일 DataFrame.
        transactions: 집계된 거래 DataFrame.

    Returns:
        외생변수가 병합된 DataFrame.
    """
    df = sales.copy()
    df = df.merge(oil[["date", "oil_price"]], on="date", how="left")
    df = df.merge(
        holidays[["date", "is_holiday", "holiday_type"]], on="date", how="left"
    )
    df = df.merge(
        transactions[["date", "total_transactions"]], on="date", how="left"
    )

    # 결측 채우기
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    df["holiday_type"] = df["holiday_type"].fillna("None")
    df["oil_price"] = df["oil_price"].interpolate(method="linear").ffill().bfill()
    df["total_transactions"] = df.groupby("family")[
        "total_transactions"
    ].transform(lambda x: x.interpolate(method="linear").ffill().bfill())

    return df


def split_by_time(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """시간 기준으로 Train/Validation/Test를 분할한다.

    - Train: ~ 2016-12-31
    - Validation: 2017-01-01 ~ 2017-06-30
    - Test: 2017-07-01 ~

    Args:
        df: date 컬럼이 있는 DataFrame.

    Returns:
        (train, val, test) 튜플.
    """
    train = df[df["date"] <= TRAIN_END].copy()
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
    test = df[df["date"] >= TEST_START].copy()
    return train, val, test


def run_pipeline(top_n: int = 5, save: bool = True) -> dict:
    """전체 전처리 파이프라인을 실행한다.

    Args:
        top_n: 선정할 Top N 상품군 수.
        save: True이면 processed 디렉토리에 저장.

    Returns:
        처리된 데이터를 담은 딕셔너리.
    """
    print("=" * 60)
    print("수요예측 데이터 전처리 파이프라인")
    print("=" * 60)

    # 1. 원본 데이터 로드
    print("\n[1/7] 원본 데이터 로드...")
    raw = load_raw_data()
    for name, df in raw.items():
        print(f"  {name}: {df.shape}")

    # 2. Top N 상품군 선정
    print(f"\n[2/7] Top {top_n} 상품군 선정...")
    top_families = get_top_families(raw["train"], n=top_n)
    print(f"  선정: {top_families}")

    # 3. 매장 합산 집계
    print("\n[3/7] 상품군별 매장 합산...")
    aggregated = aggregate_by_family(raw["train"], top_families)
    print(f"  결과: {aggregated.shape}")

    # 4. 외생변수 전처리
    print("\n[4/7] 외생변수 전처리...")
    date_range = pd.date_range(
        aggregated["date"].min(), aggregated["date"].max(), freq="D"
    )
    oil = prepare_oil_data(raw["oil"], date_range)
    holidays = prepare_holidays(raw["holidays"])
    transactions = aggregate_transactions(raw["transactions"])
    print(f"  유가: {oil.shape}, 공휴일: {holidays.shape}, 거래: {transactions.shape}")

    # 5. 데이터 병합
    print("\n[5/7] 외생변수 병합...")
    full = merge_external_data(aggregated, oil, holidays, transactions)
    print(f"  결과: {full.shape}")

    # 6. 날짜 피처 생성
    print("\n[6/7] 날짜 피처 생성...")
    full = create_date_features(full)
    print(f"  컬럼: {list(full.columns)}")

    # 7. Train/Val/Test 분할
    print("\n[7/7] 시간 기준 분할...")
    train, val, test = split_by_time(full)
    print(f"  Train: {train.shape} ({train['date'].min()} ~ {train['date'].max()})")
    print(f"  Val:   {val.shape} ({val['date'].min()} ~ {val['date'].max()})")
    print(f"  Test:  {test.shape} ({test['date'].min()} ~ {test['date'].max()})")

    # 저장
    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        full.to_csv(PROCESSED_DATA_DIR / "full_data.csv", index=False)
        train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        val.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
        test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
        pd.Series(top_families).to_csv(
            PROCESSED_DATA_DIR / "top_families.csv", index=False, header=["family"]
        )
        print(f"\n저장 완료: {PROCESSED_DATA_DIR}")

    print("\n" + "=" * 60)
    print("파이프라인 완료!")
    print("=" * 60)

    return {
        "raw": raw,
        "top_families": top_families,
        "aggregated": aggregated,
        "full": full,
        "train": train,
        "val": val,
        "test": test,
    }


if __name__ == "__main__":
    result = run_pipeline()
