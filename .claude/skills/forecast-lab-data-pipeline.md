---
name: forecast-lab-data-pipeline
description: 데이터 로딩, 전처리, Feature Engineering 패턴. Use when loading data, creating features, or modifying the preprocessing pipeline.
---

# Data Pipeline Skill

## 데이터 소스 (Kaggle Store Sales)

| 파일 | 내용 | 주요 컬럼 |
|------|------|-----------|
| train.csv | 매장x상품군x날짜별 매출 | id, date, store_nbr, family, sales, onpromotion |
| stores.csv | 매장 정보 | store_nbr, city, state, type, cluster |
| oil.csv | 일별 유가 | date, dcoilwtico (결측 많음) |
| holidays_events.csv | 공휴일/이벤트 | date, type, locale, locale_name, transferred |
| transactions.csv | 매장별 거래 건수 | date, store_nbr, transactions |

## 전처리 파이프라인 (src/data_loader.py)

```
load_raw_data() → get_top_families() → aggregate_by_family()
    → prepare_oil_data() → prepare_holidays() → aggregate_transactions()
    → merge_external_data() → create_date_features() → split_by_time()
```

실행: `python -c "from src.data_loader import run_pipeline; run_pipeline()"`

## 핵심 함수 규칙

### load_raw_data()
- 경로: `data/raw/` (PROJECT_ROOT 기준)
- parse_dates=["date"] 반드시 포함

### get_top_families(train, n=5)
- 총 매출 기준 내림차순 정렬 → 상위 N개 선정
- 현재 Top 5: GROCERY I, BEVERAGES, PRODUCE, CLEANING, DAIRY

### prepare_oil_data(oil, date_range)
- 유가 결측치: linear interpolation → ffill → bfill
- 전체 날짜 범위로 확장 (누락 날짜 보간)

### prepare_holidays(holidays)
- National 공휴일 + Event만 추출
- transferred=True 제거 (실제 다른 날 쉬므로)
- 날짜별 중복 제거 후 is_holiday=1 플래그

### create_date_features(df)
- 생성 피처: year, month, day, day_of_week, day_name, week_of_year, quarter, is_weekend, is_month_start, is_month_end, day_of_year

### split_by_time(df)
- Train: ~2016-12-31 / Val: 2017-01-01~06-30 / Test: 2017-07-01~
- 반환: (train, val, test) 튜플

## Feature Engineering (src/feature_engineering.py)

아직 미구현. 구현 시 따라야 할 패턴:

```python
def create_lag_features(df, lags=[7, 14, 28, 365]):
    """상품군별 lag feature 생성. 반드시 groupby('family') 후 shift."""

def create_rolling_features(df, windows=[7, 30]):
    """상품군별 rolling 통계. min_periods=window로 미래 유출 방지."""

def create_all_features(df):
    """모든 feature를 한번에 생성하는 통합 함수."""
```

- lag/rolling feature는 반드시 `groupby('family')`로 상품군별 생성
- `shift()` 사용 시 양수 값만 (음수 = 미래 데이터 유출)
- rolling에서 `min_periods=window` 설정하여 불완전한 윈도우 NaN 처리

## 저장된 데이터 (data/processed/)

| 파일 | 내용 |
|------|------|
| full_data.csv | 전체 기간 통합 데이터 |
| train.csv | 학습용 (~2016-12-31) |
| val.csv | 검증용 (2017-01~06) |
| test.csv | 테스트용 (2017-07~08) |
| top_families.csv | 선정된 Top 5 상품군 목록 |

## 주의사항

- 2016-04-16 에콰도르 대지진 전후 이상치 존재 → 별도 처리 필요
- 유가 데이터에 결측치 매우 많음 → 보간 필수
- transactions는 매장별이므로 전체 합산 후 사용
- 에콰도르는 석유 의존 경제 → 유가가 매출에 영향
