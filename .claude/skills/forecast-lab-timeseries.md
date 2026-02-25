---
name: forecast-lab-timeseries
description: 시계열 모델링 패턴, 데이터 유출 방지, 평가 가이드. Use when building SARIMA, Prophet, XGBoost models or evaluating forecasts.
---

# Time Series Modeling Skill

## 데이터 스코프

- **Top 5 상품군**: GROCERY I, BEVERAGES, PRODUCE, CLEANING, DAIRY
- 각 상품군은 전체 매장 합산 일별 매출 (1개 시계열/상품군)
- 총 5개 시계열에 집중

## Train/Val/Test 분할 (절대 규칙)

```
Train:      2013-01-01 ~ 2016-12-31
Validation: 2017-01-01 ~ 2017-06-30
Test:       2017-07-01 ~ 2017-08-15
```

- 시간 기준 분할만 허용. 랜덤 분할 = 미래 정보 유출
- XGBoost에서 `TimeSeriesSplit` 사용

## 데이터 유출 방지 체크리스트

- [ ] Lag feature 생성 시 미래 값 참조 없는지 확인
- [ ] Rolling 통계 계산 시 현재 시점 포함 여부 확인 (min_periods 설정)
- [ ] Val/Test 데이터로 학습하지 않는지 확인
- [ ] Feature engineering은 반드시 Train 기간 데이터만으로 파라미터 결정
- [ ] 결측치 보간 시 미래 값 사용하지 않는지 확인 (ffill만 허용, bfill은 Train 내에서만)

## 모델별 핵심 패턴

### SARIMA (statsmodels)
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ACF/PACF로 차수 결정 또는 auto_arima 사용
# order=(p, d, q), seasonal_order=(P, D, Q, m)
# m=7 (주간 계절성) 또는 m=365 (연간)
# 외생변수: exog 파라미터로 유가, 공휴일 등 추가
# 잔차 분석: Ljung-Box test, 정규성 검정
```

### Prophet (prophet)
```python
from prophet import Prophet

# 컬럼명 필수: ds (날짜), y (타겟)
# add_regressor()로 외생변수 추가: oil_price, is_holiday, onpromotion
# changepoint_prior_scale: 트렌드 유연성 (기본 0.05)
# seasonality_prior_scale: 계절성 강도
# Prophet 내장 cross_validation() 사용
```

### XGBoost
```python
import xgboost as xgb

# Feature Engineering:
#   lag: 7, 14, 28, 365일 전 매출
#   rolling: 7일, 30일 이동평균/표준편차
#   날짜: year, month, day_of_week, is_weekend, is_holiday
#   외생: oil_price, onpromotion, total_transactions
# Optuna로 하이퍼파라미터 튜닝
# SHAP으로 Feature Importance 시각화
```

## 평가 지표

```python
# Primary: MAPE (목표 12% 이하)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Secondary: RMSE, MAE
# 반드시 기록: 학습 시간, 추론 시간
# 반드시 기록: 베이스라인(Naive, MA) 대비 개선율
```

## 베이스라인 모델

- **Naive**: 전주 동일 요일 값 사용
- **Moving Average**: 7일, 30일 이동평균
- 모든 모델은 베이스라인보다 나아야 의미 있음

## 앙상블

- 가중 평균 방식: Validation MAPE 역수로 가중치 결정
- 상품군별로 최적 모델이 다를 수 있음 → 상품군별 분석 필수
