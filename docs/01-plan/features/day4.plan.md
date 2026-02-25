# Day 4: Prophet 모델

## 목표

Top 5 상품군에 대해 Prophet 모델을 구축하고 SARIMA/베이스라인 대비 성능을 비교한다.

1. Prophet 기본 피팅 + 체인지포인트 시각화
2. 외생 regressor (유가, 공휴일, 프로모션) 추가
3. 하이퍼파라미터 튜닝 (changepoint_prior_scale, seasonality_prior_scale)
4. Prophet 내장 Cross-Validation
5. Validation 예측 + 신뢰구간 시각화
6. 전 모델 MAPE 비교 → 최적 모델 선정

## 이전 결과 참고

### 베이스라인 (Day 2)

| 상품군 | 최적 베이스라인 | MAPE |
|--------|---------------|------|
| BEVERAGES | Naive_7d | 44.0% |
| CLEANING | MA_30d | 120.0% |
| DAIRY | MA_30d | 65.8% |
| GROCERY I | MA_7d | 98.1% |
| PRODUCE | MA_30d | 62.1% |

### SARIMA (Day 3)

| 상품군 | 최적 모델 | MAPE |
|--------|----------|------|
| BEVERAGES | SARIMAX | 49.7% |
| CLEANING | SARIMAX | 139.8% |
| DAIRY | SARIMAX | 89.5% |
| GROCERY I | SARIMAX | 65.2% |
| PRODUCE | SARIMAX | 79.8% |

## 산출물

| 산출물 | 경로 | 설명 |
|--------|------|------|
| Prophet 노트북 | `notebooks/04_prophet.ipynb` | 피팅 + 튜닝 + CV + 예측 |
| Prophet 모듈 | `src/models/prophet_model.py` | 재사용 가능한 Prophet 래퍼 클래스 |
| 결과 CSV | `outputs/results/prophet_results.csv` | 상품군별 MAPE/RMSE/MAE |
| 시각화 | `outputs/figures/15_prophet_components.png` | 트렌드 + 계절성 분해 |
| 시각화 | `outputs/figures/16_prophet_forecast.png` | 예측 vs 실제 + 신뢰구간 |
| 시각화 | `outputs/figures/17_prophet_cv.png` | Cross-Validation 결과 |

## 작업 상세

### Task 1: src/models/prophet_model.py

재사용 가능한 Prophet 래퍼:

```python
class ProphetModel:
    def __init__(self, changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10.0,
                 holidays_df=None,
                 regressors=None):
        ...

    def fit(self, train_df, family):
        """단일 상품군 학습. ds/y 포맷 변환 자동 처리."""

    def predict(self, periods, val_df=None):
        """n일 예측. 신뢰구간 포함."""

    def cross_validate(self, initial, period, horizon):
        """Prophet 내장 CV."""

    def get_components(self):
        """트렌드/계절성 분해 반환."""

    def summary(self):
        """모델 요약 통계."""
```

- `prophet` 라이브러리 기반
- ds(날짜), y(타겟) 컬럼명 자동 변환
- fit/predict에서 학습/추론 시간 자동 측정
- 외생 regressor: oil_price, is_holiday, onpromotion

### Task 2: Prophet 기본 피팅 + 체인지포인트 (노트북)

각 상품군에 대해:
- 기본 Prophet 피팅 (default 파라미터)
- 체인지포인트 자동 감지 → 시각화
- 트렌드 / 주간 계절성 / 연간 계절성 분해

### Task 3: 외생 regressor 추가

```python
m = Prophet()
m.add_regressor('oil_price')
m.add_regressor('is_holiday')
m.add_regressor('onpromotion')
m.fit(train_df)
```

비교:
1. **Prophet Basic**: 순수 Prophet (외생변수 없음)
2. **Prophet + Regressors**: 외생변수 포함

### Task 4: 하이퍼파라미터 튜닝

주요 파라미터:
- `changepoint_prior_scale`: [0.001, 0.01, 0.05, 0.1, 0.5] (트렌드 유연성)
- `seasonality_prior_scale`: [0.1, 1.0, 10.0] (계절성 강도)
- `seasonality_mode`: ['additive', 'multiplicative']

탐색 방법: Grid Search (조합 수 제한적이므로 가능)
평가 기준: Validation MAPE

### Task 5: Prophet Cross-Validation

```python
from prophet.diagnostics import cross_validation, performance_metrics

cv_results = cross_validation(
    model,
    initial='730 days',    # 2년 학습
    period='90 days',      # 90일마다 CV
    horizon='181 days',    # 6개월 예측
)
perf = performance_metrics(cv_results)
```

- horizon별 MAPE 변화 시각화
- 단기 vs 장기 예측 성능 비교

### Task 6: 예측 + 신뢰구간 + 모델 비교

Validation 기간 (2017-01 ~ 2017-06) 예측:
- 점 예측 + 80% / 95% 신뢰구간
- 실제값과 비교 시각화
- MAPE, RMSE, MAE 계산
- 베이스라인, SARIMA, Prophet 3종 비교표

## 의존성

- Day 2 산출물: `data/processed/` 데이터, `src/evaluation.py`
- Day 3 산출물: `outputs/results/sarima_results.csv`
- 라이브러리: prophet

## 성공 기준

- [ ] 5개 상품군 모두 Prophet 피팅 완료
- [ ] 외생 regressor 포함/미포함 비교 결과 기록
- [ ] 하이퍼파라미터 튜닝 (최소 3개 파라미터)
- [ ] Cross-Validation 실행 + horizon별 성능 시각화
- [ ] Validation MAPE 기록 + 베이스라인/SARIMA 대비 비교
- [ ] prophet_model.py 모듈 구현 완료
- [ ] 노트북에 마크다운 스토리텔링 포함

## 예상 위험

| 위험 | 대응 |
|------|------|
| Prophet 설치 문제 (pystan 의존성) | `pip install prophet` 실패 시 conda 사용 |
| CV가 오래 걸림 (horizon=181일) | initial/period 조정으로 CV 횟수 제한 |
| multiplicative 모드에서 음수 예측 | additive와 비교 후 적합한 모드 선택 |
| SARIMA보다 성능이 낮을 수 있음 | 모델별 장단점 분석으로 인사이트 도출 |
