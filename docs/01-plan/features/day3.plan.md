# Day 3: SARIMA 모델

## 목표

Top 5 상품군에 대해 SARIMAX 모델을 구축하고 베이스라인 대비 개선 여부를 확인한다.

1. ACF/PACF 분석으로 SARIMA 차수 탐색
2. auto_arima로 최적 차수 자동 결정
3. 외생변수(유가, 공휴일, 프로모션) 포함/미포함 비교
4. 잔차 분석으로 모델 적합성 검증
5. 예측 + 신뢰구간 시각화
6. Validation MAPE 기록 → 베이스라인 대비 개선율 산출

## Day 2 베이스라인 참고

| 상품군 | 최적 베이스라인 | MAPE |
|--------|---------------|------|
| BEVERAGES | Naive_7d | 44.0% |
| CLEANING | MA_30d | 120.0% |
| DAIRY | MA_30d | 65.8% |
| GROCERY I | MA_7d | 98.1% |
| PRODUCE | MA_30d | 62.1% |

## 산출물

| 산출물 | 경로 | 설명 |
|--------|------|------|
| SARIMA 노트북 | `notebooks/03_sarima.ipynb` | ACF/PACF + 모델 피팅 + 잔차 분석 + 예측 |
| SARIMA 모듈 | `src/models/sarima_model.py` | 재사용 가능한 SARIMA 래퍼 클래스 |
| 결과 CSV | `outputs/results/sarima_results.csv` | 상품군별 MAPE/RMSE/MAE |
| 시각화 | `outputs/figures/12_acf_pacf.png` | ACF/PACF 그래프 |
| 시각화 | `outputs/figures/13_sarima_forecast.png` | 예측 vs 실제 + 신뢰구간 |
| 시각화 | `outputs/figures/14_sarima_residuals.png` | 잔차 분석 4종 |

## 작업 상세

### Task 1: src/models/sarima_model.py

재사용 가능한 SARIMA 래퍼:

```python
class SARIMAModel:
    def __init__(self, order, seasonal_order, exog_cols=None):
        ...

    def fit(self, train_df, family):
        """단일 상품군 학습. 시간 측정 포함."""

    def predict(self, steps, exog=None):
        """n-step 예측. 신뢰구간 포함."""

    def get_residuals(self):
        """잔차 반환."""

    def summary(self):
        """모델 요약 통계."""
```

- `statsmodels.tsa.statespace.sarimax.SARIMAX` 기반
- 외생변수(exog): oil_price, is_holiday, onpromotion, total_transactions
- fit/predict에서 학습/추론 시간 자동 측정

### Task 2: ACF/PACF 분석 (노트북 셀)

각 상품군의 1차 차분 시계열에 대해:
- **ACF (Autocorrelation Function)**: MA 차수(q) 힌트
- **PACF (Partial ACF)**: AR 차수(p) 힌트
- lag=40까지 시각화

Day 2 정상성 검정 결과 반영:
- 대부분 d=1 권장
- 주간 계절성 m=7

### Task 3: auto_arima로 최적 차수 결정

```python
from pmdarima import auto_arima

# 상품군별 최적 차수 탐색
result = auto_arima(
    y=train_ts,
    seasonal=True,
    m=7,
    d=1,
    D=1,
    max_p=3, max_q=3,
    max_P=2, max_Q=2,
    stepwise=True,
    suppress_warnings=True,
    information_criterion='aic',
)
```

- stepwise=True로 시간 단축 (Full Grid는 너무 오래 걸림)
- 5개 상품군 각각에 대해 실행
- 결과: 상품군별 최적 (p,d,q)(P,D,Q,m) 기록

### Task 4: SARIMAX 모델 학습

각 상품군에 대해 2가지 버전 비교:
1. **SARIMA (외생변수 없음)**: 순수 시계열 모델
2. **SARIMAX (외생변수 포함)**: oil_price, is_holiday, onpromotion

비교 지표: AIC, MAPE, RMSE

### Task 5: 잔차 분석

모델 적합성 검증 4종:
1. **잔차 시계열**: 패턴 없이 랜덤해야 함
2. **잔차 히스토그램 + Q-Q plot**: 정규분포에 가까워야 함
3. **잔차 ACF**: 자기상관 없어야 함
4. **Ljung-Box 검정**: p > 0.05 → 잔차가 백색잡음

### Task 6: 예측 + 신뢰구간

Validation 기간 (2017-01 ~ 2017-06) 예측:
- 점 예측 + 80% / 95% 신뢰구간
- 실제값과 비교 시각화
- MAPE, RMSE, MAE 계산 → baseline_results와 비교

## 의존성

- Day 2 산출물: `data/processed/` 데이터, `src/evaluation.py`
- 라이브러리: statsmodels (SARIMAX), pmdarima (auto_arima)

## 성공 기준

- [ ] 5개 상품군 모두 auto_arima 차수 결정 완료
- [ ] SARIMA vs SARIMAX 비교 결과 기록
- [ ] 잔차 분석 4종 시각화 완료
- [ ] Validation MAPE 기록 + 베이스라인 대비 개선율
- [ ] sarima_model.py 모듈 구현 완료
- [ ] 노트북에 마크다운 스토리텔링 포함

## 예상 위험

| 위험 | 대응 |
|------|------|
| auto_arima 실행이 오래 걸림 (m=7) | stepwise=True, max_order 제한 |
| 일부 상품군에서 수렴 실패 | 차수를 수동 지정하고 재시도 |
| SARIMAX에서 외생변수가 오히려 성능 저하 | SARIMA와 비교하여 더 나은 모델 선택 |
| 베이스라인 MAPE가 높아서 개선율이 과장될 수 있음 | RMSE, MAE도 함께 비교 |
