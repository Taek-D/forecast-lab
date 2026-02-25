# Day 2: 시계열 분해 + 베이스라인 모델

## 목표

Day 1에서 전처리한 Top 5 상품군 시계열 데이터에 대해:
1. 시계열 분해(STL Decomposition)로 Trend/Seasonal/Residual 구조 파악
2. 정상성 검정(ADF, KPSS)으로 차분 필요 여부 결정
3. 베이스라인 모델(Naive, Moving Average) 구축 후 MAPE 기록
4. Day 3~5 모델 성능 비교의 기준선 확보

## 산출물

| 산출물 | 경로 | 설명 |
|--------|------|------|
| 분해 노트북 | `notebooks/02_decomposition.ipynb` | STL 분해 + 정상성 검정 + 베이스라인 |
| 평가 모듈 | `src/evaluation.py` | MAPE, RMSE, MAE 계산 함수 |
| Feature 모듈 | `src/feature_engineering.py` | Lag, Rolling, 날짜 피처 생성 |
| 베이스라인 결과 | `outputs/results/baseline_results.csv` | 모델별 MAPE/RMSE/MAE |
| 시각화 | `outputs/figures/07_stl_*.png` | STL 분해 그래프 |

## 작업 상세

### Task 1: STL Decomposition

**입력**: `data/processed/full_data.csv` (Top 5 상품군 일별 매출)

각 상품군에 대해:
- `statsmodels.tsa.seasonal.STL` 사용
- `period=7` (주간 계절성) 기본, `period=365` (연간)도 확인
- Trend, Seasonal, Residual 3개 컴포넌트 시각화
- 계절성 강도(Strength of Seasonality) 수치 계산

**출력**: 상품군별 STL 분해 그래프 (outputs/figures/)

### Task 2: 정상성 검정

각 상품군의 원본 시계열 + 1차 차분에 대해:
- **ADF Test** (Augmented Dickey-Fuller): p-value < 0.05 → 정상
- **KPSS Test**: p-value > 0.05 → 정상
- 두 검정 결과가 상충할 경우 → 차분 후 재검정

**출력**: 상품군별 검정 결과 요약표

| 상품군 | ADF p-value | KPSS p-value | 차분 필요 | 권장 d |
|--------|-------------|--------------|----------|--------|
| GROCERY I | ? | ? | ? | ? |
| ... | | | | |

### Task 3: src/evaluation.py 구현

```python
def mape(y_true, y_pred) -> float
def rmse(y_true, y_pred) -> float
def mae(y_true, y_pred) -> float
def evaluate_model(y_true, y_pred, model_name) -> dict
def compare_models(results: list[dict]) -> pd.DataFrame
```

- MAPE는 y_true == 0인 경우 제외 (0 나누기 방지)
- evaluate_model은 모든 지표 + 학습/추론 시간 포함

### Task 4: 베이스라인 모델

**Validation 기간** (2017-01-01 ~ 2017-06-30)에서 평가:

1. **Naive (Seasonal)**: 7일 전 동일 요일 매출 사용
   - `y_pred[t] = y_actual[t - 7]`
2. **Moving Average 7일**: 직전 7일 평균
   - `y_pred[t] = mean(y[t-7 : t])`
3. **Moving Average 30일**: 직전 30일 평균
   - `y_pred[t] = mean(y[t-30 : t])`

**출력**: `outputs/results/baseline_results.csv`

| 모델 | 상품군 | MAPE | RMSE | MAE |
|------|--------|------|------|-----|
| Naive_7d | GROCERY I | ? | ? | ? |
| MA_7d | GROCERY I | ? | ? | ? |
| MA_30d | GROCERY I | ? | ? | ? |

### Task 5: src/feature_engineering.py 구현

Day 3~5에서 사용할 Feature Engineering 함수 사전 준비:

```python
def create_lag_features(df, target, lags=[7, 14, 28, 365]) -> pd.DataFrame
def create_rolling_features(df, target, windows=[7, 30]) -> pd.DataFrame
def create_all_features(df) -> pd.DataFrame
```

- 반드시 `groupby('family')` 후 생성 (상품군별 독립)
- `shift()` 양수 값만 사용 (미래 유출 방지)
- `rolling(min_periods=window)` 설정

## 의존성

- Day 1 산출물: `data/processed/` 데이터 (train.csv, val.csv, full_data.csv)
- 추가 라이브러리: 이미 requirements.txt에 포함 (statsmodels, scikit-learn)

## 성공 기준

- [ ] 5개 상품군 모두 STL 분해 완료 + 시각화 저장
- [ ] ADF/KPSS 검정 결과 요약표 작성
- [ ] evaluation.py에 MAPE/RMSE/MAE 함수 구현 + 동작 확인
- [ ] 3개 베이스라인 모델의 Validation MAPE 기록
- [ ] feature_engineering.py 기본 구조 구현
- [ ] 노트북에 마크다운 스토리텔링 포함 (분석 → 해석 → 결론 흐름)

## 예상 위험

| 위험 | 대응 |
|------|------|
| STL period 설정이 데이터에 안 맞을 수 있음 | period=7, 30, 365 모두 시도 후 비교 |
| ADF/KPSS 결과 상충 | 차분 후 재검정, 시각적 확인 병행 |
| 베이스라인 MAPE가 너무 높으면 목표(12%) 비현실적 | 상품군별 난이도 차이 확인, 목표 재조정 고려 |
| 2016.04 지진 이후 구간이 Residual을 왜곡 | 지진 전후 별도 분석 또는 더미 변수 처리 |
