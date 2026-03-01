# ForecastLab: E-commerce 수요 예측 시스템

## 프로젝트 개요
- **목표**: Prophet vs SARIMA vs XGBoost 3개 모델 비교로 주간 매출 예측 오차 MAPE 12% 이하 달성
- **데이터**: Kaggle "Store Sales - Time Series Forecasting" (에콰도르 Favorita 매장 매출, 2013~2017)
- **기술 스택**: Python, Prophet, statsmodels(SARIMA), XGBoost, Streamlit
- **배포**: Streamlit Cloud 또는 Vercel
- **GitHub**: forecast-lab (새 레포)

## 프로젝트 경로
```
E:\프로젝트\부족한 프로젝트 시작\수요예측 시스템
```

## 폴더 구조
```
forecast-lab/
├── CLAUDE.md                    # 이 파일
├── README.md                    # 프로젝트 소개 + 결과 요약
├── requirements.txt             # 의존성
├── .gitignore
├── data/
│   ├── raw/                     # Kaggle 원본 데이터 (git 제외)
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── stores.csv
│   │   ├── oil.csv
│   │   ├── holidays_events.csv
│   │   └── transactions.csv
│   └── processed/               # 전처리된 데이터 (git 제외)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_decomposition.ipynb
│   ├── 03_sarima.ipynb
│   ├── 04_prophet.ipynb
│   ├── 05_xgboost.ipynb
│   └── 06_model_comparison.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # 데이터 로드 + 전처리
│   ├── feature_engineering.py   # 피처 생성 (lag, rolling, 날짜 등)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sarima_model.py
│   │   ├── prophet_model.py
│   │   └── xgboost_model.py
│   ├── evaluation.py            # MAPE, RMSE, MAE 등 평가
│   └── utils.py                 # 유틸리티 함수
├── app/
│   ├── app.py                   # Streamlit 메인
│   ├── pages/
│   │   ├── 1_eda.py
│   │   ├── 2_model_comparison.py
│   │   ├── 3_forecast.py
│   │   ├── 4_feature_importance.py
│   │   └── 5_inventory_simulation.py
│   └── .streamlit/
│       └── config.toml
├── models/                      # 학습된 모델 저장 (git 제외)
│   ├── sarima/
│   ├── prophet/
│   └── xgboost/
├── outputs/
│   ├── figures/                 # 주요 시각화 PNG
│   └── results/                 # 모델 결과 CSV
└── docs/
    └── screenshots/             # Notion/README용 스크린샷
```

## 데이터 설명
- **Kaggle 대회**: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
- **train.csv**: 매장×상품군×날짜별 매출 (id, date, store_nbr, family, sales, onpromotion)
- **stores.csv**: 매장 정보 (store_nbr, city, state, type, cluster)
- **oil.csv**: 일별 유가 (date, dcoilwtico) — 에콰도르는 석유 의존 경제
- **holidays_events.csv**: 공휴일/이벤트 (date, type, locale, locale_name, description, transferred)
- **transactions.csv**: 매장별 일별 거래 건수

## 분석 스코프 결정
- 54개 매장 × 33개 상품군 = 1,782개 시계열 → **전부 하지 않음**
- **Top 5 상품군 × 전체 매장 합산 = 5개 시계열**로 집중
- 이유: 포트폴리오는 깊이가 중요. 1,782개 시계열 돌리는 건 compute 낭비.
- README에 "왜 이렇게 스코프를 잡았는가" 반드시 명시

## Train/Validation/Test 분할
- **Train**: 2013-01-01 ~ 2016-12-31
- **Validation**: 2017-01-01 ~ 2017-06-30
- **Test**: 2017-07-01 ~ 2017-08-15 (대회 test 기간)
- ⚠️ 시간 기준 분할 필수. 랜덤 분할 절대 금지.

## Day-by-Day 실행 가이드

### Day 1: 데이터 탐색 + 전처리
- [x] Kaggle 데이터 다운로드 → data/raw/
- [x] 01_eda.ipynb 작성 (36셀: 20 code + 16 markdown)
  - 매출 시계열 시각화 (전체/매장별/상품군별)
  - 계절성 패턴 확인 (요일, 월, 연간)
  - 프로모션 효과 시각화
  - 결측치/이상치 탐지 (지진 2016.04.16 전후)
  - 상관관계 분석 (유가 vs 매출, 거래건수 vs 매출)
- [x] 전처리 파이프라인 (src/data_loader.py)
  - 날짜 피처 생성 (요일, 월, 분기, 공휴일 플래그)
  - 외생변수 정리 (유가 보간, 공휴일 인코딩)
  - Top 5 상품군 선정 + 매장 합산
  - Train/Val/Test 분할
- [x] EDA 시각화 6장 저장 (outputs/figures/)
- [x] 전처리 데이터 저장 (data/processed/) — full_data, train, val, test, top_families

### Day 2: 시계열 분해 + 베이스라인
- [x] 02_decomposition.ipynb
  - STL Decomposition (Trend / Seasonal / Residual)
  - 계절성 주기 확인 (7일, 30일, 365일)
  - ADF Test + KPSS Test → 정상성 검정
  - 차분 필요 여부 결정
- [x] 베이스라인 모델
  - Naive (전주 동일 요일 값)
  - Moving Average (7일, 30일)
  - 베이스라인 MAPE 기록

### Day 3: SARIMA 모델
- [x] 03_sarima.ipynb
  - ACF/PACF 분석 → p, d, q, P, D, Q, m 결정
  - SARIMAX (외생변수 포함/미포함 비교)
  - AIC 기준 후보 차수 비교
  - 잔차 분석 (Ljung-Box, 정규성, 시각화)
  - 예측 + 95% 신뢰구간
  - Validation MAPE 기록

### Day 4: Prophet 모델
- [x] 04_prophet.ipynb
  - 기본 피팅 + 체인지포인트 시각화
  - 공휴일/프로모션/유가 regressor 추가
  - changepoint_prior_scale / seasonality_prior_scale 튜닝
  - Cross-Validation (Prophet 내장)
  - Validation MAPE 기록

### Day 5: XGBoost + 모델 비교
- [x] 05_xgboost.ipynb
  - Feature Engineering: lag(7,14,28,365), rolling(7일,30일), 날짜, 외생변수
  - TimeSeriesSplit 교차검증
  - Optuna 하이퍼파라미터 튜닝
  - SHAP Feature Importance
- [x] 06_model_comparison.ipynb
  - 3모델 비교표 (MAPE, RMSE, MAE, 학습시간)
  - 상품군별 최적 모델 분석
  - 앙상블 시도 (가중 평균)
  - Test set 최종 평가

### Day 6: Streamlit 앱
- [x] app/ 구현
  - 사이드바: 상품군 선택, 예측 기간, 모델 선택
  - Tab 1: EDA 대시보드
  - Tab 2: 모델 비교
  - Tab 3: 예측 결과 + 신뢰구간
  - Tab 4: Feature Importance (SHAP)
  - Tab 5: 재고 최적화 시뮬레이션
- [x] Streamlit Cloud 배포

### Day 7: GitHub + Notion 정리
- [x] README.md 완성
- [x] 핵심 시각화 스크린샷 3~4장
- [x] Notion 포트폴리오 업로드
  - Problem: "수동 발주로 인한 재고 과잉/부족 → 기회비용 발생"
  - Solution: "Prophet/SARIMA/XGBoost 3모델 비교 기반 수요 예측 시스템"
  - Impact: "MAPE 8.19% 달성 (목표 12% 대비 -3.81%p), 베이스라인 대비 75~95% 오차 감소"
  - Learning: "시계열 모델별 특성 이해, 외생변수 효과 검증, Cross-validation 설계"

## 개발 도구 워크플로우
- **포맷터**: `ruff format src/ app/` (Write/Edit 후 자동 실행)
- **린터**: `ruff check src/ app/ --fix`
- **타입체크**: `mypy src/ --ignore-missing-imports`
- **노트북 실행**: `jupyter nbconvert --to notebook --execute notebooks/{name}.ipynb`
- **Streamlit**: `streamlit run app/app.py`
- **파이프라인**: `python -c "from src.data_loader import run_pipeline; run_pipeline()"`

## 한글 폰트 설정 (matplotlib, Agg backend)
```python
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
sns.set_style('whitegrid')  # seaborn 먼저
plt.rcParams['font.family'] = 'Malgun Gothic'  # 폰트는 반드시 seaborn 이후
plt.rcParams['axes.unicode_minus'] = False
```

## 코딩 컨벤션
- Python 3.10+
- 타입 힌트 사용
- docstring: Google style
- 시각화: matplotlib + seaborn (기본), plotly (Streamlit용)
- 한글 폰트: `plt.rcParams['font.family'] = 'Malgun Gothic'` (Windows)
- 노트북: 마크다운 셀로 분석 스토리텔링 필수
- 모든 시각화에 타이틀, 축 레이블, 범례 포함

## 평가 지표 기준
- **Primary**: MAPE (Mean Absolute Percentage Error) — 12% 이하 목표
- **Secondary**: RMSE, MAE
- **베이스라인 대비 개선율** 반드시 기록
- 학습/추론 시간도 기록 (실무 관점)

## 주의사항
- ⚠️ 시계열 데이터는 반드시 시간순 분할. 랜덤 분할 = 미래 정보 유출
- ⚠️ XGBoost lag feature 생성 시 미래 데이터 유출 주의
- ⚠️ Prophet은 ds(날짜), y(타겟) 컬럼명 필수
- ⚠️ SARIMA는 학습 시간이 길 수 있음 → Top 5 상품군으로 제한하는 이유
- ⚠️ 유가 데이터에 결측치 많음 → 보간 필요 (linear interpolation)

## Notion 포트폴리오 정보
- data_source_id: ce6722a9-00b2-4d0e-8eda-190f4ce97cb6
- 글로우색상: teal (데이터분석)
- Extra-Label: MODEL COMPARISON
