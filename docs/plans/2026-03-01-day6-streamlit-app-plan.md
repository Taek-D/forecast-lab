# Day 6: Streamlit 앱 구현 계획

## 개요
- 5개 페이지로 구성된 Streamlit 멀티페이지 앱
- CSV 결과 + PNG 시각화를 로드하여 인터랙티브 대시보드 구축
- plotly로 인터랙티브 차트, 기존 PNG는 보조 사용

## 사용 가능한 데이터
- `data/processed/full_data.csv` — 전체 데이터 (5 families, 2013~2017)
- `data/processed/train.csv`, `val.csv`, `test.csv` — 분할 데이터
- `data/processed/top_families.csv` — Top 5 상품군 리스트
- `outputs/results/model_comparison.csv` — 4모델 validation 결과
- `outputs/results/test_final_results.csv` — Test 최종 결과
- `outputs/results/ensemble_results.csv` — 앙상블 결과
- `outputs/results/xgboost_val_predictions.csv` — XGBoost validation 예측
- `outputs/results/xgboost_test_predictions.csv` — XGBoost test 예측
- `outputs/figures/*.png` — 18개 시각화

## 태스크

### Task 1: app/app.py — 메인 엔트리 + 사이드바
- st.set_page_config(page_title, page_icon, layout="wide")
- 사이드바: 프로젝트 소개, 네비게이션 안내
- 메인: 프로젝트 개요, 핵심 결과 KPI 카드 (Test MAPE 8.19%)
- 검증: `streamlit run app/app.py` 실행 확인

### Task 2: app/pages/1_eda.py — EDA 대시보드
- full_data.csv 로드
- 사이드바: 상품군 선택 (multiselect), 날짜 범위 슬라이더
- plotly 차트: 매출 시계열, 요일별 패턴, 월별 패턴
- 프로모션 효과 시각화
- 외생변수(유가, 거래건수) 상관관계

### Task 3: app/pages/2_model_comparison.py — 모델 비교
- model_comparison.csv + test_final_results.csv 로드
- 사이드바: 모델 선택 (multiselect)
- MAPE 히트맵 (plotly), 바 차트
- Val vs Test 비교 테이블
- 앙상블 결과 표시

### Task 4: app/pages/3_forecast.py — 예측 결과 시각화
- xgboost_val/test_predictions.csv + 실제 데이터 로드
- 사이드바: 상품군 선택, 기간 선택 (Val/Test)
- plotly 라인 차트: 실제 vs 예측
- 에러 분석 (잔차 분포, 시간별 에러)

### Task 5: app/pages/4_feature_importance.py — SHAP 분석
- SHAP PNG 이미지 표시 (19_shap_summary.png, 20_shap_bar.png)
- 피처 중요도 해석 텍스트
- Optuna 튜닝 히스토리 (18_optuna_history.png)

### Task 6: app/pages/5_inventory_simulation.py — 재고 시뮬레이션
- 사이드바: 안전재고 계수(slider), 리드타임(slider), 단가(input)
- 예측 기반 적정 발주량 계산
- 과잉재고/부족재고 시뮬레이션
- plotly 차트: 재고 수준 시각화, 비용 분석
