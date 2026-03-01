import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="피처 중요도 — ForecastLab", page_icon="🔬", layout="wide"
)

BASE = Path(__file__).resolve().parent.parent.parent
FIGURES = BASE / "outputs" / "figures"

st.title("🔬 피처 중요도 (SHAP)")

st.markdown(
    """
    XGBoost 모델의 예측에 각 피처가 어떤 영향을 미치는지
    SHAP(SHapley Additive exPlanations)으로 분석한 결과입니다.
    """
)

# ── 1. SHAP Bar Plot ──
st.subheader("1. 피처 중요도 순위")

shap_bar = FIGURES / "20_shap_bar.png"
if shap_bar.exists():
    st.image(str(shap_bar), use_container_width=True)
else:
    st.warning("SHAP Bar 이미지를 찾을 수 없습니다.")

st.markdown(
    """
    **Top 5 피처 해석:**

    | 순위 | 피처 | 의미 |
    |------|------|------|
    | 1 | `lag_7` | 7일 전 매출 — 가장 강력한 예측 변수 |
    | 2 | `ewm_mean_7` | 7일 지수가중이동평균 — 최근 추세에 가중치 |
    | 3 | `rolling_mean_7` | 7일 이동평균 — 최근 추세 반영 |
    | 4 | `total_transactions` | 거래 건수 — 매장 트래픽 반영 |
    | 5 | `lag_14` | 14일 전 매출 — 2주 주기 패턴 포착 |
    """
)

st.info(
    "💡 **핵심 인사이트**: lag/rolling 피처가 상위권을 독점합니다. "
    "실무에서 최근 매출 추세 모니터링이 프로모션 기획보다 예측 정확도에 더 중요합니다."
)

# ── 2. SHAP Summary Plot ──
st.subheader("2. SHAP Summary Plot")

shap_summary = FIGURES / "19_shap_summary.png"
if shap_summary.exists():
    st.image(str(shap_summary), use_container_width=True)
else:
    st.warning("SHAP Summary 이미지를 찾을 수 없습니다.")

st.markdown(
    """
    - 🔴 빨간색 = 높은 피처 값 / 🔵 파란색 = 낮은 피처 값
    - X축: SHAP value (양수 = 매출 증가 방향, 음수 = 매출 감소 방향)
    - `lag_7`이 높을수록(빨간색) SHAP value가 양수 → 매출 증가 예측
    """
)

st.divider()

# ── 3. Optuna 튜닝 히스토리 ──
st.subheader("3. Optuna 하이퍼파라미터 튜닝")

optuna_fig = FIGURES / "18_optuna_history.png"
if optuna_fig.exists():
    st.image(str(optuna_fig), use_container_width=True)
else:
    st.warning("Optuna 히스토리 이미지를 찾을 수 없습니다.")

st.markdown(
    """
    - 50회 Trial로 하이퍼파라미터 탐색
    - Baseline MAPE 51.1% → Tuned 48.5% (약 2.6%p 개선)
    - 주요 튜닝 파라미터: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
    """
)
