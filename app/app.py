import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="ForecastLab — 수요 예측 시스템",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent.parent

# ── sidebar ──
with st.sidebar:
    st.title("📈 ForecastLab")
    st.caption("E-commerce 수요 예측 시스템")
    st.divider()
    st.markdown(
        """
        **Prophet · SARIMA · XGBoost**
        3개 모델 비교로 주간 매출 예측

        - 🗂️ Kaggle Favorita 매출 데이터
        - 📅 2013 ~ 2017 (에콰도르)
        - 🎯 Top 5 상품군 집중 분석
        """
    )
    st.divider()
    st.markdown("**페이지 안내**")
    st.markdown(
        """
        1. **EDA** — 데이터 탐색
        2. **모델 비교** — 4모델 성능
        3. **예측 결과** — 실제 vs 예측
        4. **피처 중요도** — SHAP 분석
        5. **재고 시뮬레이션** — 발주 최적화
        """
    )

# ── main ──
st.title("🏠 프로젝트 개요")
st.markdown(
    """
    > **문제**: 수동 발주로 인한 재고 과잉/부족 → 기회비용 발생
    > **해결**: Prophet / SARIMA / XGBoost 3모델 비교 기반 수요 예측 시스템 구축
    """
)

# KPI cards
test_results = pd.read_csv(BASE / "outputs/results/test_final_results.csv")
avg_mape = test_results["mape"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("평균 Test MAPE", f"{avg_mape:.2f}%", delta="-3.81%p vs 목표 12%")
col2.metric("최적 모델", "XGBoost")
col3.metric("분석 상품군", "Top 5")
col4.metric("데이터 기간", "2013~2017")

st.divider()

# 상품군별 결과
st.subheader("📊 상품군별 Test MAPE (XGBoost)")

col_left, col_right = st.columns([2, 1])

with col_left:
    import plotly.express as px

    fig = px.bar(
        test_results.sort_values("mape"),
        x="family",
        y="mape",
        color="mape",
        color_continuous_scale="Tealgrn_r",
        text=test_results.sort_values("mape")["mape"].apply(lambda x: f"{x:.1f}%"),
    )
    fig.add_hline(y=12, line_dash="dash", line_color="red", annotation_text="목표 12%")
    fig.update_layout(
        xaxis_title="",
        yaxis_title="MAPE (%)",
        coloraxis_showscale=False,
        height=400,
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.dataframe(
        test_results[["family", "mape", "rmse", "mae"]].style.format(
            {"mape": "{:.2f}%", "rmse": "{:,.0f}", "mae": "{:,.0f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# 프로젝트 타임라인
st.subheader("🗓️ 프로젝트 타임라인")
timeline_data = {
    "Day": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6"],
    "내용": [
        "데이터 탐색 + 전처리",
        "시계열 분해 + 베이스라인",
        "SARIMA 모델",
        "Prophet 모델",
        "XGBoost + 모델 비교",
        "Streamlit 대시보드",
    ],
    "상태": ["✅", "✅", "✅", "✅", "✅", "🚀"],
}
st.dataframe(pd.DataFrame(timeline_data), hide_index=True, use_container_width=True)
