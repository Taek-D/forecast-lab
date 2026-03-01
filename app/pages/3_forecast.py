import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path

st.set_page_config(page_title="예측 결과 — ForecastLab", page_icon="📈", layout="wide")

BASE = Path(__file__).resolve().parent.parent.parent


@st.cache_data
def load_forecast_data():
    val = pd.read_csv(BASE / "data/processed/val.csv", parse_dates=["date"])
    test = pd.read_csv(BASE / "data/processed/test.csv", parse_dates=["date"])
    val_pred = pd.read_csv(
        BASE / "outputs/results/xgboost_val_predictions.csv", parse_dates=["date"]
    )
    test_pred = pd.read_csv(
        BASE / "outputs/results/xgboost_test_predictions.csv", parse_dates=["date"]
    )
    families = pd.read_csv(BASE / "data/processed/top_families.csv")["family"].tolist()
    return val, test, val_pred, test_pred, families


val, test, val_pred, test_pred, families = load_forecast_data()

# ── sidebar ──
with st.sidebar:
    st.header("📈 예측 설정")
    selected_family = st.selectbox("상품군 선택", families, key="fc_family")
    period = st.radio(
        "기간 선택", ["Validation (2017 상반기)", "Test (2017 하반기)"], key="fc_period"
    )

st.title("📈 예측 결과")

# 기간에 따라 데이터 선택
if period.startswith("Val"):
    actual = val[val["family"] == selected_family].sort_values("date")
    pred = val_pred[val_pred["family"] == selected_family].sort_values("date")
    pred_col = "xgboost_pred"
    period_label = "Validation (2017-01 ~ 2017-06)"
else:
    actual = test[test["family"] == selected_family].sort_values("date")
    pred = test_pred[test_pred["family"] == selected_family].sort_values("date")
    pred_col = "xgboost_pred"
    period_label = "Test (2017-07 ~ 2017-08)"

# 공통 날짜 매칭
common_dates = set(actual["date"]).intersection(set(pred["date"]))
actual_m = actual[actual["date"].isin(common_dates)].sort_values("date")
pred_m = pred[pred["date"].isin(common_dates)].sort_values("date")

# ── 1. 실제 vs 예측 라인 차트 ──
st.subheader(f"1. {selected_family} — 실제 vs 예측 ({period_label})")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=actual_m["date"],
        y=actual_m["sales"],
        name="실제",
        line=dict(color="#0d9488", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=pred_m["date"],
        y=pred_m[pred_col],
        name="XGBoost 예측",
        line=dict(color="#f59e0b", width=2, dash="dash"),
    )
)
fig.update_layout(
    height=450,
    xaxis_title="",
    yaxis_title="매출",
    legend=dict(orientation="h", y=-0.12),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── 2. 에러 분석 ──
st.subheader("2. 에러 분석")

merged = actual_m[["date", "sales"]].merge(pred_m[["date", pred_col]], on="date")
merged["error"] = merged["sales"] - merged[pred_col]
merged["abs_pct_error"] = np.abs(merged["error"]) / merged["sales"].clip(lower=1) * 100

col1, col2, col3 = st.columns(3)
col1.metric("평균 오차", f"{merged['error'].mean():,.0f}")
col2.metric("MAPE", f"{merged['abs_pct_error'].mean():.2f}%")
col3.metric("RMSE", f"{np.sqrt((merged['error'] ** 2).mean()):,.0f}")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**잔차 분포**")
    fig2 = px.histogram(
        merged,
        x="error",
        nbins=30,
        labels={"error": "잔차 (실제 - 예측)"},
        color_discrete_sequence=["#0d9488"],
    )
    fig2.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.markdown("**시간별 MAPE**")
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["abs_pct_error"],
            mode="lines+markers",
            marker=dict(size=4, color="#0d9488"),
            line=dict(color="#0d9488"),
        )
    )
    fig3.add_hline(y=12, line_dash="dash", line_color="red", annotation_text="목표 12%")
    fig3.update_layout(height=350, xaxis_title="", yaxis_title="APE (%)")
    st.plotly_chart(fig3, use_container_width=True)

# ── 3. 요일별 에러 패턴 ──
st.subheader("3. 요일별 에러 패턴")
merged["day_of_week"] = pd.to_datetime(merged["date"]).dt.day_name()
day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
day_kr = ["월", "화", "수", "목", "금", "토", "일"]

day_error = merged.groupby("day_of_week")["abs_pct_error"].mean().reindex(day_order)

fig4 = go.Figure(
    go.Bar(
        x=day_kr,
        y=day_error.values,
        marker_color="#0d9488",
        text=day_error.values.round(1),
        textposition="outside",
    )
)
fig4.update_layout(height=350, yaxis_title="평균 APE (%)", xaxis_title="요일")
fig4.add_hline(y=12, line_dash="dash", line_color="red")
st.plotly_chart(fig4, use_container_width=True)
