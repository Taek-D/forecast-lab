import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="모델 비교 — ForecastLab", page_icon="⚖️", layout="wide")

BASE = Path(__file__).resolve().parent.parent.parent


@st.cache_data
def load_results():
    comp = pd.read_csv(BASE / "outputs/results/model_comparison.csv")
    test = pd.read_csv(BASE / "outputs/results/test_final_results.csv")
    ensemble = pd.read_csv(BASE / "outputs/results/ensemble_results.csv")
    return comp, test, ensemble


comp, test, ensemble = load_results()

all_models = comp["model"].unique().tolist()

# ── sidebar ──
with st.sidebar:
    st.header("⚖️ 모델 비교 설정")
    selected_models = st.multiselect(
        "모델 선택", all_models, default=all_models, key="comp_models"
    )
    metric = st.selectbox("평가 지표", ["mape", "rmse", "mae"], key="comp_metric")
    metric_label = {"mape": "MAPE (%)", "rmse": "RMSE", "mae": "MAE"}[metric]

filtered = comp[comp["model"].isin(selected_models)]

st.title("⚖️ 모델 비교")

# ── 1. MAPE 히트맵 ──
st.subheader("1. Validation 성능 히트맵")

pivot = filtered.pivot(index="family", columns="model", values=metric)
model_order = [
    m
    for m in ["Naive_7d", "SARIMAX", "Prophet_Tuned", "XGBoost_Tuned"]
    if m in pivot.columns
]
pivot = pivot[model_order]

fig_heat = go.Figure(
    data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        text=pivot.values.round(1).astype(str),
        texttemplate="%{text}",
        colorscale="RdYlGn_r" if metric == "mape" else "Blues",
        hovertemplate="모델: %{x}<br>상품군: %{y}<br>값: %{z:.1f}<extra></extra>",
    )
)
fig_heat.update_layout(height=400, xaxis_title="", yaxis_title="")
st.plotly_chart(fig_heat, use_container_width=True)

# ── 2. 바 차트 ──
st.subheader(f"2. 상품군별 {metric_label} 비교")

fig_bar = px.bar(
    filtered,
    x="family",
    y=metric,
    color="model",
    barmode="group",
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={"family": "", metric: metric_label, "model": "모델"},
)
if metric == "mape":
    fig_bar.add_hline(
        y=12, line_dash="dash", line_color="red", annotation_text="목표 12%"
    )
fig_bar.update_layout(height=420, legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig_bar, use_container_width=True)

# ── 3. Validation vs Test 비교 ──
st.subheader("3. Validation vs Test 비교 (XGBoost)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Validation MAPE**")
    val_xgb = comp[comp["model"] == "XGBoost_Tuned"][["family", "mape", "rmse", "mae"]]
    st.dataframe(
        val_xgb.style.format({"mape": "{:.2f}%", "rmse": "{:,.0f}", "mae": "{:,.0f}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.metric("평균 Val MAPE", f"{val_xgb['mape'].mean():.2f}%")

with col2:
    st.markdown("**Test MAPE**")
    st.dataframe(
        test[["family", "mape", "rmse", "mae"]].style.format(
            {"mape": "{:.2f}%", "rmse": "{:,.0f}", "mae": "{:,.0f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.metric("평균 Test MAPE", f"{test['mape'].mean():.2f}%")

# ── 4. 앙상블 결과 ──
st.subheader("4. 앙상블 vs 단일 최적 모델")

fig_ens = go.Figure()
fig_ens.add_trace(
    go.Bar(
        name="Best Single Model",
        x=ensemble["family"],
        y=ensemble["best_single_mape"],
        text=ensemble["best_single_mape"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    )
)
fig_ens.add_trace(
    go.Bar(
        name="Weighted Avg Ensemble",
        x=ensemble["family"],
        y=ensemble["weighted_avg_mape"],
        text=ensemble["weighted_avg_mape"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    )
)
fig_ens.update_layout(barmode="group", height=400, yaxis_title="MAPE (%)")
st.plotly_chart(fig_ens, use_container_width=True)

st.info(
    "💡 XGBoost가 모든 상품군에서 압도적이므로 앙상블은 오히려 성능 하락. "
    "앙상블은 모델 간 성능 차이가 작고 예측 상관관계가 낮을 때 효과적."
)

# ── 5. 학습/추론 시간 ──
with st.expander("⏱️ 학습 및 추론 시간"):
    time_cols = ["model", "family", "train_time_sec", "predict_time_sec"]
    if all(c in comp.columns for c in time_cols):
        time_df = comp[time_cols].copy()
        time_df["total_sec"] = time_df["train_time_sec"] + time_df["predict_time_sec"]
        avg_time = time_df.groupby("model")["total_sec"].mean().reset_index()
        avg_time.columns = ["모델", "평균 소요시간 (초)"]
        st.dataframe(
            avg_time.style.format({"평균 소요시간 (초)": "{:.2f}"}),
            hide_index=True,
            use_container_width=True,
        )
