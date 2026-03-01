import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="재고 시뮬레이션 — ForecastLab", page_icon="📦", layout="wide"
)

BASE = Path(__file__).resolve().parent.parent.parent


@st.cache_data
def load_sim_data():
    test = pd.read_csv(BASE / "data/processed/test.csv", parse_dates=["date"])
    pred = pd.read_csv(
        BASE / "outputs/results/xgboost_test_predictions.csv", parse_dates=["date"]
    )
    families = pd.read_csv(BASE / "data/processed/top_families.csv")["family"].tolist()
    return test, pred, families


test, pred, families = load_sim_data()

# ── sidebar ──
with st.sidebar:
    st.header("📦 시뮬레이션 설정")
    sim_family = st.selectbox("상품군", families, key="sim_family")
    safety_factor = st.slider("안전재고 계수", 0.0, 3.0, 1.5, 0.1, key="sim_safety")
    lead_time = st.slider("리드타임 (일)", 1, 14, 3, key="sim_lead")
    unit_cost = st.number_input("단위당 원가", value=1000, step=100, key="sim_cost")
    holding_pct = st.slider("재고 보관비 (%)", 1, 30, 10, key="sim_hold")
    stockout_multiplier = st.slider(
        "품절 손실 배수", 1.0, 5.0, 2.0, 0.5, key="sim_stockout"
    )

st.title("📦 재고 최적화 시뮬레이션")

# 데이터 준비
actual_df = test[test["family"] == sim_family].sort_values("date")
pred_df = pred[pred["family"] == sim_family].sort_values("date")

common_dates = sorted(set(actual_df["date"]).intersection(set(pred_df["date"])))
actual_df = (
    actual_df[actual_df["date"].isin(common_dates)]
    .sort_values("date")
    .reset_index(drop=True)
)
pred_df = (
    pred_df[pred_df["date"].isin(common_dates)]
    .sort_values("date")
    .reset_index(drop=True)
)

actual_sales = actual_df["sales"].values
predicted_sales = pred_df["xgboost_pred"].values
dates = actual_df["date"].values

# ── 시뮬레이션 로직 ──
st.subheader("1. 발주량 계산 로직")
st.markdown(
    f"""
    - **예측 기반 발주량** = 예측 매출 × 리드타임({lead_time}일)
    - **안전재고** = 예측 표준편차 × 안전재고 계수({safety_factor})
    - **적정 발주량** = 예측 기반 발주량 + 안전재고
    """
)

# 안전재고 = rolling std of predictions * safety_factor
pred_std = (
    pd.Series(predicted_sales).rolling(lead_time, min_periods=1).std().fillna(0).values
)
safety_stock = pred_std * safety_factor * np.sqrt(lead_time)
order_qty = predicted_sales * lead_time + safety_stock

# 재고 시뮬레이션
n = len(dates)
inventory = np.zeros(n)
overstock = np.zeros(n)
stockout = np.zeros(n)

# 초기 재고 = 첫날 발주량
inventory[0] = order_qty[0]

for i in range(n):
    if i > 0:
        # 리드타임 이전에 발주한 것이 도착
        if i >= lead_time:
            inventory[i] = (
                inventory[i - 1] - actual_sales[i - 1] + order_qty[i - lead_time]
            )
        else:
            inventory[i] = (
                inventory[i - 1] - actual_sales[i - 1] + order_qty[0] / lead_time
            )

    # 과잉/부족 판정
    if inventory[i] > actual_sales[i] * 2:
        overstock[i] = inventory[i] - actual_sales[i] * 2
    elif inventory[i] < 0:
        stockout[i] = abs(inventory[i])
        inventory[i] = 0

# ── 2. 재고 수준 시각화 ──
st.subheader("2. 일별 재고 수준")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=dates, y=actual_sales, name="실제 매출", line=dict(color="#0d9488", width=2)
    )
)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=predicted_sales,
        name="예측 매출",
        line=dict(color="#f59e0b", width=2, dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=inventory,
        name="재고 수준",
        fill="tozeroy",
        line=dict(color="#6366f1"),
        opacity=0.3,
    )
)
fig.update_layout(
    height=450,
    yaxis_title="수량",
    xaxis_title="",
    legend=dict(orientation="h", y=-0.12),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── 3. 비용 분석 ──
st.subheader("3. 비용 분석")

holding_cost = overstock.sum() * unit_cost * (holding_pct / 100)
stockout_cost = stockout.sum() * unit_cost * stockout_multiplier
total_cost = holding_cost + stockout_cost

col1, col2, col3, col4 = st.columns(4)
col1.metric("과잉재고 일수", f"{(overstock > 0).sum()}일")
col2.metric("품절 일수", f"{(stockout > 0).sum()}일")
col3.metric("보관비용", f"₩{holding_cost:,.0f}")
col4.metric("품절손실", f"₩{stockout_cost:,.0f}")

st.metric("총 비용", f"₩{total_cost:,.0f}")

# 비용 비교: 예측 기반 vs 단순 평균 발주
st.divider()
st.subheader("4. 예측 기반 vs 단순 평균 발주 비교")

# 단순 평균 발주 시뮬레이션
avg_order = np.mean(actual_sales) * lead_time
inv_naive = np.zeros(n)
over_naive = np.zeros(n)
out_naive = np.zeros(n)
inv_naive[0] = avg_order

for i in range(n):
    if i > 0:
        inv_naive[i] = inv_naive[i - 1] - actual_sales[i - 1] + avg_order / lead_time
    if inv_naive[i] > actual_sales[i] * 2:
        over_naive[i] = inv_naive[i] - actual_sales[i] * 2
    elif inv_naive[i] < 0:
        out_naive[i] = abs(inv_naive[i])
        inv_naive[i] = 0

naive_holding = over_naive.sum() * unit_cost * (holding_pct / 100)
naive_stockout = out_naive.sum() * unit_cost * stockout_multiplier
naive_total = naive_holding + naive_stockout

comparison = pd.DataFrame(
    {
        "전략": ["예측 기반 발주", "단순 평균 발주"],
        "과잉재고 일수": [(overstock > 0).sum(), (over_naive > 0).sum()],
        "품절 일수": [(stockout > 0).sum(), (out_naive > 0).sum()],
        "보관비용": [holding_cost, naive_holding],
        "품절손실": [stockout_cost, naive_stockout],
        "총 비용": [total_cost, naive_total],
    }
)

st.dataframe(
    comparison.style.format(
        {"보관비용": "₩{:,.0f}", "품절손실": "₩{:,.0f}", "총 비용": "₩{:,.0f}"}
    ),
    hide_index=True,
    use_container_width=True,
)

if naive_total > 0:
    saving_pct = (naive_total - total_cost) / naive_total * 100
    if saving_pct > 0:
        st.success(f"📉 예측 기반 발주로 총 비용 **{saving_pct:.1f}% 절감** 가능")
    else:
        st.warning(
            f"현재 설정에서는 단순 평균 발주 대비 {abs(saving_pct):.1f}% 비용 증가. 파라미터를 조정해보세요."
        )
