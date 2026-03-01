import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="EDA — ForecastLab", page_icon="🔍", layout="wide")

BASE = Path(__file__).resolve().parent.parent.parent


@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "data/processed/full_data.csv", parse_dates=["date"])
    families = pd.read_csv(BASE / "data/processed/top_families.csv")["family"].tolist()
    return df, families


df, families = load_data()

# ── sidebar ──
with st.sidebar:
    st.header("🔍 EDA 설정")
    selected_families = st.multiselect(
        "상품군 선택", families, default=families, key="eda_families"
    )
    date_range = st.date_input(
        "날짜 범위",
        value=(df["date"].min(), df["date"].max()),
        min_value=df["date"].min(),
        max_value=df["date"].max(),
        key="eda_dates",
    )

# filter
mask = df["family"].isin(selected_families)
if len(date_range) == 2:
    mask &= (df["date"] >= pd.Timestamp(date_range[0])) & (
        df["date"] <= pd.Timestamp(date_range[1])
    )
filtered = df[mask]

st.title("🔍 데이터 탐색 (EDA)")

# ── 1. 매출 시계열 ──
st.subheader("1. 상품군별 매출 시계열")
fig = px.line(
    filtered,
    x="date",
    y="sales",
    color="family",
    labels={"date": "", "sales": "매출", "family": "상품군"},
)
fig.update_layout(height=450, legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, use_container_width=True)

# ── 2. 요일별 패턴 ──
st.subheader("2. 요일별 매출 패턴")
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

weekday_agg = filtered.groupby(["family", "day_name"])["sales"].mean().reset_index()
weekday_agg["day_name"] = pd.Categorical(weekday_agg["day_name"], categories=day_order)
weekday_agg = weekday_agg.sort_values("day_name")

fig2 = px.bar(
    weekday_agg,
    x="day_name",
    y="sales",
    color="family",
    barmode="group",
    labels={"day_name": "요일", "sales": "평균 매출", "family": "상품군"},
)
fig2.update_layout(height=400)
fig2.update_xaxes(ticktext=day_kr, tickvals=day_order)
st.plotly_chart(fig2, use_container_width=True)

# ── 3. 월별 패턴 ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("3. 월별 매출 추이")
    monthly = filtered.groupby(["family", "month"])["sales"].mean().reset_index()
    fig3 = px.line(
        monthly,
        x="month",
        y="sales",
        color="family",
        markers=True,
        labels={"month": "월", "sales": "평균 매출", "family": "상품군"},
    )
    fig3.update_layout(height=380)
    fig3.update_xaxes(tickmode="linear", dtick=1)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("4. 프로모션 효과")
    promo_agg = (
        filtered.groupby(["family", "onpromotion"])["sales"].mean().reset_index()
    )
    promo_agg["promo_label"] = promo_agg["onpromotion"].apply(
        lambda x: "프로모션 있음" if x > 0 else "프로모션 없음"
    )
    fig4 = px.bar(
        promo_agg,
        x="family",
        y="sales",
        color="promo_label",
        barmode="group",
        labels={"family": "", "sales": "평균 매출", "promo_label": ""},
    )
    fig4.update_layout(height=380)
    st.plotly_chart(fig4, use_container_width=True)

# ── 5. 외생변수 상관관계 ──
st.subheader("5. 외생변수 상관관계")
col3, col4 = st.columns(2)

with col3:
    fig5 = px.scatter(
        filtered.dropna(subset=["oil_price"]),
        x="oil_price",
        y="sales",
        color="family",
        opacity=0.3,
        labels={"oil_price": "유가 (USD)", "sales": "매출", "family": "상품군"},
        title="유가 vs 매출",
    )
    fig5.update_layout(height=380)
    st.plotly_chart(fig5, use_container_width=True)

with col4:
    fig6 = px.scatter(
        filtered.dropna(subset=["total_transactions"]),
        x="total_transactions",
        y="sales",
        color="family",
        opacity=0.3,
        labels={
            "total_transactions": "거래 건수",
            "sales": "매출",
            "family": "상품군",
        },
        title="거래 건수 vs 매출",
    )
    fig6.update_layout(height=380)
    st.plotly_chart(fig6, use_container_width=True)

# ── 데이터 미리보기 ──
with st.expander("📋 데이터 미리보기"):
    st.dataframe(filtered.head(100), use_container_width=True)
    st.caption(f"전체 {len(filtered):,}행 중 상위 100행 표시")
