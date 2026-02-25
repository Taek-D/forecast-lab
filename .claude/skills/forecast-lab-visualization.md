---
name: forecast-lab-visualization
description: 시각화 패턴, 한글 폰트 설정, Streamlit 차트 가이드. Use when creating plots, charts, or Streamlit dashboard components.
---

# Visualization Skill

## 한글 폰트 설정 (필수 순서)

```python
import matplotlib
matplotlib.use('Agg')  # 노트북이 아닌 스크립트에서
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 1. 폰트 매니저 재구성
fm._load_fontmanager(try_read_cache=False)

# 2. seaborn 스타일 먼저
sns.set_style('whitegrid')

# 3. 폰트는 반드시 seaborn 이후
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
```

- 이 순서를 바꾸면 한글이 깨짐
- 노트북에서는 `%matplotlib inline` 추가

## matplotlib + seaborn 패턴 (노트북/저장용)

### 기본 시각화 템플릿

```python
fig, ax = plt.subplots(figsize=(14, 6))
# ... 플롯 코드 ...
ax.set_title('제목', fontsize=14, fontweight='bold')
ax.set_xlabel('X축 레이블')
ax.set_ylabel('Y축 레이블')
ax.legend(loc='best')
plt.tight_layout()
plt.savefig('outputs/figures/이름.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 필수 규칙
- 모든 시각화에 타이틀, 축 레이블, 범례 포함
- 저장: `outputs/figures/` 디렉토리, PNG 150 dpi
- `figsize`: 단일 차트 (14, 6), 서브플롯 (16, 10) 이상
- `tight_layout()` 또는 `bbox_inches='tight'` 필수

### 시계열 시각화 패턴

```python
# 여러 상품군 비교
fig, axes = plt.subplots(len(families), 1, figsize=(16, 4*len(families)), sharex=True)
for i, family in enumerate(families):
    subset = df[df['family'] == family]
    axes[i].plot(subset['date'], subset['sales'])
    axes[i].set_title(family)
    axes[i].axvline(pd.Timestamp('2017-01-01'), color='red', linestyle='--', label='Val 시작')
```

### 모델 비교 시각화 패턴

```python
# 예측 vs 실제 + 신뢰구간
ax.plot(dates, actuals, label='실제', color='black', linewidth=1.5)
ax.plot(dates, predictions, label='예측', color='blue', linewidth=1.5)
ax.fill_between(dates, lower_ci, upper_ci, alpha=0.2, color='blue', label='95% 신뢰구간')
```

## plotly 패턴 (Streamlit용)

```python
import plotly.express as px
import plotly.graph_objects as go

# Streamlit에서는 plotly 사용
fig = px.line(df, x='date', y='sales', color='family',
              title='상품군별 매출 추이')
fig.update_layout(
    xaxis_title='날짜',
    yaxis_title='매출',
    template='plotly_white',
    height=500
)
st.plotly_chart(fig, use_container_width=True)
```

## Streamlit 대시보드 패턴

```python
import streamlit as st

# 사이드바
st.sidebar.selectbox('상품군', families)
st.sidebar.slider('예측 기간 (일)', 7, 90, 30)
st.sidebar.radio('모델', ['SARIMA', 'Prophet', 'XGBoost'])

# 메트릭 카드
col1, col2, col3 = st.columns(3)
col1.metric('MAPE', f'{mape:.1f}%', delta=f'{improvement:.1f}%')
col2.metric('RMSE', f'{rmse:,.0f}')
col3.metric('MAE', f'{mae:,.0f}')

# 탭 구조
tab1, tab2 = st.tabs(['EDA', '모델 비교'])
with tab1:
    st.plotly_chart(fig1, use_container_width=True)
```

## 색상 팔레트

```python
# 상품군별 고정 색상 (일관성)
FAMILY_COLORS = {
    'GROCERY I': '#1f77b4',
    'BEVERAGES': '#ff7f0e',
    'PRODUCE': '#2ca02c',
    'CLEANING': '#d62728',
    'DAIRY': '#9467bd',
}

# 모델별 고정 색상
MODEL_COLORS = {
    'SARIMA': '#e74c3c',
    'Prophet': '#3498db',
    'XGBoost': '#2ecc71',
    'Ensemble': '#f39c12',
    'Baseline': '#95a5a6',
}
```
