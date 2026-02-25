---
name: forecast-lab-architecture
description: 프로젝트 구조, 모듈 의존성, 코딩 컨벤션 가이드. Use when creating new modules, files, or refactoring structure.
---

# Forecast Lab Architecture

## 프로젝트 구조

```
forecast-lab/
├── src/                        # 핵심 로직 (순수 Python, 프레임워크 비의존)
│   ├── data_loader.py          # 데이터 로드 + 전처리 파이프라인
│   ├── feature_engineering.py  # lag, rolling, 날짜 피처 생성
│   ├── evaluation.py           # MAPE, RMSE, MAE 평가 함수
│   ├── utils.py                # 유틸리티 (경로, 로깅 등)
│   └── models/                 # 모델별 분리
│       ├── sarima_model.py     # SARIMAX 래퍼
│       ├── prophet_model.py    # Prophet 래퍼
│       └── xgboost_model.py    # XGBoost 래퍼
├── notebooks/                  # 분석 노트북 (실험 + 스토리텔링)
├── app/                        # Streamlit UI (src/ 임포트만)
├── data/raw/                   # Kaggle 원본 (git 제외)
├── data/processed/             # 전처리 결과 (git 제외)
├── models/                     # 학습된 모델 저장 (git 제외)
└── outputs/                    # 시각화 + 결과 CSV
```

## 모듈 의존성 방향

```
notebooks/ ──→ src/ ←── app/
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
  data_loader  models/  evaluation
        │       │
        ▼       ▼
  feature_engineering
```

- **src/**: 핵심 로직. 외부 의존 없이 독립 실행 가능해야 함
- **notebooks/**: src/ 함수를 호출하여 분석. 실험 코드는 노트북에만 작성
- **app/**: src/ 함수를 호출하여 UI 구성. 비즈니스 로직 금지
- 순환 의존 절대 금지

## 코딩 컨벤션

- Python 3.10+, 타입 힌트 필수
- docstring: Google style
- `from __future__ import annotations` 모든 모듈 상단에 포함
- 경로: `pathlib.Path` 사용 (os.path 금지)
- 상수: 모듈 상단에 대문자 스네이크케이스 (`PROJECT_ROOT`, `TRAIN_END`)
- import 순서: stdlib → third-party → local

## 새 파일 생성 규칙

1. `src/`에 새 모듈 추가 시 → `__init__.py`에서 export 여부 결정
2. `src/models/`에 새 모델 추가 시 → 동일한 인터페이스 패턴 따르기:
   - `fit(train_df, val_df) -> model`
   - `predict(model, periods) -> predictions_df`
   - `evaluate(predictions, actuals) -> metrics_dict`
3. 노트북 추가 시 → `XX_이름.ipynb` 번호 패턴 유지
4. app 페이지 추가 시 → `app/pages/N_이름.py` 패턴 유지
