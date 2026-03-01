"""
ForecastLab 프로젝트 초기 세팅 스크립트
실행: python setup.py
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 1. 폴더 구조 생성
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/models",
    "app/pages",
    "app/.streamlit",
    "models/sarima",
    "models/prophet",
    "models/xgboost",
    "outputs/figures",
    "outputs/results",
    "docs/screenshots",
]

print("📁 폴더 구조 생성 중...")
for folder in folders:
    path = os.path.join(PROJECT_ROOT, folder)
    os.makedirs(path, exist_ok=True)
    print(f"  ✅ {folder}/")

# __init__.py 생성
for init_path in ["src/__init__.py", "src/models/__init__.py"]:
    path = os.path.join(PROJECT_ROOT, init_path)
    if not os.path.exists(path):
        with open(path, "w") as f:
            pass
        print(f"  ✅ {init_path}")

# Streamlit config
config_path = os.path.join(PROJECT_ROOT, "app/.streamlit/config.toml")
if not os.path.exists(config_path):
    with open(config_path, "w") as f:
        f.write("""[theme]
primaryColor = "#0d9488"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#e2e8f0"
font = "sans serif"

[server]
headless = true
""")
    print("  ✅ app/.streamlit/config.toml")

# 2. Kaggle 데이터 다운로드
print("\n📦 Kaggle 데이터 다운로드 중...")
print("  ⚠️  kaggle API 토큰이 필요합니다 (~/.kaggle/kaggle.json)")
print(
    "  ⚠️  없으면 수동 다운로드: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data\n"
)

data_dir = os.path.join(PROJECT_ROOT, "data/raw")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "kaggle", "-q"],
        check=True,
    )
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "store-sales-time-series-forecasting",
            "-p",
            data_dir,
        ],
        check=True,
    )
    # zip 해제
    import zipfile

    zip_path = os.path.join(data_dir, "store-sales-time-series-forecasting.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        os.remove(zip_path)
        print("  ✅ 데이터 다운로드 + 압축 해제 완료!")
except Exception as e:
    print(f"  ❌ Kaggle 다운로드 실패: {e}")
    print("  → 수동으로 다운로드 후 data/raw/ 에 넣어주세요")

# 3. 가상환경 안내
print("\n🐍 가상환경 생성 (권장):")
print(f'  cd "{PROJECT_ROOT}"')
print("  python -m venv .venv")
print("  .venv\\Scripts\\activate    # Windows")
print("  pip install -r requirements.txt")

print("\n✨ 세팅 완료! Claude Code로 Day 1을 시작하세요:")
print(f'  cd "{PROJECT_ROOT}"')
print("  claude")
print('  > "Day 1 시작: EDA 노트북 작성해줘"')
