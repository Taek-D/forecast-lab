전체 프로젝트의 코드 품질을 검사하고 자동 수정합니다.

1. `ruff check src/ app/ --fix` 실행 — 자동 수정 가능한 린트 이슈 처리
2. `ruff format src/ app/` 실행 — 코드 포맷팅
3. `mypy src/ --ignore-missing-imports` 실행 — 타입 체크
4. 결과 요약 출력 (수정된 파일 수, 남은 이슈)
