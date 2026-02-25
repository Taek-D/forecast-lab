지정된 노트북을 실행하고 결과를 확인합니다.

사용법: /run-notebook $ARGUMENTS

1. `jupyter nbconvert --to notebook --execute notebooks/$ARGUMENTS --output $ARGUMENTS` 실행
2. 실행 중 에러가 있으면 해당 셀과 에러 메시지 출력
3. 성공 시 실행 완료 메시지 출력
