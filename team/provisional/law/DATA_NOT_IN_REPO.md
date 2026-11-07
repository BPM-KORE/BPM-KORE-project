# 이 폴더의 데이터는 공개 저장소에서 제거했다

`knowledge.csv`와 `source_log.csv`는 출처 기록의 `license_checked`가 전부 `no`(재배포 가능 여부 미확인)라서 공개 저장소에서 뺐다.

- 데이터를 가진 사람은 파일을 로컬에 두고 환경변수 `KORE_KB`에 경로를 지정하면 실험 스크립트가 그대로 돈다.
  예: `KORE_KB=/내/경로/knowledge.csv python3 experiments/retrieval_diag/...`
- 이 경로의 파일은 다시 커밋되지 않도록 `.gitignore`에 올렸다.
- 라이선스 확인(사람이 공식 원문과 대조)이 끝나면 `source_log.csv`의 `license_checked`를 `yes`로 바꾸고 다시 공개할 수 있다.
- 주의: 이 PR이 합쳐져도 **이미 공개된 커밋 이력에는 데이터가 남는다.** 이력에서 지우려면 별도의 이력 정리(force push)가 필요하며, 팀 전체의 합의가 필요하다.
