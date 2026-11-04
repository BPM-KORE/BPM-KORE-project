# Colab + Drive 기준선 실행 셀, test manifest 도구 (산출 보고서)

작성: 2026-11-03, 브랜치 `exp/colab-drive-2026-11-03`. `kore/`, `data/`, 설정은 바꾸지 않았다.

## 무엇을 했나

1. `experiments/colab_drive/run_baselines_cell.py`: Colab에 통째로 붙여넣는 단일 셀.
   Drive 마운트 -> 저장소 clone/pull(`REPO_URL`, `BRANCH` 변수, 토큰 없음) -> `pip install -e .` -> GPU 점검 ->
   `python -m kore run --config configs/default.json --knowledge team/provisional/law/knowledge.csv --questions <v2 CSV> --conditions no_retrieval vanilla_rag kore gold_context --generator hf --generator-config configs/generator.json --out Drive/KORE_results/<날짜>_<모델>_<revision 앞 7자리>` -> 요약 블록 1개.
   진행 출력은 `run_stdout.log`에만 쌓고 화면에는 요약 블록만 낸다.
2. `experiments/test_manifest/make_manifest.py`: `make`(test 질문마다 question_id, SHA-256, 글자 수 3개만 CSV로), `verify`(비공개 원본과 manifest 대조: 일치/불일치/원본에 없음/manifest에 없음의 id만 출력), `scan`(저장소 텍스트 파일에 test 질문·참조답이 그대로 있는지 검사. 파일 경로와 id만 출력). 질문·정답 내용은 어디에도 출력하지 않는다.
   해시 입력은 NFC + 공백 정리 후 질문, 참조답, 루브릭, 정렬한 근거ID를 U+001F로 이은 문자열이다(`|`는 루브릭·ID 안에 있어서 경계가 모호해 쓰지 않았다).
3. `experiments/test_manifest/test_manifest.py`: 합성 데이터로만 시험하는 7개 테스트.

## 재개 방식 (kore/pipeline.py를 읽고 확인)

- `run_experiment(resume=True)`(CLI 기본값)가 `run_log.jsonl`에서 끝난 (question_id, condition)을 읽어 건너뛴다. 잘린 마지막 줄은 새 줄로 분리하고 무시한다.
- `env.json`의 지문(설정 해시, 데이터 해시, 생성기 이름, seed)이 다르면 `ValueError`로 멈춘다. 생성기 이름에 모델·revision·dtype·장치(cuda/cpu)가 들어가므로, 중간에 GPU 종류가 바뀌면(예: T4 -> CPU) 같은 폴더로 이어지지 않는다.
- 셀은 같은 모델·revision의 폴더 중 `SUMMARY_DONE`이 없는 최신 폴더를 다시 쓴다(자정을 넘겨도 폴더가 갈라지지 않는다). 끝나면 `SUMMARY_DONE`, `summary.txt`를 쓴다.
- `results.csv`는 실행이 끝났을 때만 쓰인다. 중간 상태는 `run_log.jsonl`로 본다.

## 무엇을 검증했나 (GPU 없음, CPU extractive 생성기로 대체)

환경 변수로 마운트·clone·pip을 건너뛰고 임시 폴더를 Drive 대신 썼다. `KORE_GENERATOR=extractive`, 질문은 `/home/claude/bpm-kore-project/.../evaluation_questions_20_v2.csv`.

- 셀 실행 성공. 20문항(dev), 4조건, 출력 폴더 `KORE_results/2026-11-03_extractive_norev/`. 요약 블록:
  `no_retrieval rubric 0.000 / vanilla_rag 0.113 recall 0.483 / kore 0.113 recall 0.367 / gold_context 0.242 recall 1.000`.
  이 숫자는 extractive 대체 생성기의 값이라 연구 결과가 아니다. 명령 조립이 맞는지 보여 줄 뿐이다.
- 재개 시험: `run_log.jsonl`을 29줄(마지막 줄은 잘린 상태)로 잘라 같은 셀을 다시 실행 -> 유효 80행(20문항 x 4조건)으로 복구, 요약 출력.
- 이미 끝난 폴더에서 같은 셀을 다시 실행해도 같은 폴더를 쓰고 새로 계산하지 않았다.
- `python -m pytest -q`: 226 passed (기존 테스트 변화 없음). `python -m pytest -q experiments/test_manifest`: 7 passed.
- 실제 v2 CSV는 전부 dev라서 `make`가 "test 분할 질문이 없다"로 종료 코드 1을 냈다(의도한 동작). 실제 평가 데이터로는 manifest를 만들지 않았다.

## 미완성 (Colab에서 서준이 해 볼 것)

1. **GPU 경로 미검증.** `HFGenerator`의 모델 로딩, 생성, 요약 블록의 dtype·GPU 표시는 T4에서 처음 돌려야 안다. 첫 실행은 `configs/generator.json` revision의 모델 다운로드(약 8GB)가 있다.
2. 셀의 `BRANCH`는 기본 `main`이다. v2 CSV가 들어간 PR #3이 합쳐지기 전이면 그 브랜치 이름으로 바꾼다(CSV가 없으면 `kore run`이 바로 실패한다). Drive 마운트 승인 팝업이 한 번 뜬다.
3. 실제 test manifest는 아직 없다. test 분할이 생기면 비공개 Drive 원본으로 `make`를 돌려 `test_manifest.csv`만 커밋하고, 커밋 전에 `scan`과 `check_manifest_clean`을 돌린다.

## 한계

- 20문항, rubric 키워드 일치는 의미 평가의 대용치다. 조건 간 차이를 결론으로 쓰지 않는다.
- `experiments/test_manifest`는 기본 `testpaths = ["tests"]` 밖이라 `python -m pytest -q`에는 잡히지 않는다. 경로를 명시해서 돌린다(pyproject는 바꾸지 않았다).
- manifest의 해시는 내용이 한 글자만 달라도 불일치로 본다. 어느 필드가 바뀌었는지는 알려 주지 않는다. 글자 수 3개가 단서다.
- 짧은 test 문장(12자 미만)은 `scan`이 검색하지 않는다. 해시는 정규화 규칙(공백, NFC, ID 순서)에만 둔감하다.
- 공개 저장소이므로 manifest의 글자 수와 해시도 test 질문 존재를 알린다. 내용 복원은 어렵지만, 짧은 질문의 무차별 대입은 이론상 가능하다.
