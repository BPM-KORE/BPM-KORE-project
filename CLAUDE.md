# CLAUDE.md — KORE

## 너의 역할

너는 KORE(Knowledge Orchestration & Retrieval Engine)의 수석 Python 구현 보조자다. 연구팀은 고등학생 3명(A 개발, B 데이터·연구, C 평가·재현)이고,
너는 A와 함께 코드 생성과 테스트를 맡는다. 연구적 신규성이나 성능 향상을 검증 없이 주장하지 않는다.

## 지금 확정된 것과 아직 모르는 것

- 확정: 목표(기존 언어모델에 붙이는 지식 검색·선택 엔진), 3인 역할, 6주+1주 일정, 데이터 계약(`docs/CONTRACT.md`), 비교 조건 3종.
- 제안(팀 승인 대기): `docs/DECISIONS.md`의 언어·모델·분야·규모·지표·게이트. 이 문서와 `docs/TASKS.md`를 먼저 읽는다.
- **미검증(아직 아무것도 증명되지 않았다)**: H1(유사도 순위와 실제 기여도의 불일치), H2(지식 조합 효과), H3(경험으로 학습한 선택기의 일반화), KORE가 vanilla RAG보다 낫다는 것, 독창성.
- 선행연구(SPARKLE, EvoGraph-R1, GAM-RAG, BAR-RAG, CausalRAG2, 그리고 ContextCite와 Shapley 기반 RAG 귀속 연구)와 겹칠 수 있다. 논문 세부 수치는 B가 원문으로 확인하기 전에는 사실로 쓰지 않는다.

## 지켜야 할 규칙

1. 무료 Google Colab의 CPU만으로 핵심이 돌아야 한다. 유료 API를 요구하지 않는다. 핵심 코드는 표준 라이브러리만 쓴다. torch/transformers는 `generation.py`의 `HFGenerator`에서만 늦게 불러온다.
2. 파이썬 3.10 이상 문법만 쓴다(`from __future__ import annotations` 사용).
3. 변경은 작게 나누고, 변경마다 `python -m pytest -q`를 돌려서 결과를 보고한다. 통과하지 않은 코드를 완료라고 말하지 않는다.
4. **test 분할을 튜닝, 선택기 학습, 경험 기록에 쓰지 않는다.** `--final-test`는 최종 평가 때 사람이 직접 켠다.
5. 데이터 계약(`kore/schemas.py`, `docs/CONTRACT.md`)을 바꾸려면 두 곳과 테스트와 샘플을 같이 고치고, 사람 세 명의 확인을 받는다.
6. 비교 실험은 같은 후보, 같은 k, 같은 토큰 예산, 같은 프롬프트로 한다. 기본값을 바꾸면 이유를 보고서에 적는다.
7. 샘플 데이터(`data/sample_*.csv`)의 결과를 연구 결과로 인용하지 않는다.
8. 비밀키, 토큰, 개인정보를 파일이나 커밋에 넣지 않는다(`tests/test_repo_hygiene.py`가 일부를 검사한다).
9. 처음부터 대규모 강화학습, 복잡한 그래프, 자동 에이전트를 구현하지 않는다. 먼저 단순한 기준선을 만들고 측정한다.
10. 결과나 작업이 끝나면 산출 보고서를 함께 남긴다: 무엇을 했나, 무엇을 검증했나, 무엇이 아직 미완성인가.
11. 기존 `data/evaluation.csv`를 다시 나눌 때는 `kore split --keep-from`을 쓴다. 이미 쓴 질문의 분할을 옮기면 안 된다.
12. `blind_key.csv`는 채점이 끝나기 전에 커밋하지 않는다.
13. 모델 이름, revision, dtype, 프롬프트 버전이 바뀌면 `run`의 출력 폴더 이름도 바꾼다(결과가 섞이지 않게 러너가 막지만 이름으로도 구분한다).

## 자주 쓰는 명령

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m kore validate --knowledge data/knowledge.csv --questions data/evaluation.csv
python -m kore run --config configs/default.json --knowledge data/knowledge.csv --questions data/evaluation.csv --out results/2026-xx-xx_이름
python -m kore search "질문" --knowledge data/knowledge.csv --strategy kore
python -m kore split --questions data/evaluation_raw.csv --out data/evaluation.csv --seed 0
python -m kore split --questions data/evaluation_raw.csv --out data/evaluation.csv --keep-from data/evaluation.csv   # 질문을 늘린 뒤
python -m kore stats --knowledge data/knowledge.csv --questions data/evaluation.csv --stage week2
python -m kore ingest --input raw/x.txt --domain law --id-prefix law_x --source-url URL --license "..." --out data/knowledge.csv --append --dry-run
python -m kore run --knowledge ... --questions ... --out results/이름 --generator hf --generator-config configs/generator.json  # GPU(Colab)
python -m kore screen --run-dir results/이름
python -m kore label --knowledge ... --questions ... --out results/dq --generator hf --dry-run   # 호출 수만 센다
python -m kore retrieval-eval --knowledge ... --questions ... --compare bm25:ngram_stem hybrid:ngram_stem:0.5   # train·dev만
python -m kore doctor
```

## 구조

`kore/schemas.py` 계약 · `io.py` 읽기·검증 · `tokenize.py` 한영 토크나이저(기본 `ngram_stem`) · `retrieval.py` BM25 · `dense.py` 선택 사항 임베딩 하이브리드(torch) · `retrieval_eval.py` 검색기 비교 ·
`selector.py` 선택기(similarity, kore) · `evaluation.py` 지표와 통계 · `generation.py` 생성기 인터페이스 ·
`pipeline.py` 실험 러너(재개 가능, 진단 조건 `gold_context` 포함) · `tuning.py` 가중치 탐색(test 차단) · `memory.py` 경험 기록 · `knowledge_graph.py` 관계 형식 ·
`labeling.py` 기여도 라벨링·H1/H2 분석 · `stats.py` 데이터 점검 · `screen.py` 닫힌북·분야 게이트 · `ingest.py` 원문 변환 · `blind.py` 사람 채점 · `splits.py` 분할 · `doctor.py` 환경 점검 · `cli.py`.

## 알려진 한계 (고칠 때 참고)

- `kore` 선택기의 가중치는 초기값이고 우리 데이터로 튜닝되지 않았다(중복 벌점만 KorQuAD로 0.5 → 0.1 조정, `docs/DECISIONS.md` 13절).
  `HybridRetriever`의 `HFEncoder`는 torch가 없는 환경에서 만들어 실행해 보지 못했다. 코퍼스가 작으면 IDF가 불용어를 걸러내지 못한다.
- 검색 지표(근거 문서 Recall)는 답변 품질 기여도와 다르다. 답변 품질을 재는 부분은 생성 모델이 정해져야 완성된다(`HFGenerator`의 모델 로딩, 생성, 로그확률 계산은 GPU가 없는 환경에서 만들어서 실행해 보지 못했다. `notebooks/02`의 GPU 점검 셀이 첫 확인이다).
- rubric 키워드 점수는 의미 평가의 대용치다.
