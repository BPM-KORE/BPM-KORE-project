# 할 일: A·B·C

`docs/DECISIONS.md`의 결정을 팀이 받아들였다는 전제로 쓴 작업표다. 결정 중 바꾸고 싶은 게 있으면 먼저 DECISIONS에 이유를 적고 바꾼다.
주차는 시작일 기준 상대 주차다. 달력 날짜는 팀이 정한다. 명령은 저장소 맨 위 폴더에서 실행한다.
모든 명령은 먼저 `--help`로 옵션을 볼 수 있다(예: `python -m kore ingest --help`).

## 0. 모두가 지키는 규칙 (여섯 개)

1. `main`에는 테스트가 통과한 것만 올린다. 직접 올리지 않고 Pull Request로 합친다.
2. **test 분할 질문의 결과는 최종 평가 전에 보지 않는다.** `--final-test`는 5주차 승인 뒤 한 번만 켠다.
3. `blind_key.csv`(채점표 정답 키)는 채점이 끝나기 전에 저장소에 올리지 않고, 채점자에게 보여 주지 않는다.
4. 질문을 더 만든 뒤 다시 나눌 때는 반드시 `--keep-from`을 붙인다(아래 C의 3번). 붙이지 않으면 명령이 거부한다.
5. 출처를 직접 열어 보지 않은 문서는 `knowledge.csv`에 넣지 않는다. 이용 조건을 모르면 `unknown`으로 적고 B가 확인할 때까지 데이터로 쓰지 않는다.
6. 비밀키, 토큰, 개인정보를 파일이나 채팅에 붙이지 않는다. 막히면 이슈에 쓴다.

## 1. 첫 48시간

### A

1. `docs/START.md` 2~4단계대로 무료 조직을 만들고 그 아래 비공개 저장소로 올린다. B와 C를 조직에 초대하고 저장소에 Write 권한을 준다. B, C의 토큰 승인 요청을 승인한다.
2. 자기 노트북에서 `pip install -e ".[dev]"` 후 `python -m pytest -q`. 전부 통과해야 한다. 그 뒤 `python -m kore doctor`의 결과를 이슈에 붙인다(Windows면 어디서 막혔는지 같이).
3. Colab에서 `notebooks/01_colab_quickstart.ipynb`를 끝까지 돌린다. G0 통과.
4. Colab에서 `notebooks/02_colab_gpu_run.ipynb`의 1번(준비), 2번(GPU 점검) 셀을 돌린다. Colab Secrets에 `GITHUB_TOKEN`(읽기 전용 토큰)과 `KORE_REPO_URL`(저장소 주소)을 저장해 둔다(`START.md` 4단계).
5. 점검 결과(통과/실패, 초당 토큰 수, 호출당 시간)를 이슈 하나에 그대로 붙인다. 실패하면 `CONFIG`를 `configs/generator_fallback_qwen25_3b.json`으로 바꿔 다시 한다. G1.

### B

0. `docs/START.md` 1·4·6단계를 따라 계정, 토큰, `notebooks/03_colab_data_tools.ipynb`를 준비한다. 아래 명령은 노트북 03의 3-B 셀이 대신 돌려 준다(파이썬 설치 필요 없음).
1. `docs/CONTRACT.md` 2절과 `docs/DECISIONS.md` 3절을 읽는다.
2. 이용 조건을 직접 확인해서 `docs/DECISIONS.md` 3절 아래에 확인한 날짜와 함께 적는다: 한국어 위키백과 하단의 저작권 안내, 저작권법 제7조 원문, 국가법령정보센터의 이용 안내, 쓸 공공기관 문서의 공공누리 표시(제1유형인지).
3. 분야마다 문서 10개를 모은다(총 30개). 원문을 복사해서 txt 파일로 저장한 뒤 `ingest`로 올린다.
   ```bash
   python -m kore ingest --input raw/law_labor.txt --domain law --id-prefix law_labor \
     --source-url "https://www.law.go.kr/..." --license "저작권법 제7조(법령)" --mode article \
     --out data/knowledge.csv --append --accessed-date 2026-xx-xx --dry-run
   ```
   `--dry-run`으로 어떻게 잘리는지 먼저 보고, 맞으면 `--dry-run`을 빼고 다시 실행한다. 출처 기록(`data/source_log.csv`)이 같이 쌓인다.
4. `ingest`는 출처 기록에 `license_checked=no`로 쌓는다. 원문 페이지에서 이용 조건을 직접 읽고, 확인한 줄의 `license_checked`를 `yes`로, `verified_by`를 자기 이름으로 바꾼다. `kore stats`가 yes가 아닌 줄을 알려 준다.
5. `python -m kore validate --knowledge data/knowledge.csv` 통과 후 PR(웹에서 올리는 방법은 `START.md` 6단계).

### C

0. `docs/START.md` 1·4·6단계를 따라 준비한다. 분할과 점검은 노트북 03의 3-C 셀이 대신 돌려 준다.
1. `docs/CONTRACT.md` 3절의 "질문을 쓸 때 지킬 것"을 읽는다.
2. 분야마다 질문 7개를 쓴다(총 21개). 질문 한 줄마다 정답 근거 문서 id(B가 올린 것), 난이도, `reference_answer`(짧은 한 문장), `rubric`(항목 2~3개)을 모두 적는다. 21개 중 근거가 둘 이상인 질문이 6개 이상 되게 한다.
   - 근거 문장을 그대로 베끼지 않는다. 질문에 정답 키워드를 넣지 않는다.
   - B의 문서가 아직 없으면 B가 올린 파일을 기다린다. B의 첫 20개 문서가 `main`에 올라오면 시작한다.
3. 질문 파일은 `data/evaluation_raw.csv`에 쓴다(split 열은 `dev`로 두면 된다).
4. 처음 한 번만 나눈다:
   ```bash
   python -m kore split --questions data/evaluation_raw.csv --knowledge data/knowledge.csv --out data/evaluation.csv --seed 0
   python -m kore validate --knowledge data/knowledge.csv --questions data/evaluation.csv
   ```
5. 통과하면 PR.

## 2. 3~4일차: 분야 점검 (G2)

A가 Colab에서 한다. 세 분야에 질문이 7개씩 있어야 한다.

```bash
python -m kore run --knowledge data/knowledge.csv --questions data/evaluation.csv \
  --out results/g2_screen --generator hf --generator-config configs/generator.json \
  --conditions no_retrieval vanilla_rag gold_context
python -m kore screen --run-dir results/g2_screen
```

- `no_retrieval`이 높으면(모델이 이미 안다) 그 분야의 질문을 더 어려운 문서로 바꾼다. B와 C가 같이.
- `gold_context`가 낮으면 질문, 루브릭, 모델 중 하나가 문제다. `screen.csv`에 표시된 질문을 C가 먼저 열어 본다.
- 후보 재현율이 낮으면 검색이 정답 문서를 후보 10개에 못 올린 것이다. 질문 표현과 문서 길이를 보고, 그래도 안 되면 A가 토크나이저를 손본다.
- 분야 하나가 계속 통과하지 못하면 그 분야를 바꾼다. 결과는 이슈에 붙인다.

## 3. 역할별 상시 작업

### A (개발·통합)

- 매일 `git pull` 후 `python -m pytest -q`와 `kore validate`. 깨졌으면 그날 안에 알린다.
- Colab 실행을 맡는다. 세션이 끊기면 같은 셀을 다시 실행한다(이어서 돌아간다). 결과는 Drive의 `kore_results`에 쌓인다.
- 코드 변경은 작게 나누고 테스트를 붙인다. Claude Code를 쓸 때는 `CLAUDE.md`를 읽게 하고, 나온 코드는 테스트를 통과하기 전에는 믿지 않는다.
- 노트북에서 돌린 결과 폴더(`results/...`)의 요약 표를 C에게 넘긴다.

### B (데이터·연구)

- 문서 수집: 매 주 목표는 아래 4절. 문서는 100~1500자로 자르고, 한 문서에 한 가지 사실 덩어리만 둔다.
- 방해 문서를 일부러 섞는다: 비슷한 주제의 다른 조항, 같은 용어가 나오는 다른 제도. 모든 문서가 정답 문서와 거리가 멀면 선택 알고리즘을 시험할 수 없다.
- `python -m kore stats --knowledge data/knowledge.csv --questions data/evaluation.csv --stage week1` (또는 week2, final)을 문서를 올릴 때마다 돌려 필수 항목을 맞춘다. 복사 위험과 중복 문서 경고가 나오면 고친다.
- 논문 조사: `CLAUDE.md`에 적힌 선행연구 중 H1·H2와 가장 가까운 두 편(ContextCite, Shapley 기반 RAG 귀속 연구 arXiv 2507.04480)을 원문으로 읽고, 우리가 하려는 것과 무엇이 같고 다른지 한 쪽으로 적는다. 읽기 전에는 독창성을 말하지 않는다. 3주차 전까지.

### C (평가·재현)

- 질문 작성: 주차 목표는 4절. 모든 질문에 `reference_answer`와 `rubric`을 둘 다 적는다(`reference_answer`는 기여도 라벨링에 쓴다).
- 질문을 늘려서 다시 나눌 때(튜닝이나 라벨링을 이미 한 뒤에도 안전하다):
  ```bash
  python -m kore split --questions data/evaluation_raw.csv --knowledge data/knowledge.csv \
    --out data/evaluation.csv --keep-from data/evaluation.csv --seed 0
  ```
  "이전 분할을 유지한 질문 N개"에서 N이 이전 질문 수와 같은지 확인한다.
- 매주 결과 확인: 조건별 표(`results.csv`), 쌍체 차이(`comparisons.csv`)를 읽고, 숫자가 이상하면 원인을 적는다(예: 한 분야만 다르다, 한 질문이 점수를 끌어올린다).
- 사전 등록 파일(`docs/prereg/`)을 5주차 전에 쓴다. 양식은 `docs/PROTOCOL.md` 2절.
- 사람 블라인드 채점(아래 5절)을 맡는다.

## 4. 주차별 계획 (7주)

| 주 | A | B | C | 그 주의 끝 |
|---|---|---|---|---|
| 1 | 1·2절 전부. G0, G1, G2 | 분야별 문서 17개 이상(총 50개). 이용 조건 확인 | 분야별 질문 10개(총 30개). 처음 split | `stats --stage week1` 필수 통과 |
| 2 | 노트북 02의 2-1 셀로 검색기 비교(`kore retrieval-eval`): 하이브리드가 후보 재현율을 확실히 올리면 켠다. 그다음 개발용 실행(3번 셀) | 총 100개 | 총 60개. `--keep-from`으로 다시 나눔 | G3: `stats --stage week2` 통과, 후보 재현율 0.8 이상 |
| 3 | 기여도 라벨링(노트북 02의 4번 셀, dev만, 후보 6개·크기 2) | 총 150개. 선행연구 한 쪽 요약 | 총 90개. 닫힌북 점검 결과로 질문 고침 | G4: `kore label-report`로 H1 갈림길 판정(DECISIONS 6절) |
| 4 | G4 신호가 있으면: train까지 넓히고 후보 8·크기 3. 없으면: `kore tune`으로 가중치 선택기 비교 | 총 200개. 근거 둘 이상 질문에 쓸 문서 보강 | 총 120개. 블라인드 채점표 만들기 | 데이터 동결 후보 |
| 5 | 데이터 동결 후 `stats --stage final`. 가중치 확정(dev만) | 출처 기록과 이용 조건 전수 확인 | 사전 등록 작성, 세 명 승인 | G5. 승인 뒤에만 `--final-test` |
| 6 | 최종 실행(test 한 번) | 선행연구와의 차이 정리 | 결과 표, 사람 채점 결과 정리 | 결과 확정 |
| 7 | 재현 확인(깨끗한 Colab에서 처음부터), 데모. 공개하기로 했으면 데이터 카드 | 보고서의 배경·한계 | 보고서의 방법·결과 | 보고서와 재현 가이드 |

규모는 DECISIONS 4절의 목표(문서 약 200, 질문 120)다. 시간이 모자라면 아래 최소 성공선으로 간다.

## 5. 사람 블라인드 채점

자동 rubric 점수가 사람의 판단과 맞는지 확인하는 절차다. 채점자는 2명이어야 일치도를 낼 수 있다(B와 C).

1. A가 노트북 02의 5번 셀로 채점표를 만든다(test가 아닌 질문에서 30개, 어느 조건의 답인지 가려짐). 같은 답은 하나로 합쳐진다.
2. `blind_sheet.csv`를 B와 C에게 따로 준다. `blind_key.csv`는 A만 갖고 채점이 끝나기 전에 아무도 열지 않는다.
3. 각자 상의하지 않고 0(틀림), 1(부분), 2(맞음)로 채점한다.
4. 둘 다 끝나면:
   ```bash
   python -m kore blind-score --sheet sheet_B.csv sheet_C.csv --key blind_key.csv
   ```
   일치도(가중 카파)와 자동 점수와의 순위상관이 나온다. 일치도가 낮으면 채점 기준부터 다시 맞춘다.
5. 채점 후 키를 저장소에 올린다.

## 6. 최소 성공선 (시간이 모자랄 때)

주당 시간이 계획보다 적거나 분야 하나가 G2를 통과하지 못할 때의 하한선이다.

- 문서 100개, 질문 60개(분야 2~3개), 한 번의 `kore` 대 `vanilla_rag` 비교.
- test는 12개 안팎이라 작은 차이는 구분되지 않는다. 결과 문장은 "이 표본으로는 구분하지 못했다"일 수 있고, 그것도 정상적인 결과다.
- 기여도 라벨링은 dev에서 후보 6개·크기 2만 하고, H1 라벨링 결과를 예비 신호로만 보고한다.
- 학습 기반 선택기(H3)는 하지 않는다.

## 7. 알아둘 것

- **Windows(A)**: 핵심 코드는 표준 라이브러리만 쓰지만 이 환경에서는 Linux만 시험했다. 막히면 `kore doctor` 결과와 오류 메시지를 이슈에 붙인다. 경로는 `/`로 써도 된다. 한글이 깨지면 CSV를 UTF-8로 저장했는지 먼저 본다(엑셀에서 저장할 때 "CSV UTF-8"을 고른다).
- **엑셀로 CSV를 열 때**: `|`로 이은 목록과 `/` 동의어가 그대로 유지되는지, id가 `1.0E+03`처럼 바뀌지 않았는지 저장 전에 본다.
- **Colab**: GPU는 보장되지 않는다. 안 잡히면 시간을 바꿔 다시 연결한다. 계정을 여러 개 돌려서 제한을 피하지 않는다. 오래 걸리는 작업은 중간 결과가 Drive에 남게 되어 있으니 끊겨도 같은 셀을 다시 실행한다.
- **모델 다운로드**: 처음 한 번 수 GB를 받는다. 세션마다 다시 받으므로 시간이 걸린다.
- **샘플 데이터**(`data/sample_*.csv`)의 결과는 배관 점검용이다. 보고서나 발표에 쓰지 않는다.
- **막히면**: 이슈에 (1) 실행한 명령 (2) 오류 전문 (3) 기대한 결과를 적는다.
