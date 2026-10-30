# KORE: Knowledge Orchestration & Retrieval Engine

기존 언어모델에 붙이는 전문지식 검색·선택 엔진을 만드는 3인 연구 프로토타입이다.
질문에 **실제로 도움이 될 지식**을 골라서 모델에 넘기는 방법을 연구하고, 단순 RAG와 같은 조건에서 비교한다.

> **상태(2026-10-23)**: 뼈대, 실험 러너, 무료 오픈 모델 생성기, 기여도 라벨링, 데이터 점검 도구가 코드로 있다. 모델을 올리는 부분은 GPU가 없는 환경에서 만들어서 Colab 점검 셀(`notebooks/02`)로 처음 확인해야 한다. KORE 선택기가 단순 유사도 선택보다 낫다는 것, 연구 가설 H1·H2·H3, 그리고 독창성은
> **아직 하나도 검증되지 않았다.** 샘플 데이터(15문서·9질문)는 배관 점검용이라 어떤 결론도 뒷받침하지 못한다.

## 빠른 시작

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m kore validate --knowledge data/sample_knowledge.csv --questions data/sample_eval.csv
python -m kore run --config configs/default.json \
    --knowledge data/sample_knowledge.csv --questions data/sample_eval.csv --out results/sample_run
python -m kore search "Which planet is the largest in the Solar System?" --knowledge data/sample_knowledge.csv
```

핵심 코드는 표준 라이브러리만 쓰고 CPU에서 돈다. 파이썬 3.10 이상이 필요하다. 돈이 드는 API는 쓰지 않는다.

**처음 하는 사람은 [`docs/START.md`](docs/START.md)(GitHub에 올리고 Colab에서 돌리기)와 [`docs/TASKS.md`](docs/TASKS.md)(누가 무엇을)부터 읽는다.**

### Colab에서 한 번에

`notebooks/01_colab_quickstart.ipynb`의 첫 코드 셀 하나만 실행하면 설치, 테스트, 검증, 샘플 실험이 끝난다.
셀 맨 위의 `REPO_URL`을 팀 저장소 주소로 바꾼다. 결과는 Drive에 저장되고, 연결이 끊겨도 같은 셀을 다시 실행하면 이어서 돈다.

## 명령 한눈에

| 명령 | 하는 일 | 주로 쓰는 사람 |
|---|---|---|
| `kore doctor` | 파이썬, GPU, 패키지, 데이터 폴더 점검 | A |
| `kore ingest` | 붙여 넣은 원문(법령, 안내문, 위키)을 문서 CSV와 출처 기록으로 자른다 | B |
| `kore validate` | 데이터 계약 검증 | B, C |
| `kore stats` | 데이터 건강 점검(규모, 균형, 복사 위험, 중복, 후보 재현율, 출처) | B, C |
| `kore split` | 질문을 분야별로 층화해 train/dev/test 배정. 질문을 늘려 다시 나눌 때는 `--keep-from` | C |
| `kore run` | 조건별 실험(`--generator hf`면 무료 오픈 모델로 생성) | A |
| `kore screen` | 닫힌북 필터와 분야 게이트 | A, C |
| `kore tune` | 가중치 탐색(train·dev만) | A |
| `kore label` / `label-report` | 문서 기여도 라벨링과 H1·H2 예비 분석 | A |
| `kore blind-export` / `blind-score` | 사람 블라인드 채점표 만들기와 채점 | A, B, C |
| `kore retrieval-eval` | 검색기 설정 비교(BM25, 토크나이저, 임베딩 하이브리드), train·dev만 | A |
| `kore search` | 질문 하나로 검색 결과 보기 | 모두 |

## 구조

```
kore/
  schemas.py          데이터 계약(KnowledgeItem, EvalQuestion)
  io.py               knowledge/evaluation 읽기와 검증
  tokenize.py         한국어·영어 혼용 토크나이저(CPU)
  retrieval.py        BM25 검색기, 분야 라우팅, 질문어 IDF
  dense.py            선택 사항: 임베딩 + BM25 하이브리드 검색(torch 필요)
  retrieval_eval.py   검색기 비교
  selector.py         similarity(기준선)와 kore(가중 점수) 선택기
  evaluation.py       Recall/Precision/Hit/MRR, 루브릭, 부트스트랩 구간, 쌍체 비교
  generation.py       생성기(추출형 기준선, 무료 Hugging Face 모델 HFGenerator, 프롬프트 버전 기록)
  pipeline.py         재개 가능한 실험 러너(no_retrieval / vanilla_rag / kore / ablation / gold_context)
  tuning.py           가중치 탐색(test 분할을 구조적으로 거부)
  memory.py           검색 경험 기록(test 질문은 기록 거부)
  knowledge_graph.py  관계 파일 형식과 검증(MVP에서는 검색에 쓰지 않음)
  labeling.py         문서 기여도 라벨링과 H1·H2 분석(ΔQ)
  stats.py            데이터 건강 점검
  screen.py           닫힌북 필터, 분야 게이트
  ingest.py           원문 → 문서 CSV 변환
  blind.py            사람 블라인드 채점표와 채점
  doctor.py           환경 점검
  splits.py           분할과 누출 점검
  cli.py              위 명령의 입구
tests/                pytest
data/                 샘플 데이터(실제 데이터는 knowledge.csv, evaluation.csv)
configs/              실험 설정 JSON
notebooks/            Colab 노트북(01 빠른 시작, 02 GPU 실행, 03 B·C 데이터 작업)
docs/                 DECISIONS(결정), TASKS(할 일), CONTRACT(데이터 계약), BOUNDARIES(작업 경계), PROTOCOL(실험 규칙), REPORT(보고서)
results/              실험 결과(실행마다 하위 폴더 + 산출 보고서)
app.py                Gradio 데모
CLAUDE.md             Claude Code가 읽는 규칙
```

## 문서

- [처음 시작하기](docs/START.md): GitHub 조직·저장소·토큰, Colab 실행, B·C의 데이터 올리기
- [결정 기록](docs/DECISIONS.md): 언어, 모델, 분야, 규모, 지표, 게이트와 그 근거
- [할 일](docs/TASKS.md): A·B·C별 작업과 7주 계획
- [데이터·인터페이스 계약](docs/CONTRACT.md): B와 C가 만들 파일의 열과 규칙
- [작업 경계](docs/BOUNDARIES.md): 누가 어느 파일을 맡고 어떻게 합치는가
- [실험 프로토콜](docs/PROTOCOL.md): 사전 등록, test 규칙, 통계, 보고 규칙
- [검토 보고서](docs/REPORT_2026-10-23.md): 확정과 미검증의 구분, 실현 가능성, 수정 제안, 선행연구 겹침

## 데모

```bash
pip install -e ".[demo]"
python app.py
```

Gradio가 필요하다. 검색 함수는 테스트로 검사하고, UI 객체가 만들어지는 것까지는 gradio 6.30.0에서 확인했다.
브라우저에서 버튼을 눌러 보는 것까지는 해 보지 못했으니, 처음 실행하면 화면 동작을 직접 확인한다.
