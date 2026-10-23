# KORE: Knowledge Orchestration & Retrieval Engine

기존 언어모델에 붙이는 전문지식 검색·선택 엔진을 만드는 3인 연구 프로토타입이다.
질문에 **실제로 도움이 될 지식**을 골라서 모델에 넘기는 방법을 연구하고, 단순 RAG와 같은 조건에서 비교한다.

> **상태(2026-10-23)**: 뼈대와 실험 러너가 동작한다. KORE 선택기가 단순 유사도 선택보다 낫다는 것, 연구 가설 H1·H2·H3, 그리고 독창성은
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

핵심 코드는 표준 라이브러리만 쓰고 CPU에서 돈다. 파이썬 3.10 이상이 필요하다.

### Colab에서 한 번에

`notebooks/01_colab_quickstart.ipynb`의 첫 코드 셀 하나만 실행하면 설치, 테스트, 검증, 샘플 실험이 끝난다.
셀 맨 위의 `REPO_URL`을 팀 저장소 주소로 바꾼다. 결과는 Drive에 저장되고, 연결이 끊겨도 같은 셀을 다시 실행하면 이어서 돈다.

## 구조

```
kore/
  schemas.py          데이터 계약(KnowledgeItem, EvalQuestion)
  io.py               knowledge/evaluation 읽기와 검증
  tokenize.py         한국어·영어 혼용 토크나이저(CPU)
  retrieval.py        BM25 검색기, 분야 라우팅, 질문어 IDF
  selector.py         similarity(기준선)와 kore(가중 점수) 선택기
  evaluation.py       Recall/Precision/Hit/MRR, 루브릭, 부트스트랩 구간, 쌍체 비교
  generation.py       생성기 인터페이스(추출형 기준선, Hugging Face 래퍼)
  pipeline.py         재개 가능한 실험 러너(no_retrieval / vanilla_rag / kore / ablation)
  tuning.py           가중치 탐색(test 분할을 구조적으로 거부)
  memory.py           검색 경험 기록(test 질문은 기록 거부)
  knowledge_graph.py  관계 파일 형식과 검증(MVP에서는 검색에 쓰지 않음)
  cli.py              validate / run / search / split
tests/                pytest
data/                 샘플 데이터(실제 데이터는 knowledge.csv, evaluation.csv)
configs/              실험 설정 JSON
notebooks/            Colab 노트북
docs/                 CONTRACT(데이터 계약), BOUNDARIES(작업 경계), PROTOCOL(실험 규칙), REPORT(검토 보고서)
results/              실험 결과(실행마다 하위 폴더 + 산출 보고서)
app.py                Gradio 데모
CLAUDE.md             Claude Code가 읽는 규칙
```

## 문서

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
