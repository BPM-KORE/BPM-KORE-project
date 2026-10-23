# 데이터·인터페이스 계약 v1.0

A, B, C가 서로를 기다리지 않고 일하려면 파일 형식이 하나여야 한다. 이 문서가 그 기준이다.
코드 쪽 정의는 `kore/schemas.py`에 있고, 두 곳이 어긋나면 `tests/`가 실패하도록 해 두었다.

## 1. 공통 규칙

- 인코딩은 UTF-8이다. 엑셀이나 구글 시트에서 내려받을 때 "CSV UTF-8"을 고른다(BOM이 붙어도 읽는다).
- 한 행이 한 레코드다. 셀 안의 줄바꿈과 쉼표는 따옴표로 감싸면 된다(시트 프로그램이 알아서 한다).
- 목록은 `|`로 잇는다. 예: `ast_001|ast_005`. 그래서 id에는 `|`와 공백을 쓰지 않는다.
- id는 영문, 숫자, `_`, `.`, `-`만 쓴다. 한번 정한 id는 바꾸지 않는다(질문의 근거 연결이 깨진다).
- **통합 조건: `python -m kore validate`가 오류 0건이어야 한다.** 경고는 사람이 읽고 판단한다.

## 2. `data/knowledge.csv` (담당: B)

| 열 | 필수 | 설명 |
|---|---|---|
| `id` | 예 | 문서 고유 id. 분야 접두사를 권장한다(`ast_001`) |
| `domain` | 예 | 분야 이름. 철자를 통일한다(같은 분야를 두 가지로 쓰지 않는다) |
| `title` | 예 | 문서 제목 |
| `text` | 예 | 본문. 한 문서에 한 주제만 담는다. 30자 미만이면 경고, 6000자 초과도 경고 |
| `source_url` | 예 | 원문 주소. `http(s)://`로 시작한다. 직접 쓴 문서는 `local:설명`으로 적는다 |
| `license` | 예 | 라이선스 이름(예: `CC-BY-4.0`). 확인하지 못했으면 `unknown`이라 적고, 확인 전에는 연구에 쓰지 않는다 |
| `reliability` | 아니오 | 0~1. 출처 신뢰도. 비우면 0.5. 기준(예: 논문 1.0, 공식 문서 0.8, 블로그 0.4)을 `data/source_log.csv`에 적어 둔다 |
| `lang` | 아니오 | `ko` 또는 `en` |

검증기가 잡는 것: 필수 열 누락, 빈 필수값, id 형식·중복, reliability 범위, 완전히 같은 본문(경고), 라이선스 미확인(경고), 주소 형식(경고).
검증기가 **못** 잡는 것: 내용이 사실인지, 라이선스를 제대로 읽었는지. 이건 B가 원문을 직접 열어서 확인한다.

### 출처 기록 `data/source_log.csv` (담당: B)

열: `id, source_url, accessed_date, license_checked, verified_by, note`. 형식만 정했고 검증기는 아직 없다.
`license_checked`는 `yes`/`no`, `accessed_date`는 `YYYY-MM-DD`로 적는다.

## 3. `data/evaluation.csv` (담당: C)

| 열 | 필수 | 설명 |
|---|---|---|
| `question_id` | 예 | 질문 고유 id (`q001`) |
| `question` | 예 | 질문 문장 |
| `expected_source_ids` | 예 | 정답 근거 문서 id를 `|`로 잇는다. knowledge에 실제로 있는 id여야 한다 |
| `domain` | 예 | 질문의 분야 |
| `difficulty` | 예 | `easy`, `medium`, `hard` 중 하나 |
| `split` | 예 | `train`, `dev`, `test` 중 하나 |
| `reference_answer` | 둘 중 하나 | 사람이 읽는 모범 답안 |
| `rubric` | 둘 중 하나 | 자동 채점용 키워드. 항목은 `|`, 한 항목 안의 동의어는 `/`로 잇는다 |

`rubric` 예: `lower/lowers|alternative` 는 "lower 또는 lowers"와 "alternative"가 모두 답변에 있어야 만점이다.
키워드 일치는 의미 평가의 대용치일 뿐이다. 숫자 항목(`7`)은 `17`에 걸리지 않게 경계를 본다.

### 질문을 쓸 때 지킬 것

- 근거 문서의 문장을 그대로 베끼지 않는다. 베끼면 BM25가 유리해져서 검색 성능이 부풀려진다.
- 정답 키워드를 질문에 넣지 않는다(검증기가 경고한다).
- 근거가 둘 이상 필요한 질문을 일부 넣는다. 지식 조합 효과(H2)는 이런 질문이 있어야 볼 수 있다.
- 모델이 근거 없이도 맞히는 질문(닫힌북 정답)을 걸러내는 절차는 `docs/PROTOCOL.md`에 있다.

### 분할 규칙 (C가 지킨다)

- `python -m kore split --questions 원본.csv --out evaluation.csv --seed 0`으로 분야별 층화 배정을 한다.
- **한번 정한 split은 옮기지 않는다.** 질문을 새로 추가하는 건 괜찮지만, 기존 질문을 test에서 빼거나 넣지 않는다.
- test 질문의 결과는 최종 평가 전에는 보지 않고, 그 결과를 보고 알고리즘을 고치지 않는다.

## 4. `data/relations.csv` (담당: B, 선택)

열: `src_id, dst_id, relation, evidence_url`(앞의 세 열이 필수). 모든 관계에 근거 주소를 붙인다. MVP에서는 검색과 선택에 쓰지 않고 형식만 고정했다.
`python -m kore validate --knowledge data/knowledge.csv --relations data/relations.csv`로 검증한다.
양 끝 id가 knowledge에 있는지, 자기 자신을 가리키는지, 중복인지, 근거 주소가 있는지 본다.

## 5. 파이썬 인터페이스

| 원문서(7절) | 구현 | 비고 |
|---|---|---|
| `retrieve(question, k=10)` | `BM25Retriever.retrieve(question, k=10, domains=None)` | 점수가 0보다 큰 문서만 최대 k개 돌려준다 |
| `select_knowledge(question, candidates, k=3)` | `select_knowledge(question, candidates, k=3, strategy="kore", weights=None, token_budget=None, tokenizer_mode="ngram", term_weights=None)` | `strategy`는 `similarity`(기준선)와 `kore` |
| `evaluate(question, selected, reference)` | `evaluate(question, selected, reference, answer=None)` | `reference`에 `expected_source_ids`, `rubric_items`를 담는다 |

`KnowledgeItem`은 원문서의 필드(`id, domain, title, text, source_url, license, score`)를 그대로 두고 아래를 **추가**했다.
원문서 7절의 "score를 단계별로 분리하는 편이 좋다"는 권고를 반영한 것이다.

| 추가 필드 | 의미 |
|---|---|
| `retrieval_score` | 검색 단계(BM25) 점수 |
| `selection_score` | 선택 단계 점수 |
| `reliability` | 0~1 출처 신뢰도 |
| `lang` | `ko`/`en` |

`score`는 계약 호환용으로 남겼고, 가장 최근 단계의 점수가 들어간다.

## 6. 검증과 올리기

```bash
python -m kore validate --knowledge data/knowledge.csv --questions data/evaluation.csv
```

오류가 있으면 종료 코드 1을 돌려주고, 줄 번호와 이유를 알려 준다. Git이 익숙하지 않으면 GitHub 웹에서
"Add file → Upload files"로 `data/` 폴더에 올리고 Pull Request를 만든다. CI가 같은 검증을 자동으로 돌린다.

## 7. 계약을 바꾸려면

열 추가·이름 변경은 `kore/schemas.py`, 이 문서, 샘플 데이터, 테스트를 한 PR에서 같이 고치고 세 사람이 모두 확인한다.
`CONTRACT_VERSION`을 올리고, 이미 만든 데이터의 이전 방법을 PR 설명에 적는다.
