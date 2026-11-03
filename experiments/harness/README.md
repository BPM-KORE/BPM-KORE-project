# Harness² 구조 MVP 산출 보고서 (2026-11-03)

## 한 줄 요약

Google Harness²의 제안(Propose), 시험(Probe), 조합(Compose) 구조만 KORE에 맞춰 자체 구현한 뼈대다. 공식 구현은 유료 Vertex AI가 필요해서 쓰지 않았다.
가짜 생성기로 배관이 도는 것까지만 확인했다. **검증된 개선은 하나도 없다. 이 폴더의 어떤 숫자도 성능 주장으로 쓰지 않는다.**

## 무엇을 했나

| 경로 | 내용 |
|---|---|
| `harness/components.py` | 편집 가능한 구성요소 4종(선택기 가중치, k, 프롬프트 틀, 질의 변환 규칙)과 편집 기록. 상태: 제안, 시험중, 사람 승인 대기, 채택, 기각, 무효 |
| `harness/safety.py` | test 분할 거부, 제안 안의 금지 내용 검사, 실험 폴더 밖 쓰기 거부, 소스 코드 외부 호출 검사 |
| `harness/memory.py` | `kore.memory.ExperienceLog`를 그대로 재사용하고 교훈 기록(`lessons.jsonl`)을 덧붙인 래퍼 |
| `harness/stats.py` | 작은 dev 세트의 한계를 보고서에 자동으로 적는다(문항 수, 부트스트랩 구간, 구분 불가 경고) |
| `experiments/harness/loop.py` | 한 바퀴 루프, 체크포인트, 상한, 사람 승인(`approve`), CLI |
| `experiments/harness/fakes.py` | 모델 없이 도는 가짜 생성기(`ExtractiveGenerator` 기반) |
| `experiments/harness/proposals/example_proposal.json` | 예시 제안 파일 |
| `experiments/harness/test_harness.py` | 시험 37개 |

`kore/`와 `data/`는 수정하지 않았다.

### 한 바퀴의 흐름

1. **제안**: 사람이나 Claude Code가 쓴 제안 JSON(`{"edits":[{id, component, change, rationale, confidence}]}`)을 읽는다. 제안 파일은 `status`나 `result`를 정할 수 없다. 검사에 실패한 편집은 버리지 않고 '무효'와 이유를 남기며, 위험한 원문은 보관하지 않고 해시만 남긴다.
2. **시험**: 편집마다 `kore.pipeline.run_experiment`를 `kore` 조건으로 호출한다. 같은 dev 질문, 같은 시드, 같은 후보 풀 설정(`candidate_k`, 검색기, 토큰 예산은 편집 불가)을 쓴다. 기준선은 편집 없는 설정이다.
3. **조합**: 문항별 점수 차이표(`table_it{n}.csv`)와 `report.md`를 만든다. 편집은 `사람 승인 대기`에서 멈춘다. 루프 코드는 채택을 호출하지 않는다.
4. **승인**: 사람이 `approve`로 채택·기각한다. 시험을 마친 편집만 결정할 수 있고, 승인자와 이유가 없으면 거부한다. 이때 교훈이 근거 숫자(문항 수, 평균 차이, 구간)와 함께 저장된다. 채택은 다음 바퀴의 기준 설정에만 반영되고 `kore/`나 `configs/`는 바뀌지 않는다.

상한: 반복 최대 5회(고정), 시간(초), 생성기 호출 수, 바퀴당 유효 편집 8개. 매 편집이 끝날 때마다 `checkpoint.json`을 원자적으로 저장한다.

### 비교 조건을 지키지 못하는 편집

"같은 후보, 같은 k, 같은 토큰 예산" 원칙을 편집 자체가 깨는 경우가 있다. 숨기지 않고 시험표에 '바뀐 비교 조건'으로 적는다.

- `k` 편집: k가 바뀐다.
- `query_rules` 편집: 질의가 달라져서 후보 목록이 바뀐다. 변환된 질문이 생성기 프롬프트에도 들어간다(파이프라인이 검색용 질의와 생성용 질문을 나누지 않기 때문이다).
- `prompt_template` 편집: 프롬프트가 바뀐다(이것이 편집의 목적이다).
- 후보 수, 토큰 예산, 검색기는 편집을 거부한다.

## 무엇을 검증했나

`python -m pytest -q`(기존 226개 통과), `python -m pytest -q experiments/harness`(37개 통과). 이 폴더의 시험은 `pyproject.toml`의 `testpaths = ["tests"]`에 들어 있지 않아서 경로를 따로 줘야 한다. 기본 실행에 넣으려면 `testpaths`를 바꾸거나 `tests/`에 얇은 import 파일을 두어야 하는데, 새 파일은 두 폴더에만 만들라는 지시 때문에 하지 않았다. 팀 결정이 필요하다.

| 요구 | 확인한 방법 |
|---|---|
| (i) test 분할 거부 | `load_dev_questions`가 test 행을 객체로 만들지 않는다. `splits`에 test가 있으면, 혼합 질문 목록을 `HarnessLoop`에 넣으면, 경험 기록에 test를 넣으면 모두 `SplitViolation` |
| (ii) 외부 호출 없음 | AST로 `harness/`와 `experiments/harness/`의 모든 `.py`를 검사: 금지 import(requests, urllib, socket, subprocess, torch 등)와 URL·키 문자열이 없다. 검사기가 심어 둔 위반을 잡는지도 시험한다. `safety.py`(패턴 정의)와 `test_*.py`는 문자열 검사에서만 빠진다 |
| (iii) 폴더 밖 쓰기 | `..`, 절대 경로, 심볼릭 링크 탈출을 거부. 편집의 `target_path`는 `exp_dir/edits/` 안으로만 허용. CLI의 `--exp-dir`도 `experiments/harness/` 안으로 제한 |
| (iv) 중단 후 재개 | 호출 상한으로 멈춘 뒤 상한을 올려 재개, 생성 도중 예외로 죽은 뒤 재개, 두 경우 모두 중단 없는 실행과 체크포인트(바퀴, 기준선, 채택 목록), 보고서, 표가 같다. 입력 데이터나 제안 파일이 바뀌면 재개를 거부 |
| (v) 나쁜 제안 | test 읽기 시도 3종, 유료·URL·키·Vertex·requests 4종, 미지의 키, 자기 채택(`status: adopted`)을 모두 무효 처리하고 생성기를 호출하지 않는다 |
| (5) 통계 한계 | 문항 수, 95% 부트스트랩 구간(`kore.evaluation.bootstrap_ci`), 0 포함 시 "이 구간이면 차이를 구분할 수 없다", 점수가 달라진 문항 수, 다중 비교 경고, 검출 가능 최소 차이 근사 |

샘플 데이터 7문항(train 5 + dev 2)으로 CLI를 돌려 봤을 때 가짜 생성기 점수가 모든 편집에서 같았다(샘플이 너무 쉽다). 즉 **루프가 차이를 잡아내는지는 아직 보지 못했다.** 시험은 차이가 없는 경우의 배관과 안전 장치를 확인한 것이다. 샘플 결과는 연구 결과로 인용하지 않는다.

## 아직 미완성인 것

1. **진짜 생성기 연결 없음.** `HFGenerator`(Qwen3-4B, Colab GPU)에 편집된 설정을 넘기는 팩토리를 만들지 않았다. 특히 `prompt_template`은 `kore.generation.build_prompt`가 고정 문자열이라 아직 어디에도 반영되지 않는다. 가짜 생성기에서는 템플릿이 이름에만 들어가고 답은 바뀌지 않는다. 진짜 생성기를 붙이기 전에는 `prompt_template` 편집의 효과를 잴 수 없다.
2. **20문항 dev의 한계는 코드가 알려 줄 뿐 없애지 못한다.** 편집을 여러 개 시험하면서 같은 dev로 고르면 과적합한다. 편집 선택용 세트와 확인용 세트를 나누는 규칙(예: train에서 제안, dev에서 확인)은 정하지 않았다.
3. **제안 자동 생성 없음.** 제안은 사람이나 Claude Code가 JSON으로 쓴다. 이전 교훈을 제안 단계에서 자동으로 보여 주는 일은 같은 편집 반복 거부(`was_rejected`)까지만 했다.
4. 경험 기록이 시험 결과 저장과 한 단계로 묶여 있지 않아서, 저장 직전에 프로세스가 죽으면 경험 줄이 중복될 수 있다(체크포인트와 보고서에는 영향이 없다).
5. 점수는 `rubric_score`(키워드 대용치)이고 의미 평가가 아니다. 생성기가 없으면 `recall`로 바뀐다.
6. 질의 변환 규칙은 리터럴 문자열 연산만 허용한다. 정규식, 임의 코드, 한국어 형태소 처리는 넣지 않았다.

## 공식 8개 구성요소와의 대응

공식 구성요소 이름은 지시문의 목록을 따랐고, 공식 구현의 세부는 확인하지 않았다.

| 공식 구성요소 | KORE에서의 대응 | 상태 |
|---|---|---|
| 시스템 프롬프트 | `prompt_template` 편집(`{question}`, `{evidence}`만 허용) | 형식 검사와 기록만 구현. 생성기에 연결하지 않음 |
| 가드레일 | `harness/safety.py`(test 거부, 금지 내용, 경로 제한, 상한) + `kore.screen`의 분야 게이트는 그대로 둠 | 구현. 가드레일 자체는 편집 대상이 아님 |
| 메모리 | `harness/memory.py`(`ExperienceLog` 재사용 + 교훈 기록) | 구현. 교훈을 읽어 쓰는 쪽은 중복 기각 거부까지만 |
| 도구 안내 | 대응 없음 | 뺌. KORE 생성기는 도구를 호출하지 않는다 |
| 서브에이전트 | 대응 없음 | 뺌. 편집 제안과 분석은 사람과 Claude Code가 맡는다 |
| 스킬 | 대응 없음 | 뺌 |
| 스크립트 | `query_rules`(질의 변환). 코드가 아니라 선언형 규칙으로만 표현 | 구현(리터럴 연산 3종) |
| 플러그인 | 가장 가까운 것은 `selector_weights`와 `k`(선택기 매개변수). 선택기·검색기 교체는 대응시키지 않음 | 일부만 구현 |

## 사용법

```bash
python -m pytest -q
python -m pytest -q experiments/harness
python -m experiments.harness.loop run --knowledge data/sample_knowledge.csv --questions data/sample_eval.csv \
  --splits train dev --exp-dir runs/demo --proposals experiments/harness/proposals/example_proposal.json
python -m experiments.harness.loop approve --knowledge data/sample_knowledge.csv --questions data/sample_eval.csv \
  --splits train dev --exp-dir runs/demo --edit-id ex_k2 --decision rejected --approver 이름 --reason "구간이 0을 포함한다"
```

CLI는 가짜 생성기만 쓴다. 실제 실험은 `HarnessLoop(..., generator_factory=...)`에 `HarnessConfig`를 받는 팩토리를 넘겨서 쓴다. 결과는 `experiments/harness/runs/`(git 무시)에 남는다. 실제 dev 질문으로 돌리기 전에 `python -m kore validate`로 계약을 확인한다.
