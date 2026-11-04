# 하네스 어댑터 (배관 확인) 2026-11-04

**이 문서는 배관 확인이다. 성능을 확인한 것이 아니다.** 어떤 점수도 연구 결과로 쓰지 않는다.

## 무엇을 연결했나

`experiments/harness/adapter.py`가 편집된 설정(JSON)을 실제 실험 경로로 넘긴다.

| 편집 설정 | 넘어가는 곳 |
|---|---|
| `k`, `candidate_k`, `token_budget`, `weights`, `tokenizer_mode`, `retriever` | `PipelineConfig` -> `kore.pipeline.run_experiment(config=)` |
| `query_rules` | 질문 문자열만 바꾼 사본을 `run_experiment`에 넘긴다(질문 id, 분할, 정답 근거는 그대로) |
| `prompt_template` | `TemplateGenerator`가 틀로 프롬프트를 직접 만들어 모델에 넘긴다 |

원인: `kore.generation.build_prompt`는 고정 함수라서 틀을 받지 못한다. `kore/`는 수정하지 않으므로 `Generator` 프로토콜(`name`, `generate`)을 따르는 감싸는 생성기를 따로 만들었다. 틀이 없거나 근거가 비면(`no_retrieval`) `build_prompt`를 그대로 쓴다. 기본 틀과 `build_prompt`가 글자 단위로 같다는 것을 시험이 확인한다. 생성기 이름에 틀 해시가 붙어서, 틀이 다른 실행이 한 `out_dir`에 섞이면 러너가 막는다.

설정 JSON은 두 모양을 받는다: `HarnessConfig.to_dict()` 모양, 또는 평평한 `{"k":2,"candidate_k":6,"token_budget":300,"prompt_template":"..."}`. 모르는 키와 `{question}`, `{evidence}` 밖의 자리표시자는 오류다.

같은 조건 검사: `check_config_parity`(설정끼리), `check_run_parity`(실행 로그끼리: 같은 질문 집합, 선택 수 <= k, 문맥 토큰 <= 예산, 같은 후보 문서 목록), `assert_parity`. 편집이 일부러 바꾸는 키(예: k 편집이면 `vary=("k",)`)만 허용한다.

## 시험

`python3 -m pytest -q experiments/harness`: 49개 통과(어댑터 시험 12개 포함). `python3 -m pytest -q`: 226개 통과. 어댑터 시험은 틀 편집 하나가 실제 프롬프트와 출력을 바꾸는지, 같은 설정이 같은 프롬프트를 내는지, k와 토큰 예산이 선택에 닿는지를 본다. 가짜 모델은 프롬프트 해시를 답에 붙이므로 이 시험이 통과하는 것은 "프롬프트가 도착했다"는 뜻일 뿐이다.

## GPU에서 처음 확인할 것 (HFGenerator 경로)

`HFTemplateGenerator`는 `HFGenerator`를 상속해 `_prompt`만 바꾼 클래스다. import만 확인했고 인스턴스를 만들거나 실행한 적이 없다(GPU와 torch가 없다). Colab에서 다음을 먼저 본다.

1. 인스턴스 생성: 부모 `__init__`이 `self.template`을 읽기 전에 속성을 세팅하는 순서가 문제없는지.
2. `_prompt`가 채팅 템플릿을 거쳐 기대한 문자열을 내는지(`render_prompt`와 틀 없을 때 같은 문자열인지).
3. 틀을 바꾸면 `max_input_tokens` 초과(`PromptTooLongError`)가 조건마다 다르게 나지 않는지.
4. `answer_logprob`와 `generate`가 같은 `_prompt`를 쓰는지.

## 한계

- 가짜 모델이라 틀 편집이 답 품질에 주는 영향은 전혀 모른다.
- `TemplateGenerator`의 `max_input_tokens` 검사는 `estimate_tokens` 근사치다. 실제 토크나이저 수와 다르다.
- `candidate_k`, 토큰 예산 변경은 설정 변환만 되고, 하네스 루프(`loop.py`)가 이를 막는 정책은 이 어댑터가 바꾸지 않았다. `loop.py`는 아직 이 어댑터를 쓰도록 바꾸지 않았다.
- 틀의 `{evidence}` 형식은 `build_prompt`를 복제한 것이라, `kore/generation.py`가 바뀌면 어긋날 수 있다(기본 틀 동등성 시험이 잡는다).
