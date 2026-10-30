# KORE B 담당 — 선행연구 2편 분석 (AI 원문 검토본)

확인일: 2026-10-30  
검토 범위: 두 논문의 초록·서론·방법론·실험 설정·결과·결론 및 관련 부록. 사람이 직접 검토한 것으로 표시하지 않음.  
원본 CSV 열 구조 유지: `paper,question_no,question,answer,evidence_location,read_scope`.

## 1. Source Attribution in Retrieval-Augmented Generation (Nematov et al., 2025)

공식 원문: https://arxiv.org/pdf/2507.04480

- **문제:** RAG 답변을 만드는 데 어떤 문서가 기여했는지 알아내는 데 필요한 다수의 LLM 호출이 비쌈 (§1).
- **자료/모델:** BioASQ·Natural Questions에서 각각 100질문과 질문별 10문서로 평가 (§3.4). LLaMA-3.2-8B-Instruct, Mistral-7B-Instruct, Qwen-3B-Instruct (§3.3).
- **방법:** 모든 문서로 생성된 목표 답변 `R_target`을 기준으로, 부분집합 S에서 그 답변의 *로그우도*를 효용으로 정의. Shapley/LOO/TMC/Beta/Kernel SHAP/ContextCite 비교 (Eq. 2, §3.2).
- **결과:** Kernel SHAP 및 ContextCite의 근사 성능이 좋은 편 (§4.1, Figure 1). 합성 문서 실험에서는 중복 자료 위치 편향, 다단계 추론의 중간 근거 과소평가 등을 관찰 (§4.3, Figures 2–5).
- **KORE와 관계:** 기여도 측정, 상호작용 분석은 중복. 다만 이 논문 자체는 과거 기록을 축적하여 새 질의의 문서 선택 정책을 학습하는 연구가 중심이 아님.

## 2. RepoShapley (Huo et al., Findings of ACL 2026)

공식 논문 정보: https://aclanthology.org/2026.findings-acl.505/  
공식 PDF: https://aclanthology.org/2026.findings-acl.505.pdf

- **문제:** 코드 자동완성에서 다른 파일의 코드를 붙일 때, 하나만 보면 무의미한 코드라도 조합하면 중요할 수 있고, 반대로 그럴듯한 코드가 성능을 해칠 수 있음 (§1).
- **자료/모델:** The Stack에서 정제한 Python 저장소 29만 개의 학습 자료, RepoEval·CrossCodeEval·CrossCodeLongEval 실험 (§4.1). StarCoderBase/CodeLlama 계열; Java·C#·TypeScript 코드 평가도 부록에 있음 (Appendix C.8/Table 10).
- **방법:** 단일 청크 teacher-forced 로그우도 신호 → 상호작용 근사 surrogate → 소규모 조합 Shapley → 실제 생성기 후검증 → KEEP/DROP 및 NEED/DONE 결정을 모델에 지도학습 (§3.3–3.4, Figure 3).
- **결과:** 저자들은 CODEFILTER 등의 기준선보다 더 좋은 코드 완성 지표를 보고 (§4.2/Table 1), 상호작용·후검증을 제거하면 성능이 떨어지는 결과를 제시 (§4.3/Table 2).
- **KORE와 관계:** 문서 **선택 정책 학습**이라는 주장이 이미 기존 논문에 존재. 단순히 Shapley를 써서 자료 조합을 평가하고 불필요 자료를 버리는 연구만으로는 KORE의 독창성을 주장하기 어려움.

## KORE 연구 설계에 대한 제언 (논문의 사실이 아니라 비교를 통한 제안)

1. **검증 가능한 독립 변수:** 유사도 상위 k개 / 무작위 k개 / 단일 문서 기여도 기반 k개 / 문서 조합 기반 k개.
2. **측정 기준:** 한국어 법률 질문 정답률, 답변 근거의 정확성, 근거 누락률, 토큰·호출 비용. 특정 방법의 우수성은 실험 전에는 주장하지 않음.
3. **문제 유형별 분석:** 법률+시행령 복수 근거 필요 / 유사어만 공유하는 혼동 문서 / 상충하거나 중복되는 근거.
4. **독창성 주의:** 법률 도메인과 한국어 사용 자체만으로 방법론의 새로움을 확정하지 말 것. KORE 구현이 무엇인지 서준(A)에게 확인하고 비교 기준을 정할 것.
5. **실험 성능 비교에서 데이터 유출 방지:** 문서 기여도 계산에 정답을 사용했다면 평가 시점에 정답을 미리 사용한 것으로 취급하지 않도록 학습·평가 분리.

## 정정 사항

이전 연구 초안에서 첫 번째 논문의 효용을 '답변 유사도 기반'이라고 설명한 것은 부정확함. 공식 원문 Equation (2)의 정의는 목표 응답 시퀀스 로그우도임. CSV에 정정함.

## 남은 검증

- 사용자가 두 원문의 §3~4 및 주요 표를 직접 확인하는 절차는 미완료.
- KORE의 현재 모델, 실제 점수함수, 선택 정책, 실험 계획은 팀 코드/지침의 최신 상태와 대조해야 함.
- 이 조사만으로 KORE가 새롭다거나 기존 방법보다 우수하다고 주장할 수 없음.
