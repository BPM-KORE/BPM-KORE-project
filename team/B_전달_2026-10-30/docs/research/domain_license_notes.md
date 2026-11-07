# 외부 분야 데이터셋 이용 검토 - 2026-10-30

- HotpotQA: 공식 배포 홈페이지 https://hotpotqa.github.io/ 및 제작진 GitHub https://github.com/hotpotqa/hotpot 의 README는 데이터셋에 CC BY-SA 4.0을 명시. **코드 라이선스(Apache 2.0)와 데이터 라이선스를 혼동하지 말 것**.
- SciFact: 공식 라이선스 https://github.com/allenai/scifact/blob/master/LICENSE.md — claims_* JSONL의 주장 및 근거 주석 CC BY 4.0, corpus.jsonl 논문 초록 ODC-By 1.0, 코드 Apache 2.0. **단일 라이선스라고 표시하면 오류**. Hugging Face의 별도 배포 페이지 표시와 충돌할 수 있으므로 다운로드 출처를 특정할 것.
- QASPER: https://huggingface.co/datasets/allenai/qasper/blob/main/README.md 에 CC BY 4.0 표기. 논문 전문이 S2ORC에서 추출되었으므로 **개별 논문 전문 저작권 문제는 추가 검토**. 원본 AI2 배포처와 일치 여부 확인 필요.
- 공통: 연구용 데이터로 편입 승인 여부는 팀장(A) 및 담당자(B)가 원문과 조건을 확인한 뒤 정한다. AI가 직접 본 출처 내용과 사용자 확인을 구별해 기록.
