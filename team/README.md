# team/: 지용(B)·범규(C) 작업 자료

이 폴더는 B·C가 각자 일을 시작할 때 쓰는 자료다. 처음에 GitHub 저장소 없이 시작할 수 있게 만든 묶음이라, 저장소가 생긴 지금은 아래처럼 읽는다.

## 먼저 읽을 것
- `00_먼저읽기.md`: 순서, 함정, 열 이름, 확인하지 못한 것
- 지용은 `01_지용_안내서.md`, 범규는 `02_범규_안내서.md`
- AI에게 붙일 시작문: `03_AI에게_붙일_시작문_그리고_구글챗_글.txt`

## 00_먼저읽기.md와 저장소가 다른 부분
`00_먼저읽기.md`는 저장소가 없을 때 쓴 문서라서 아래 경로가 다르다.

| 00 문서가 말하는 곳 | 저장소의 실제 위치 |
|---|---|
| `참고자료/docs/` (CONTRACT, DECISIONS, TASKS, START 등) | `docs/` |
| `참고자료/kore/evaluation.py` | `kore/evaluation.py` |
| `참고자료/예시데이터/` | `data/sample_knowledge.csv`, `data/sample_eval.csv` |
| `코드/KORE_코드_8f70421.zip` | 이 저장소 자체 |
| `노트북/03_colab_데이터도구_zip판.ipynb` (GitHub 없이 zip으로 코드를 올리는 판) | 같은 폴더(`team/노트북/`). 저장소에 접근할 수 있으면 원래 판 `notebooks/03_colab_data_tools.ipynb`를 써도 된다 |

## 작업 파일을 놓는 곳
- `작업폴더_지용/`, `작업폴더_범규/` 안의 파일은 비어 있는 틀이다. 채운 결과는 안내서가 정한 경로로 옮긴다: `docs/research/…`, `docs/rubric_scale.md`, `data/evaluation_raw.csv`, `review/gpt_review_log.csv`.
- 틀을 `data/`에 미리 두지 않은 이유: 빈 틀이 점검 도구와 테스트에 걸릴 수 있어서다. 채운 뒤에 옮긴다.
- `작업폴더_지용/raw/`의 안내 파일은 `.gitignore`의 `raw/` 규칙 때문에 GitHub에 올라가지 않는다. 원문 txt는 올리지 않는 것이 규칙이다.
