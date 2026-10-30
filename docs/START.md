# 처음 시작하기: GitHub에 올리고 Colab에서 돌리기

위에서부터 순서대로 따라 한다. 돈이 드는 단계는 없다. 막히면 그 단계 번호와 화면에 나온 오류를 그대로 이슈나 단톡에 붙인다.

| 단계 | 누가 | 걸리는 시간(어림) |
|---|---|---|
| 1. 계정 준비 | 셋 다 | 10분 |
| 2. 저장소 올리기 | A | 15~30분 |
| 3. B, C 초대 | A, 그리고 B·C가 수락 | 10분 |
| 4. Colab용 읽기 토큰 | 셋 다 각자 | 10분 |
| 5. Colab에서 첫 실행 | A | 20~40분(모델 다운로드 포함) |
| 6. B·C의 데이터 올리기 | B, C | 처음 30분 |

## 1. 계정 준비 (셋 다)

1. GitHub 계정을 각자 만든다(https://github.com/signup). 한 계정을 나눠 쓰지 않는다.
2. 이메일 인증을 끝낸다. 인증이 안 되면 4단계 토큰을 만들 수 없다.
3. Colab에 쓸 Google 계정을 정한다. 각자 자기 계정을 쓴다.
4. 자기 GitHub 사용자 이름을 A에게 알려 준다.

## 2. 저장소 올리기 (A)

### 먼저 정할 것 두 가지

**(1) 저장소 주인: A 개인 계정이 아니라 무료 조직(organization)으로 만든다.**
개인 계정 저장소에 초대된 사람(B, C)은 그 저장소용 읽기 토큰(4단계)을 직접 만들 수 없다. 그러면 B와 C는 Colab에서 저장소를 받지 못한다.
조직 저장소는 조직 멤버가 각자 자기 토큰을 만들 수 있다. 조직은 무료 플랜으로 만들 수 있다.

1. GitHub 오른쪽 위 프로필 사진 → **Settings** → 왼쪽 **Access** 아래 **Organizations** → **New organization** → **Free** 플랜.
2. 조직 이름 예: `BPM-KORE`(아무도 안 쓰는 이름이어야 한다). 연락 이메일은 A의 이메일. "My personal account"를 고른다.

**(2) 비공개(private)로 시작한다.**

| | 비공개(private) | 공개(public) |
|---|---|---|
| 남이 보나 | 멤버만 | 누구나 |
| `main` 보호 규칙(PR 없이 못 올리게 막기) | **무료 플랜에서는 못 쓴다** | 쓸 수 있다 |
| GitHub Actions(CI) | 월 2,000분 무료. 다 쓰면 결제수단이 없을 때 멈춘다(돈이 나가지 않는다) | 무료 |
| 아직 정리 안 된 데이터, test 질문 | 숨겨진다 | 다 보인다 |

test 질문이 미리 공개되면 "공개된 질문을 모델이 학습했을 수 있다"는 문제가 생기고, 라이선스 확인이 덜 된 문서가 퍼질 수 있다.
공개는 데이터 정리가 끝난 뒤 따로 정한다(`DECISIONS.md` 11절).
비공개에서는 "`main`에 직접 올리지 않는다"를 기계가 막아 주지 않는다. 셋이 약속으로 지킨다.
CI 시간 어림: 테스트 한 번은 1분 안쪽이고 파이썬 버전 여러 개를 돌리므로 푸시 한 번에 몇 분이다. 하루 몇 번 푸시하는 정도면 2,000분 안에 들어간다(어림이고 재 보지 않았다).

### 방법 1: 웹에서 끌어다 놓기 (Git을 몰라도 된다)

1. 받은 `kore-project.zip`을 푼다.
2. https://github.com/new 에서 Owner를 **방금 만든 조직**으로 바꾸고, Repository name에 `kore`, **Private** 선택, "Add a README file"은 **체크하지 않는다**. Create repository.
3. 새 저장소 화면에서 "uploading an existing file" 링크를 누른다.
4. 푼 폴더 **안의 내용물 전체**를 브라우저로 끌어다 놓는다(`kore`, `docs`, `README.md` 등이 바로 보이는 단계).
   한 번에 100개까지, 파일 하나 25MiB까지 올라간다. 이 저장소는 70개 안팎이라 한 번에 된다.
5. 아래 Commit changes를 누른다.
6. **확인**: 저장소 첫 화면에 `.github` 폴더와 `.gitignore`가 보이는지 본다. macOS Finder는 점(.)으로 시작하는 파일을 숨겨서 빠질 수 있다. 빠졌으면 Finder에서 `Cmd + Shift + .`로 숨긴 파일을 보이게 하고 그 둘만 다시 올린다.

### 방법 2: 명령어로 올리기 (A가 Claude Code를 쓰면 이쪽이 편하다)

Git과 GitHub CLI(`gh`)를 설치한다.

```bash
# Windows (PowerShell)
winget install --id Git.Git -e --source winget
winget install --id GitHub.cli --source winget

# macOS (Homebrew가 있을 때)
brew install git gh
```

설치 뒤 터미널을 새로 열고, 푼 폴더로 가서(`BPM-KORE`은 1단계에서 만든 조직 이름으로 바꾼다):

```bash
cd kore-project
git config --global user.name "내 이름"
git config --global user.email "GitHub에 등록한 이메일"
gh auth login                      # GitHub.com → HTTPS → 브라우저로 로그인 을 고른다
git init -b main
git add .
git commit -m "KORE 초기 코드"
gh repo create BPM-KORE/kore --private --source=. --remote=origin --push
```

마지막 줄이 저장소를 만들고 올리기까지 한다. 끝나면 `gh repo view --web`으로 브라우저에서 확인한다.

### 올리면 안 되는 것 (두 방법 모두)

`.gitignore`가 아래를 막아 준다. 방법 1(웹)은 `.gitignore`를 보지 않으므로 직접 빼고 올린다.

- `raw/` 폴더(원문 txt). 이용 조건이 불분명할 수 있다.
- `results/` 아래의 `blind_key.csv`. 채점이 끝난 뒤 `git add -f`로 올린다.
- `.env`, 토큰, 비밀번호가 든 파일.

## 3. B, C 초대 (A)

1. 조직 페이지(`https://github.com/BPM-KORE`) → **People** → **Invite member** → B, C의 GitHub 사용자 이름 → 역할 **Member**.
2. B, C는 메일로 온 초대를 **수락**한다.
3. 쓰기 권한 주기: 저장소 `kore` → **Settings** → **Collaborators and teams** → **Add people** → B, C를 넣고 권한 **Write**.
   (조직 멤버의 기본 권한은 저장소를 읽기만 할 수 있게 되어 있을 수 있다. Write를 줘야 B, C가 PR 브랜치를 만들 수 있다.)

## 4. Colab용 읽기 토큰 (셋 다 각자)

비공개 저장소는 Colab이 그냥 받을 수 없어서 "읽기만 되는 열쇠"를 만든다. 토큰은 비밀번호와 같다. 채팅, 파일, 노트북 코드에 붙이지 않고, 남의 토큰을 빌려 쓰지 않는다.

1. GitHub 오른쪽 위 프로필 사진 → **Settings** → 왼쪽 맨 아래 **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**.
2. Token name: `kore-colab-read`. Expiration: 90일 정도.
3. **Resource owner: 조직(`BPM-KORE`)**. Repository access: **Only select repositories** → `kore`.
4. Permissions → Repository permissions → **Contents: Read-only**. 다른 것은 건드리지 않는다.
5. Generate token → 나온 값을 복사한다(이 화면을 닫으면 다시 볼 수 없다).
6. B, C의 토큰은 처음에 **승인 대기(pending)**로 나온다(조직의 기본 설정이 관리자 승인이다). A가 조직 **Settings** → 왼쪽 **Personal access tokens** 아래 승인 요청 목록에서 승인한다. A 자신의 토큰은 승인이 필요 없다.
7. Colab에서 노트북을 열고 왼쪽의 **열쇠 아이콘(Secrets)** → **새 보안 비밀 추가** → 이름 `GITHUB_TOKEN`, 값에 붙여 넣기 → **노트북 액세스**를 켠다.
8. 같은 곳에 하나 더: 이름 `KORE_REPO_URL`, 값 `https://github.com/BPM-KORE/kore.git`(내 조직 이름). 이걸 넣어 두면 노트북 02, 03의 셀마다 `REPO_URL`을 고칠 필요가 없다. 노트북 01은 셀이 하나라 그 셀의 `REPO_URL`만 고친다.

토큰이 안 되면(오류 403/404): 노트북 셀의 `REPO_URL`이 `https://github.com/BPM-KORE/kore.git`처럼 조직 이름으로 되어 있는지, 토큰이 승인됐는지, Secrets의 노트북 액세스가 켜져 있는지 순서대로 본다.

## 5. Colab에서 첫 실행 (A)

1. GitHub 저장소에서 `notebooks/01_colab_quickstart.ipynb`를 연다 → 파일 위의 **Open in Colab** 버튼이 없으면 https://colab.research.google.com → **GitHub** 탭 → 저장소 주소를 넣고 "비공개 저장소 포함"을 체크해서 연다.
2. 첫 코드 셀 맨 위 `REPO_URL`을 `https://github.com/BPM-KORE/kore.git`(내 조직 이름)으로 바꾸고 실행. 테스트가 전부 통과해야 한다. **G0 통과.**
3. `notebooks/02_colab_gpu_run.ipynb`를 같은 방법으로 연다. 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU**.
4. 4단계 8번의 `KORE_REPO_URL`을 저장했으면 손댈 것이 없다. 안 했으면 셀마다 `REPO_URL`을 바꾼다.
5. 1번(준비) → 2번(GPU 점검) 순서로 실행. 2번의 마지막 줄 "판정: 통과"를 확인한다. **G1 통과.**
   - "문제 있음"이면 2번 셀의 `CONFIG`를 `configs/generator_fallback_qwen25_3b.json`으로 바꿔 다시 실행한다.
   - GPU가 안 잡히면 시간을 두고 다시 연결한다. 무료 Colab은 GPU를 보장하지 않는다.
6. 2번 셀 출력 전체를 복사해서 이슈에 붙인다(모델 이름, 걸린 시간, 판정).

데이터가 아직 없으면 노트북이 샘플로 대신 돈다. 그 숫자는 배관 점검용이고 연구 결과가 아니다.

## 6. B·C의 데이터 올리기

B와 C는 파이썬을 설치하지 않아도 된다. `notebooks/03_colab_data_tools.ipynb`(GPU 필요 없음)에서 명령을 돌리고, 결과 파일을 GitHub 웹에서 올린다.

1. Colab에서 노트북 03을 열고 1번 셀을 실행한다(최신 `main`을 받는다). 4단계의 `GITHUB_TOKEN`, `KORE_REPO_URL`이 내 Colab Secrets에 있어야 한다.
2. 2번 셀로 내 파일을 올린다. B는 원문 txt, C는 `evaluation_raw.csv`.
3. B는 3-B 셀(`DRY_RUN = True`로 먼저 보고 → `False`로 다시), C는 3-C 셀을 실행한다. `[실패] (필수)`가 남으면 고친다.
4. 4번 셀로 결과 CSV를 내려받는다(`WHO`를 B 또는 C로).
5. GitHub 저장소 → `data` 폴더로 들어가서 **Add file → Upload files** → 내려받은 CSV를 끌어다 놓는다.
6. 아래에서 **Create a new branch for this commit and start a pull request**를 고르고 브랜치 이름을 `data/b-1024`처럼 적는다 → **Propose changes** → **Create pull request**.
7. PR 화면에서 CI(초록 체크)가 끝나기를 기다린다. 빨간 X면 눌러서 오류 줄을 보고 고친다. 초록이면 다른 한 명이 읽고 **Merge**.

두 사람이 같은 CSV를 동시에 고치면 나중 PR이 충돌한다. B는 `knowledge.csv`, `source_log.csv`만, C는 `evaluation_raw.csv`, `evaluation.csv`만 올린다. 올리기 전에 노트북 03의 1번 셀을 다시 돌려서 최신 `main`에서 시작한다.

## 7. 다음

여기까지 되면 `docs/TASKS.md` 1절의 나머지(문서 30개, 질문 21개)와 2절(분야 점검)로 간다.
노트북 02의 **2-1. 검색기 비교** 셀은 데이터가 20개 이상 쌓였을 때 처음 돌린다(정답 근거가 후보에 들어오는지 본다).
