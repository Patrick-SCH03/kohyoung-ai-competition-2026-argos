# 출처와 파생 부분

Copyright (c) 2026 Team ARGOS. 라이선스: CC BY-NC-SA 4.0 (전문은 [LICENSE](LICENSE)).

## 대회 자료

본 연구는 ICCAS 2026과 연계하여(in conjunction with) 개최된 AI Contest를 위해 고영테크놀러지(Koh Young Technology)가 제공한 AOI 검사 최적화 데이터셋 및 평가 도구를 사용하였으며, 해당 자료는 CC BY-NC-SA 4.0 라이선스로 제공됩니다.

- 대회 공식 저장소: https://github.com/iccas2026-kohyoung/contest
- 이 저장소에는 대회 데이터, 채점기, 실행기, 참조 솔버를 넣지 않았습니다.

## 채점기를 옮긴 부분

아래 파일은 대회 채점기(`utils/checker.py`)의 규칙과 계산을 같은 결과가 나오도록 새로 구현했습니다. 대회 코드를 불러오지 않고 독립적으로 동작합니다.

| 파일 | 옮긴 내용 |
|---|---|
| `ARGOS/ct_sim.py` | 촬영·복원·검사 파이프라인의 CT 계산과 코어 수 선택(min over k of CT(k) + 10k) |
| `ARGOS/acceleration.py` | 위 계산의 컴파일 커널 |
| `ARGOS/release_tools.py` | 코어별 검사 일정 계산 |
| `ARGOS/geometry.py` | 부품이 FOV에 들어가는지 판정하는 규칙(옆면 절반 FOV, 허용 오차) |
| `ARGOS/verify.py` | 커버·순서 규칙 검사 |

그 밖의 탐색·최적화 코드는 팀 ARGOS가 작성했습니다. 파생 부분이 있으므로 저장소 전체를 CC BY-NC-SA 4.0으로 공개합니다.

---

# Attribution and derived parts (English)

Copyright (c) 2026 Team ARGOS. License: CC BY-NC-SA 4.0 (full text in [LICENSE](LICENSE)).

## Contest material

This work uses the AOI inspection optimisation dataset and evaluation tools provided by Koh Young Technology for the AI Contest held in conjunction with ICCAS 2026. These materials are provided under the CC BY-NC-SA 4.0 license.

- Official contest repository: https://github.com/iccas2026-kohyoung/contest
- This repository does not include the contest data, checker, runner or reference solver.

## Parts adapted from the checker

The files below re-implement the rules and computations of the contest checker (`utils/checker.py`) so that they give the same results. They run on their own and do not import contest code.

| File | Adapted content |
|---|---|
| `ARGOS/ct_sim.py` | CT computation of the capture, reconstruction and inspection pipeline, and the core count choice (min over k of CT(k) + 10k) |
| `ARGOS/acceleration.py` | compiled kernels of the above |
| `ARGOS/release_tools.py` | per-core inspection scheduling |
| `ARGOS/geometry.py` | the rule deciding whether a part fits in a FOV (half window for side parts, tolerance) |
| `ARGOS/verify.py` | coverage and order rule checks |

All other search and optimisation code was written by Team ARGOS. Because the repository contains derived parts, all of it is released under CC BY-NC-SA 4.0.
