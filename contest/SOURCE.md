# 대회 자료 출처

[English](#source-of-the-contest-material)

이 폴더는 제4회 고영 AI 경진대회(ICCAS 2026) 자료입니다. 고영테크놀러지가 제공했고, 대회 규칙에 따라 CC BY-NC-SA 4.0을 따릅니다. 파일은 바꾸지 않았습니다.

| 경로 | 출처 |
|---|---|
| `evaluation_data/<보드>/ARGOS/output_fov.csv` | 주최 측이 ARGOS 팀 제출 저장소에 돌려준 공식 평가 출력 (커밋 `ce3e4d64`, 2026-10-12) |
| `evaluation_results/EVALUATION_INFO.txt`, `evaluation_results/summary_ARGOS.csv` | 위와 같은 커밋의 평가 요약 |
| `SOURCE.md` | 팀 ARGOS가 쓴 이 문서 |
| 그 밖의 모든 파일 (`2_run.py`, `utils/`, `simulation_data/`, `evaluation_data/`의 입력과 참조 솔버 출력, `ky_solver/`, 문서 등) | 대회 공식 저장소 [iccas2026-kohyoung/contest](https://github.com/iccas2026-kohyoung/contest) 커밋 `f08ed80a` (2026-10-20) |

- 공식 저장소의 `.gitignore`는 이 저장소 규칙과 겹쳐서 옮기지 않았습니다.
- 팀 제출 저장소로 함께 온 `evaluation_data/<보드>/solver.json`은 넣지 않았습니다. 우리 솔버가 만든 파일이 아니며, 다른 참가 팀의 실행 기록으로 보입니다.
- 참조 솔버 `ky_solver/`는 공식 저장소에 공개된 컴파일본(소스 비공개)을 그대로 실었습니다. 규칙의 라이선스 문장은 "대회가 제공하는 모든 자료"를 대상으로 하지만, 열거 항목(데이터셋, 채점기, 시뮬레이션 도구, 문서)에 참조 솔버를 따로 적지는 않았습니다.
- `python 2_run.py --solver ARGOS --data evaluation_data --eval`로 다시 채점하면 합계(ΣCT 7,668.8, Σk×10 5,640)는 공식 결과와 같습니다. 다만 6보드는 보드별 CT가 공식 요약과 0.01~0.32초 다릅니다(`S02_dense_fast_008`, `S02_dense_fast_009`, `S03_dense_slow_007`, `S03_dense_slow_010`, `S03_dense_slow_011`, `S04_grid_fast_002`). 이 중 `S03_dense_slow_011`은 FOV 수도 79와 80으로 다릅니다. 저장된 출력과 요약이 같은 실행의 기록이 아닌 것으로 보입니다. 큰 보드에서는 병렬 작업자 타이밍 때문에 실행마다 계획이 조금 다를 수 있습니다. 주최 측 파일은 그대로 두었습니다.

권장 사사: 본 연구는 ICCAS 2026과 연계하여 개최된 AI Contest를 위해 고영테크놀러지(Koh Young Technology)가 제공한 AOI 검사 최적화 데이터셋 및 평가 도구를 사용하였으며, 해당 자료는 CC BY-NC-SA 4.0 라이선스로 제공됩니다.

---

# Source of the contest material

This folder holds the material of the 4th Koh Young AI Competition (ICCAS 2026), provided by Koh Young Technology and covered by CC BY-NC-SA 4.0 under the contest rules. Files are unchanged.

| Path | Source |
|---|---|
| `evaluation_data/<board>/ARGOS/output_fov.csv` | official evaluation outputs the organizers returned to Team ARGOS's submission repository (commit `ce3e4d64`, 2026-10-12) |
| `evaluation_results/EVALUATION_INFO.txt`, `evaluation_results/summary_ARGOS.csv` | evaluation summary from the same commit |
| `SOURCE.md` | this document, written by Team ARGOS |
| Every other file (`2_run.py`, `utils/`, `simulation_data/`, the inputs and reference outputs in `evaluation_data/`, `ky_solver/`, documents, …) | official contest repository [iccas2026-kohyoung/contest](https://github.com/iccas2026-kohyoung/contest), commit `f08ed80a` (2026-10-20) |

- The official `.gitignore` is not copied because it overlaps the rules of this repository.
- `evaluation_data/<board>/solver.json`, which also arrived in the submission repository, is not included. Our solver did not write it; it appears to be another team's run record.
- The reference solver `ky_solver/` is the compiled build (source not provided) published in the official repository, included unchanged. The license sentence in the rules covers "all materials provided by this competition", but its list (datasets, checker, simulation tools, documentation) does not name the reference solver separately.
- Re-scoring with `python 2_run.py --solver ARGOS --data evaluation_data --eval` gives the official totals (ΣCT 7,668.8, Σk×10 5,640). Six boards differ from the official summary by 0.01–0.32 s in per-board CT (`S02_dense_fast_008`, `S02_dense_fast_009`, `S03_dense_slow_007`, `S03_dense_slow_010`, `S03_dense_slow_011`, `S04_grid_fast_002`), and `S03_dense_slow_011` also has 80 FOVs instead of 79. The saved outputs and the summary seem to come from different runs; on large boards, parallel worker timing can change the plan slightly from run to run. The organizers' files are kept as they are.

Recommended acknowledgement: This work used the AOI inspection optimization dataset and evaluation tools provided by Koh Young Technology for the AI Contest held in conjunction with ICCAS 2026. The materials are available under CC BY-NC-SA 4.0.
