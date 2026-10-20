# 대회 자료 출처

[English](#source-of-the-contest-material)

이 폴더는 제4회 고영 AI 경진대회(ICCAS 2026) 자료입니다. 고영테크놀러지가 제공했으며, 라이선스는 CC BY-NC-SA 4.0입니다. 파일은 바꾸지 않았습니다.

| 경로 | 출처 |
|---|---|
| 그 밖의 모든 파일 (`2_run.py`, `utils/`, `simulation_data/`, `evaluation_data/`의 입력과 `ky_solver/` 출력, `ky_solver/`, 문서 등) | 대회 공식 저장소 [iccas2026-kohyoung/contest](https://github.com/iccas2026-kohyoung/contest) 커밋 `943c885f` (2026-10-20) |
| `evaluation_data/<보드>/ARGOS/output_fov.csv` | 주최 측이 ARGOS 팀 제출 저장소에 돌려준 공식 평가 출력 (커밋 `ce3e4d64`, 2026-10-12) |
| `evaluation_results/EVALUATION_INFO.txt`, `evaluation_results/summary_ARGOS.csv` | 위와 같은 커밋의 평가 요약 |

- 공식 저장소의 `.gitignore`는 이 저장소 규칙과 겹쳐서 옮기지 않았습니다.
- 팀 제출 저장소로 함께 온 `evaluation_data/<보드>/solver.json`은 넣지 않았습니다. 우리 솔버가 만든 파일이 아니며, 다른 참가 팀의 실행 기록으로 보입니다.
- `python 2_run.py --solver ARGOS --data evaluation_data --eval`로 다시 채점하면 합계(ΣCT 7,668.8, Σk×10 5,640)는 공식 결과와 같습니다. 다만 4보드(`S02_dense_fast_008`, `S02_dense_fast_009`, `S03_dense_slow_010`, `S03_dense_slow_011`)는 보드별 CT가 0.02~0.32초 다릅니다. 저장된 출력과 요약이 같은 실행의 기록이 아닌 것으로 보입니다. 큰 보드에서는 병렬 작업자 타이밍 때문에 실행마다 계획이 조금 다를 수 있습니다. 주최 측 파일은 그대로 두었습니다.

권장 사사: 본 연구는 ICCAS 2026과 연계하여 개최된 AI Contest를 위해 고영테크놀러지(Koh Young Technology)가 제공한 AOI 검사 최적화 데이터셋 및 평가 도구를 사용하였으며, 해당 자료는 CC BY-NC-SA 4.0 라이선스로 제공됩니다.

---

# Source of the contest material

This folder holds the material of the 4th Koh Young AI Competition (ICCAS 2026), provided by Koh Young Technology under CC BY-NC-SA 4.0. Files are unchanged.

| Path | Source |
|---|---|
| Every other file (`2_run.py`, `utils/`, `simulation_data/`, the inputs and `ky_solver/` outputs in `evaluation_data/`, `ky_solver/`, documents, …) | official contest repository [iccas2026-kohyoung/contest](https://github.com/iccas2026-kohyoung/contest), commit `943c885f` (2026-10-20) |
| `evaluation_data/<board>/ARGOS/output_fov.csv` | official evaluation outputs the organizers returned to Team ARGOS's submission repository (commit `ce3e4d64`, 2026-10-12) |
| `evaluation_results/EVALUATION_INFO.txt`, `evaluation_results/summary_ARGOS.csv` | evaluation summary from the same commit |

- The official `.gitignore` is not copied because it overlaps the rules of this repository.
- `evaluation_data/<board>/solver.json`, which also arrived in the submission repository, is not included. Our solver did not write it; it appears to be another team's run record.
- Re-scoring with `python 2_run.py --solver ARGOS --data evaluation_data --eval` gives the official totals (ΣCT 7,668.8, Σk×10 5,640). Four boards (`S02_dense_fast_008`, `S02_dense_fast_009`, `S03_dense_slow_010`, `S03_dense_slow_011`) differ by 0.02–0.32 s in per-board CT, so the saved outputs and the summary seem to come from different runs. On large boards, parallel worker timing can change the plan slightly from run to run. The organizers' files are kept as they are.

Recommended acknowledgement: This work used the AOI inspection optimization dataset and evaluation tools provided by Koh Young Technology for the AI Contest held in conjunction with ICCAS 2026. The materials are available under CC BY-NC-SA 4.0.
