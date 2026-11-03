# 4th Koh Young AI Contest — ICCAS 2026, Team ARGOS (3rd place)

[English](README.md) | **한국어**

제4회 고영 AI 경진대회 3위(동상) 수상작. ICCAS 2026(일본 삿포로) 구두 발표. [상장](assets/certificate.jpg)

| 팀원 |
|---|
| [@Patrick-SCH03](https://github.com/Patrick-SCH03) |
| [@xjx06127](https://github.com/xjx06127) |
| [@naeun211](https://github.com/naeun211) |
| [@rosuasu0721-cloud](https://github.com/rosuasu0721-cloud) |
| [@hoon0503](https://github.com/hoon0503) |

## 문제

AOI(Automatic Optical Inspection)는 PCB 위 component를 자동으로 검사하는 장비다.
카메라 헤드가 PCB 위를 이동하며 FOV(Field of View, 한 번에 찍는 직사각형 영역) 단위로 Capture를 하고, FOV마다 3D Recon과 Inspection을 한다.
이 일을 여러 core가 나눠서 동시에 한다.

```
time ─────────────────────────────────────────────▶

Core 0  [cap1] move [cap2] move [cap3]
Core 1        [rec1]                  [rec3]
Core 2                    [rec2]
Core 3              [ins1a]     [ins2a]     [ins3a]
Core 4              [ins1b]     [ins2b]     [ins3b]  ← CT
```

cap = Capture, rec = 3D Recon, ins = Inspection. 숫자는 FOV 번호, a와 b는 그 FOV 안의 component다. 카메라가 다음 FOV로 이동하는 동안 3D Recon이 돌고, Recon이 끝나면 그 FOV의 Inspection이 시작된다.

- **Core 0**: 이동 + Capture
- **Core 1, 2**: 3D Recon (FOV당 고정 시간)
- **Core 3 이상**: component Inspection (component마다 시간이 다르고, 덜 바쁜 core로 간다)
- **CT (Cycle Time)**: 마지막 core가 끝나는 시각

### 정해야 할 것 세 가지

| | 결정 | 출력 |
|---|---|---|
| **STEP 1** | **FOV Placement**: FOV를 어디에, 몇 장 | `x`, `y` |
| **STEP 2** | **Component Assignment**: 어떤 component를 어느 FOV에서 검사할지 | `comp_idx` |
| **STEP 3** | **Inspection Order**: FOV를 어떤 순서로 찍을지 (Fiducial → Barcode → Normal) | 행 순서 |

셋은 서로 얽혀 있어서 NP-hard다.
FOV를 줄이면 Capture가 줄지만 한 FOV에 Inspection이 몰리고, 순서는 이동 시간과 Inspection 시작 시각을 함께 바꾼다.

### Score

낮을수록 좋고, 보드별로 더한다.

```
score = CT(k*) + 10k* + RT                    # RT = solver 실행 시간(초)
k*    = argmin_{k=4..max_core} CT(k) + 10k    # core 수는 checker가 고른다
```

### 규칙

- 모든 component가 FOV 안에 들어가야 한다. `side=1` component는 같은 중심의 **절반 크기 Side FOV** 안에 들어가야 한다.
- FOV보다 큰 component는 여러 FOV로 빈틈없이 덮는다(tiling).
- 한 FOV에 **step**(단차)을 섞을 수 없고, 연달아 찍는 두 FOV의 step이 다르면 **+5초**다.
- 순서는 **Fiducial(단독, 맨 앞) → Barcode → Normal**이다.

## 우리 solver 핵심: 실제 score로만 판정한다

<h3 align="center">Simulation-based solver that judges every plan by the real score</h3>

Fewest FOVs, Shortest path는 좋은 참고일 뿐 목표가 아니다.

- **Shortest path의 함정**: Inspection은 그 FOV의 Capture와 Recon이 끝나야 시작된다. 무거운 FOV를 늦게 찍으면 카메라가 일찍 끝나도 CT가 늘어난다.
- **Fewest FOVs의 한계**: ILP로 FOV 수를 최소로 줄여도 우리 solver에서는 score가 나아지지 않았고, RT만 늘었다.

그래서 **greedy cover와 Shortest path로 계획을 먼저 만들고, 고르고 고치는 건 실제 score로 한다.**
checker와 같은 계산을 하는 simulator를 solver 안에 넣었다. 후보마다 k = 4..max_core의 CT를 전부 계산하고 `min_k CT(k) + 10k`로 비교한다.

### 전체 흐름

```
greedy cover 5개      # 프로세스마다 하나씩
    │
    ▼
order 후보 7개 중 score가 가장 낮은 것
    │
    ▼
center refinement     # FOV 중심을 순서상 앞뒤 FOV 쪽으로 옮기고 order 후보를 다시 비교
    │
    ▼
finish                # critical relocate, heavy reassign, simulated annealing, release
    │
    ▼
polish                # order와 component 배정을 조금씩 바꿔 보기
    │
    ▼
5개 계획 중 score가 가장 낮은 것 → output_fov.csv
```

center refinement부터 polish까지는 score가 나아질 때만 반영한다. 단계별 내용은 STEP 3에 있다.

## STEP 1, 2: FOV Cover

### Valid box

component마다 "FOV 중심이 여기 있으면 component가 다 찍히는" 사각형을 구한다.

```
lx = br_x - W/2,   hx = tl_x + W/2,   ly = br_y - H/2,   hy = tl_y + H/2
W, H = FOV 크기 (side component는 절반)
```

이러면 FOV Placement는 **valid box들을 적은 수의 점으로 모두 찌르는 문제**(rectangle piercing, NP-hard)가 된다. 점 하나가 FOV 하나다. step마다 따로 푼다.

### Weighted greedy

격자점마다 그 점에 FOV를 두면 덮는 component의 weight 합을 구한다. 합이 가장 큰 점에 FOV를 놓고, 덮인 component를 지운다. 모든 component가 덮일 때까지 반복한다.

```
w_i = ( 1 / ((폭_i + r)(높이_i + r)) )^p
폭_i, 높이_i = component i의 valid box 크기 (hx - lx, hy - ly)
r            = 격자 간격
```

- `(폭 + r)(높이 + r) / r²`는 valid box 안의 격자점 수, 즉 그 component를 덮을 수 있는 FOV 중심 후보 수다. 이 수가 적은 component(큰 component, side component)일수록 weight가 크다.
- 이런 component를 뒤로 미루면 종종 그 하나 때문에 FOV를 한 장 더 쓰게 된다.
- `p`가 클수록 이런 component를 먼저 잡는다. `p = 0`이면 기본 greedy다.

어느 보드에 어떤 설정이 맞을지 미리 알 수 없어서 variant 5개를 만든다.

| variant | p | r | side 먼저 |
|---|---|---|---|
| V0 | 1.5 | 0.5 mm | |
| V1 | 1 | 0.5 mm | ✓ |
| V2 | 1 | 1.0 mm | |
| V3 | 0 | 1.0 mm | ✓ |
| V4 | 0 | 0.5 mm | |

### greedy 뒤 정리

- **Redundant removal**: 담은 component가 전부 다른 FOV에도 들어가는 FOV를 지운다.
- **Elimination**: FOV 안의 component를 모두 주변 FOV로 옮길 수 있으면 그 FOV를 없앤다. 받는 FOV는 free box(담은 component의 valid box 교집합) 안에서 중심을 옮길 수 있다. 바로 옮길 수 없으면 주변 FOV가 자기 component 하나를 세 번째 FOV로 넘겨 자리를 만든다(depth-2 ejection chain).
- **Recenter**: 남은 FOV의 중심을 free box 가운데로 옮긴다.

### RWLS

component가 격자처럼 늘어선 보드와 camera-bound 보드(카메라가 병목인 보드)는 RWLS(Row Weighting Local Search)로 FOV 수를 더 줄인 cover를 하나 더 만들어 본다. 이 cover로 order까지 다시 짜서 score가 더 낮을 때만 바꾼다.

## STEP 3: Inspection Order

이동 거리가 아니라 **Inspection이 언제 끝나는지**를 기준으로 짠다.

### 왜 Shortest path가 아닌가

Inspection은 그 FOV의 Capture와 Recon이 끝나야 시작된다. 무거운 FOV를 늦게 찍으면 카메라가 일찍 끝나도 그 Inspection 하나 때문에 모두가 기다리고 CT가 늘어난다. Shortest path가 늘 최선이 아닌 이유다.

### 단계

1. **Blocks**: type(Fiducial / Barcode / Normal)과 step으로 묶고, step이 덜 바뀌도록 block 순서를 정한다.
2. **Block 안 경로**: 이동 시간 기준으로 nearest neighbor → 2-opt → Or-opt.
3. **Candidate orders 7개**를 만들어 score가 가장 낮은 것을 고른다.
   - Shortest path: 2번 경로 그대로
   - Light last: 가장 가벼운 FOV를 마지막에
   - Heaviest first: 가장 무거운 FOV를 맨 앞에
   - Heavy front: 무거운 FOV 몇 개를 앞에
   - Deadline fix: 마지막 Capture까지 못 끝날 Inspection이 있으면 그 FOV를 앞으로
   - Reversed steps: step block 순서 뒤집기
   - Prelude: step change를 감수하고 무거운 FOV 1, 2, 4개를 맨 앞에
4. **Block insertion, center refinement**: 작은 step 구간을 다른 구간 사이에 넣어 보고, FOV 중심을 순서상 앞뒤 FOV 쪽으로 옮겨 이동을 줄인 뒤 후보를 다시 비교한다.
5. **Finish**:
   - critical relocate: 가장 늦게 끝나는 Inspection이 든 FOV를 앞으로
   - heavy reassign: 무거운 component를, 그 component도 덮으면서 더 일찍 찍는 FOV로
   - simulated annealing: order를 섞어 보기
   - release: 가장 늦게 끝나는 Inspection을 풀어 주는 FOV를 같은 step 구간 안에서 앞으로
6. **Polish**: order와, 각 component를 어느 FOV에서 검사할지를 조금씩 바꿔 본다.

4~6번의 모든 변경은 **score가 나아질 때만** 남긴다.

## 결과

공식 evaluation data 96보드.

| | ARGOS | Koh Young baseline |
|---|---:|---:|
| Total score | **13,351.1** | 14,117.0 |
| ΣCT / ΣRT / Σ10k | 7,668.8 / 42.4 / 5,640 | 8,539.2 / 147.8 / 5,430 |
| 평균 FOV 수 | 49.9 | 61.1 |
| 보드당 평균 RT | 0.44초 | 1.54초 |

13팀 중 3위. 96보드 모두 규칙 위반 없이 통과했다.
공식 발표 값이다. 항목마다 따로 반올림돼서 세 값을 더하면 13,351.2가 된다.

## 저장소 구조

```
├── ARGOS/                  # solver (대회에 제출한 그대로)
│   ├── solver.py           #   entry point solve(), 전체 흐름
│   ├── geometry.py         #   입력 읽기, valid box, 큰 component의 tile FOV 위치
│   ├── cover.py            #   weighted greedy, redundant removal, elimination, recenter
│   ├── setcover.py         #   RWLS
│   ├── setcover_py.py      #   setcover.py를 numba 없이 돌리는 버전
│   ├── order.py            #   blocks, block 안 경로, candidate orders 7개
│   ├── refine.py           #   center refinement (이동이 줄도록 FOV 중심 옮기기)
│   ├── assign.py           #   heavy reassign
│   ├── anneal.py           #   simulated annealing, polish
│   ├── polish_nb.py        #   polish를 numba로 빠르게
│   ├── release.py          #   release
│   ├── release_tools.py    #   release.py가 쓰는 Inspection 일정 계산
│   ├── recover.py          #   가까운 FOV 두 장의 component를 다시 나누기
│   ├── ct_sim.py           #   CT simulator (checker와 같은 계산)
│   ├── acceleration.py     #   ct_sim.py를 numba로 빠르게
│   ├── verify.py           #   대회 규칙 검사, 어긋나면 고치기
│   ├── parallel.py         #   cover 5개를 여러 프로세스로 나눠 실행
│   └── atomic_io.py        #   output 저장 (쓰다 끊겨도 깨지지 않게)
├── contest/                # 대회 공식 자료 (원본 그대로)
│   ├── 1_generate_data.py  #   data generator
│   ├── 2_run.py            #   실행, 채점 (--eval: 저장된 output만 다시 채점)
│   ├── 3_view.py           #   viewer (FOV 배치, core별 시간표)
│   ├── utils/              #   checker.py
│   ├── simulation_data/    #   연습 data 96보드
│   ├── evaluation_data/    #   evaluation data 96보드, 보드별 ARGOS와 baseline output
│   ├── evaluation_results/ #   공식 evaluation 결과
│   ├── ky_solver/          #   Koh Young baseline solver (컴파일본)
│   ├── greedy_solver/      #   공식 greedy 예제
│   ├── my_solver/          #   공식 solver 템플릿
│   ├── *.html              #   대회 규칙, 데이터 설명, 가이드, 튜토리얼, Q&A
│   ├── SOURCE.md           #   이 폴더 파일들의 출처
│   └── …                   #   그 밖의 공식 파일
├── README.md               # 영어 README
├── README.ko.md            # 이 문서
├── requirements.txt        # 패키지 버전
├── LICENSE                 # CC BY-NC-SA 4.0
└── NOTICE.md               # 대회 자료 출처, checker를 참고해 다시 구현한 부분
```

## 실행

Python 3.12, 패키지 버전은 `requirements.txt`.

```bash
pip install -r requirements.txt
cp -r ARGOS contest/
cd contest
python 2_run.py --solver ARGOS --data evaluation_data --eval     # 대회 output만 다시 채점
```

다시 풀면 `evaluation_data/<보드>/ARGOS/`의 대회 output을 덮어쓴다. 남겨 두려면 위 채점을 먼저 해 둔다.

```bash
python 2_run.py --solver ARGOS --data evaluation_data            # 풀고 채점
```

## 라이선스

[CC BY-NC-SA 4.0](LICENSE). 출처를 밝히고, 상업적으로 쓰지 않고, 고친 결과물도 같은 조건으로 공개해야 한다.

- `contest/`는 고영테크놀러지가 제공한 대회 자료 원본이다. 출처는 [contest/SOURCE.md](contest/SOURCE.md).
- `ct_sim.py`, `acceleration.py`, `release_tools.py`, `geometry.py`, `verify.py`는 대회 checker의 계산을 참고해 독립적으로 다시 구현했다. 그래서 저장소 전체를 같은 라이선스로 공개한다. 자세한 내용은 [NOTICE.md](NOTICE.md).
- 고영테크놀러지나 대회 주최인 제어로봇시스템학회(ICROS)의 공식 코드가 아니다.
