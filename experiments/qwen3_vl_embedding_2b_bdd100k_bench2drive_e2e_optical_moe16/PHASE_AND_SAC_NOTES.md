# Phase training and closed-loop SAC notes

## Bounded phase is not a whole-network finite-value guarantee

`raw_phase=0` becomes a spatially constant physical phase of `pi` under
`2*pi*sigmoid(raw_phase)`. This is intensity-equivalent to an identity plane,
but `sigmoid(NaN)` is still NaN. A non-finite input adapter, router/OEO tensor,
gradient, Adam moment, or raw phase remains non-finite through phase modulation.

BC therefore separates Actor, CCD recombiner, optical adapter/OEO, physical
phase, and router optimizer groups. Epoch metrics record raw phase magnitude,
physical phase min/max/std, phase delta from initialization, and phase-gradient
RMS/max. Forward checks name the first invalid optical boundary.

The server checkpoints quantify the under-training issue:

| checkpoint | Vision phase std | Language phase std |
|---|---:|---:|
| old Grocery best | 0.0318 rad | 0.0135 rad |
| reproduced Grocery best | 0.2106 rad | 0.1309 rad |
| validated CIFAR best | 0.4494 rad | not applicable |

Thus a finite model can still have a nearly constant and physically ineffective
phase mask. The formal Bench2Drive config now uses independent phase LRs of
`2e-3` in BC stage 1 and `1e-3` in stage 2, while keeping the non-phase optical
electronics at `2e-4` and `1e-4`.

## Python 3.8 / Python 3.11 bridge

CARLA 0.9.15 runs in the dedicated Python-3.8 `RFL` environment. Qwen3-VL and
SAC run in Python 3.11 `xml`. `carla_env_server.py` owns the CARLA vehicle,
synchronous RGB/collision/lane sensors and route state. `carla_bridge.py`
provides the Gymnasium-style proxy over authenticated localhost IPC.

CARLA reserves RPC port `24515` and streaming port `24516`; the bridge uses
`24615`. The one-command lifecycle is documented in `RUN_COMMANDS.md`.

The route environment is for closed-loop SAC training. Official Bench2Drive
scenario/leaderboard evaluation is still a distinct final protocol; route
training reward must not be reported as the official driving score.

