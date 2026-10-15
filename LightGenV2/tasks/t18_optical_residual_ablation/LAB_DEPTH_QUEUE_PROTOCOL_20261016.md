# Accepted L2/L4 one-owner optical queue

Direct human authorization in the coordinating optical-residual chat was verified
2026-10-16. Only this window owns lab queue/capture/readout files. Preserve all L3
TRAIN6088/TEST14664 and running capture. The old from-zero MLP is prohibited;
same simulation-trained8554-parameter head and its frozen TRAIN mean/std required.
Simulation-head source/training is owned by the other chat, not modified here.

Asset selection: accepted_models.csv SHA-bound manifest, never latest-directory
or last-checkpoint selection. L2residual E1143 SHA5c62ae252e3dfd2e4ab752f06546a4698fef8cebf962cce94d667cb0178d5709;
L2control E98 SHA5d0c8c2bee0e48d9d9297d31c68ce2163347e031af764e2c7c5e6d6afcded2d3.
L4residual E488 SHA9b7c135d5f902d542d6d765f96b096f2990cdbc1004bf8ff7d26a5e710a54d27;
L4control E100 SHA693226bbac3fcdc166741a8df18f1005408e08314e384f6bdf7db8ad01ca6f36.
All optical phases frozen; ten original windows/IDs/encoding/route/OEO unchanged.

Reuse unchanged L3 helpers by import, never overwrite the in-use L3 capture
source. Depth constructor alone selects2 or4 main stages; strict complete state
load, config/epoch/SHA checks, original forward and native BMP parity on balanced
ten TRAIN images must pass for all four exact PTs before hardware queue admission.
Those CPU checks are not physical performance. No new optical training/GPU.

Wait for actual L3 TEST complete report and SDK release. Then L2 pilot60 frames
and full15564, L4 pilot100 and full25940, serial one SDK owner. Pilots do not
select PTs or claim TEST accuracy. Within each arm/stage TRAIN761 immediately
followed by TEST1833, then switch layer; complete arm before next arm. No VAL.
Use common approved response621 map/2000us/Gain_X4/wait240/1016 active native,
20261015 corners/hv-inverse phase/flip-v CCD. No exposure/LUT/geometry sweep,
peak stretch or trial retries. p99<15 or saturation>15% stops and retains evidence.
Every raw/crop/input/phase/ID/receipt retained, transient native BMP only removed
after its SHA/canonical input is preserved. CPU/data identity checks before fit.

Queue is capture-only; electronic fits require delivered, SHA-verified simulation
head manifest per depth. Same feature/math/MLP/buffers, optical PTs frozen, no
random fallback, no physical normalization refit. Physical E0 is the already
simulation-trained parent, not random; it may be retained if fine-tuning worsens.
TRAIN gradients only, TEST-best development selection/no VAL. Results and gaps
are measured, not guaranteed; no weakening control or changing sample identities.
Lab only receives Git source and SHA assets; never pushes local lab code upstream.
