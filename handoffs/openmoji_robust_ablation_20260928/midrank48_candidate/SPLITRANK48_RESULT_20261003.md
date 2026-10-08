# Split editor48 / original decoder64: completed, target not met

2026-10-03 04:42 Beijing. Initial architecture fixed before simulation training:
editor rank48, original decoder rank64, four optical fusion gates alpha=.80,
shared head240664 / decoder30162 parameters. No robust tricks, no adaptation-time
extra layer/branch. Initial PT SHA782e608ba914064cd8a10bd0dbc7d3da63dae35a0646cacb893657608954cab8.

| Same original TEST1000 | Changed-cell accuracy |
| --- | ---: |
| Original selected PT, normal CPU simulation | 88.35% |
| Same PT, new six-layer physical deployment | 5.15% |
| Original decoder adaptation, selected e160, strict CPU saved-PT reload | 61.65% |

Target original simulation ×.97 =85.6995%, **not met**. Adapted score is26.70
percentage points below original simulation (relative gap30.22%). Very low direct
score is severe domain failure, not a desirable demonstration or proof of cause.
Simulation of the adapted PT itself has NOT been measured in this round.

Independent original TRAIN1000 and TEST1000 each have6000 new CCD, six layers
1000PNG+1000receipts, zero TRAIN/TEST source or ID overlap. Same phaseSHA,
2000us/GainX4/wait240/ROI/canonical flip_v/bounded no-peak-rescale BMP contract;
all receipt phase/signal audits pass, no old G2/G5 CCD reused. Original raw CCD
and receipts retained in existing bench project:
`E:/code/guest/2026OpticsMoE/OpenMoji_Robust_Rank64_SHS_20261002/runs/splitrank48_test1000`
and `runs/splitrank48_train1000`.

Only original shared_readout.decoder30162 parameters trained160epochs using
TRAIN1000 gradients; user-approved TEST every5epochs highest selection, noVAL
selection, noTEST gradients. This is a TEST-selected DEVELOPMENT result, not
independent generalization. Saved best strictCPU reload passed, full1000samples;
upstream before/after identical:
1b3f0c697875df1d0a26c609550ecd4a93f527f0a7a08c122020a55c1316912e.

Best SHA c494f182e6ddc34760fec0a0a03e85a45d98670d736040225733f3f5c0c9b3be.
Last SHA 98c95da5204bc7df65bf4d3ce566a0eb0aae995064b350e6112d473ee60d0ad3.
Adapter report SHA cf376e729d3d9e758d6d20448def9c36daadf8a3c03cf544c7385ef109154cd5.
Local backup directory `splitrank48_decoder_train1000_testselected_20261003/`:
best/last/report/history/execution/strict_reload/test_samples/progress.
Published source factory/capture17606c3 and adapter6754bad9 remain preserved in
existing task branch/runtime; no source/architecture modification during this run.

Quality tradeoffs: preserved-cell98.3355%→93.5654%, sceneexact0.9%→9.9%.
Adapted changed-cell by operation: add68.4%, replace20.0%, move65.4%, remove92.8%.
Category recovery, especially replacement, remains weak. Do not advertise only
changed-cell recovery while concealing preserved-region loss.

TRAIN changed-cell e50=.634/e100=.726/e160=.797; TEST=.548/.5825/.6165.
Loss4.6769→1.10296 still falls at budget end. Evidence supports incomplete
recovery plus a TRAIN/TEST gap, **not** a proven decoder ceiling or solely a
generalization problem. No unbounded continuation or new architecture silently
started. Further work requires a bounded, justified protocol; any upstream
change requires new physical capture, not cached old CCD reuse.

Task Ready return0, strict_reload PASS; no experiment Python/SDK processes,
bench4060 at319MiB desktop baseline. Original best/last/CCD and all older final
ABO/G5/G2 versions retained. External upload remains paused. This experiment
round is complete despite missing scientific target; periodic monitor ends.
