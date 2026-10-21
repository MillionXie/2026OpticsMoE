"""One L6-only recovery after the human restored the camera USB connection.

Preserve the zero-frame failure, check TRAIN/background in the SAME connection
used for fresh TRAIN1600 -> TEST411, then use the original suffix fitter.
No new mask search, device/LUT adjustment, background subtraction or retry.
"""
import argparse
from pathlib import Path
import sys
import time

CORE_SHA = '977eb00b88d85f1deb31eda557aa9e753fc6b15dfb8cf30d0aa529268c9c10d5'
L6_PT = '2efaf23e81dbba412816a1bbeb4624f159debc3cd7dde534b223088a2ebb8a0b'


def archive_zero_frame_failure(m, folder):
    """Recoverable, exact-child rename; never move valid CCDs or a broad root."""
    folder = Path(folder).resolve()
    source = folder / 'capture'
    target = folder / 'capture_failed_start_20261021'
    assert source.resolve().parent == folder and target.resolve().parent == folder
    assert source.is_dir() and not source.is_symlink() and not target.exists()
    failure = m.read(source / 'failure.json')
    assert failure['completed_receipts'] == 0 and 'DVP Start: -1001' in failure['error']
    assert {p.name for p in source.iterdir()} <= {
        'protocol.json', 'failure.json', 'read_only_device_failure_20261021.json'}
    assert not m.LOCK.exists()
    identities = {p.name: m.sha(p) for p in source.iterdir()}
    source.rename(target)
    assert identities == {p.name: m.sha(p) for p in target.iterdir()}
    return dict(old_failure_preserved=str(target), files_SHA256=identities,
                valid_frames_moved=0, deleted=False)


def repeat_admission(np, correlation, first, repeat):
    """Predeclared within-connection repeat gate, not a classifier score gate."""
    import cv2
    a = cv2.resize(first.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA)
    b = cv2.resize(repeat.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA)
    pcc = correlation(a, b)
    ratio = float(repeat.mean()) / max(float(first.mean()), 1e-12)
    return dict(passed=bool(pcc >= .95 and .8 <= ratio <= 1.2),
                repeat_coarse_PCC=pcc, repeat_mean_ratio=ratio,
                minimum_repeat_PCC=.95, repeat_mean_ratio_interval=[.8, 1.2])


def preflight_bench(m, sequence, data, cfg, output):
    import cv2
    import numpy as np
    from PIL import Image
    nominal = sequence / 'L6_response_20261020/capture/0'
    report = m.read(nominal / 'report.json')
    assert report['physical_ccd'] == 2011 and m.sha(nominal / 'phase.bmp') == report['phase_SHA256']
    lineage = m.read(sequence / 'assets/TRAIN_lineage.json')
    assert [v['label'] for v in lineage['records'][:20]] == list(range(20))
    assert all(v['kind'] == 'original' for v in lineage['records'][:20])
    phase = nominal / 'phase.bmp'
    drive = np.asarray(m.read(cfg['response_lut'])['drive'], np.uint8)
    warp = cv2.getPerspectiveTransform(np.float32(m.corners(Path(cfg['coordinates']))),
        np.float32([[0, 0], [477, 0], [477, 477], [0, 477]]))

    class PreflightBench(m.Bench):
        def __enter__(self):
            super().__enter__()
            try:
                output.mkdir()
                m.save(output / 'protocol.json', dict(source_git=m.git_head(),
                    source_SHA256=m.sha(__file__), TRAIN_indices=list(range(20)),
                    TEST_used=False, VALID=0, formal_frames_reused=0,
                    expected_diagnostic_frames=23, same_connection_as_formal_capture=True,
                    user_report='USB replugged; nearby small ambient light switched on',
                    exposure_coordinates_LUT_unchanged=True, background_subtracted=False,
                    minimum_repeat_PCC=.95, repeat_mean_ratio_interval=[.8, 1.2],
                    previous_vs_current_comparison='descriptive; different capture times, not isolated ambient-light causality'))
                self.stage(phase)
                records=[]; crops=[]; whites=0
                cases=[('background_0',None),('background_1',None)] + [
                    (f'TRAIN_{i:02d}',i) for i in range(20)] + [('TRAIN_repeat_00',0)]
                for tag, index in cases:
                    if index is None:
                        gray=np.zeros((478,478),np.uint8); identity=None
                    else:
                        receipt=m.read(nominal/'train'/f'{index:06d}.json')
                        canonical=nominal/'train'/f'{index:06d}_input.png'
                        assert receipt['sample_id']==str(data['train'][2][index])==lineage['records'][index]['sample_id']
                        assert m.sha(canonical)==receipt['canonical_SHA256'] and receipt['phase_SHA256']==report['phase_SHA256']
                        gray=np.asarray(Image.open(canonical));identity=receipt['sample_id']
                    bmp=output/(tag+'.bmp');Image.fromarray(m.native(drive[gray],'amplitude')).save(bmp)
                    raw,meta=self.capture(bmp)
                    assert abs(meta['exposure_us']-60006)<2 and meta['gain']==1
                    assert not any(meta[k] for k in ('flip_h','flip_v','rotate','opposite'))
                    crop=m.orient(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=0),cfg['orientation'])
                    rawpath=output/(tag+'_raw.png');croppath=output/(tag+'.png')
                    Image.fromarray(raw).save(rawpath,compress_level=1);Image.fromarray(crop).save(croppath,compress_level=1)
                    row=dict(tag=tag,TRAIN_index=index,sample_id=identity,frame_metadata=meta,
                        mean=float(crop.mean()),p99=float(np.percentile(crop,99)),
                        saturation_fraction=float((crop==255).mean()),raw_SHA256=m.sha(rawpath),
                        crop_SHA256=m.sha(croppath),BMP_SHA256=m.sha(bmp),phase_SHA256=m.sha(phase),
                        captured_at_ns=time.time_ns())
                    if index is not None:
                        prior=np.asarray(Image.open(nominal/'train'/f'{index:06d}.png'))
                        row.update(previous_pixel_PCC=m.correlation(prior,crop),previous_mean=float(prior.mean()))
                    records.append(row);crops.append(crop);m.save(output/(tag+'.json'),row)
                    m.save(output/'progress.json',dict(frames=len(records),expected=23,latest=row))
                    whites=whites+1 if row['saturation_fraction']>.99 else 0
                    assert whites<2,'Two consecutive whole-crop white preflight frames; preserve and stop'
                verdict=repeat_admission(np,m.correlation,crops[2],crops[-1])
                m.save(output/'report.json',dict(state='complete',records=records,**verdict,
                    TRAIN_unique=20,TEST_used=False,VALID=0,same_connection=True,
                    no_classifier_accuracy_claim=True,exposure_coordinates_LUT_unchanged=True,
                    ambient_impact_not_assumed_negligible=True,background_subtracted=False))
                assert verdict['passed'],'Same-input TRAIN repeat unstable; no automatic retry or settings change'
                return self
            except BaseException:
                try:self.stage(phase)
                finally:super().__exit__(*sys.exc_info())
                raise
    return PreflightBench


class CaptureAPI:
    """Explicit dependency injection; never monkeypatch a module or algorithm."""
    def __init__(self,base,bench):self.base=base;self.Bench=bench
    def __getattr__(self,name):return getattr(self.base,name)


def run(a):
    import psutil
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    torch.set_num_threads(2);torch.manual_seed(73)
    assert m.sha(loop.__file__)==CORE_SHA
    campaign=a.sequence/'TRAIN_fullbatch_adjoint_sign_20261021'
    returned=m.read(campaign.with_suffix('.return.json'));launch=m.read(campaign.with_suffix('.launch.json'))
    assert returned['returncode']==1 and 'DVP Start: -1001' in returned['error']
    assert not (psutil.pid_exists(launch['pid']) and psutil.Process(launch['pid']).create_time()==launch['created'])
    assert m.read(campaign/'L4_round1/capture/report.json')['physical_ccd']==2011
    assert m.read(campaign/'L4_round1/suffix/report.json')['state']=='complete'
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir()
    m,data,dm,cfg,roots,parents=loop.base_admission(a)
    spec=loop.SPECS[6];session,parent,fields,identities,original=roots[6]
    folder=campaign/'L6_round1';proposal=folder/'proposal';pt=proposal/'candidate.pt'
    assert m.sha(pt)==L6_PT==m.read(proposal/'report.json')['candidate_PT_SHA256']
    model,_=m.model_for(pt,L6_PT,6,session/'source_reference.ts')
    loop.validate_state(m,model,m.state_cpu(original),loop.mask_keys(model,spec['stage']))
    assert not (folder/'suffix').exists()
    config=m.read(m.HERE/'lab_caltech20_DVP_closedloop_profile_20261021.json')
    config.update({k:m.read(m.HERE/'lab_caltech20_DVP_closedloop_sign_profile_20261021.json')[k]
                   for k in ('suffix_epochs','suffix_fit_seconds')})
    loop.gpu_admission(m,config)
    moved=archive_zero_frame_failure(m,folder)
    protocol=dict(method='single_L6_USB_recovery_same_connection_TRAIN_preflight',
        source_git=m.git_head(),source_SHA256=m.sha(__file__),capture_core_SHA256=CORE_SHA,
        original_failure_return_SHA256=m.sha(campaign.with_suffix('.return.json')),
        proposal_PT_SHA256=L6_PT,hardware_SHA256=loop.HARDWARE,parents=parents,
        preserved_failure=moved,TRAIN=1600,TEST=411,VALID=0,diagnostic_TRAIN_frames=23,
        TEST_gradient=False,TEST_informed_development=True,automatic_retry=False,recurring=False,
        L4_recaptured=False,new_mask_search=False,next_layer_started=False,
        user_report='USB replugged; nearby small ambient light switched on')
    m.save(a.out/'protocol.json',protocol)
    previous={k:v for k,v in fields.items() if k!=spec['stage']}
    bench=preflight_bench(m,a.sequence,data,cfg,a.out/'TRAIN_light_preflight')
    newfields=loop.acquire(CaptureAPI(m,bench),a,data,dm,cfg,previous,model,L6_PT,spec['stage'],folder/'capture')
    loop.gpu_admission(m,config)
    fitcfg=dict(config,_reference=str(session/'source_reference.ts'))
    model,correct=loop.refine(m,data,newfields,model,spec,fitcfg,folder/'suffix')
    row=dict(depth=6,state='complete',candidate_correct=correct,global_incumbent_correct=291,
        improved_global=correct>291,target=322,actual_gate_passed=correct>=322,
        new_physical_frames=2011,diagnostic_frames=23,full_physical=False,
        fit_report_SHA256=m.sha(folder/'suffix/report.json'),next_layer_started=False,
        recovery_protocol_SHA256=m.sha(a.out/'protocol.json'))
    m.save(a.out/'report.json',dict(**row,SDK_released=not m.LOCK.exists(),
        original_failure_preserved=True,old_higher_PT_retained=True))
    m.save(campaign/'L6_current.json',row)
    print(m.json.dumps(row),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
