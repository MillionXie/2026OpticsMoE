"""Bounded TRAIN-only phase/response replay; never rewrites a LUT or a model.

Nominal, flat, half-code and global-offset phase patterns are diagnostic stimuli,
not a new deployment encoding. The nominal phase is restored before SDK release.
Two white frames stop the experiment; all captured evidence is retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

HARDWARE = '0c6b75372fb5d2fc51f0cc2ae14eee3b5babac2cb8b62fc8709e9ac3a8417982'
CASES = ((4, '2', '1_binary_repaired_20261020',
          '15d290f1d59ead1a09db37781891a34c5d76943c4a62189ab25a1544c8758fc8'),
         (6, '0', 'router_binary_repaired_20261020',
          '5881bbe7643ab284618b9be8c912b78e297a7c911ef45ba76ed1eba5ee7d302e'))


def phase_patterns(phase):
    import numpy as np
    assert phase.shape == (478, 478) and phase.dtype == np.uint8
    return [('nominal_before', phase.copy()), ('flat', np.zeros_like(phase)),
            ('half_code', (phase.astype(np.uint16) // 2).astype(np.uint8)),
            ('global_offset128', ((phase.astype(np.uint16) + 128) % 256).astype(np.uint8)),
            ('nominal_after', phase.copy())]


def checker_phase(level):
    import numpy as np
    assert level in (0, 64, 128, 192)
    yy, xx = np.indices((478, 478))
    return (((xx // 16 + yy // 16) % 2) * level).astype(np.uint8)


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_paths(cfg, sequence):
    return [sequence / 'calibration/admission/admitted_hardware.json'] + [
        Path(cfg[key]) for key in ('coordinates', 'response_lut', 'phase_lut')]


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import cv2
    import numpy as np
    from PIL import Image
    import lab_caltech20_dvp_layerwise_20261019 as m
    m.torch.set_num_threads(2)
    assert not a.out.exists() and not m.LOCK.exists()
    assert m.session_hardware_digest(a.sequence) == HARDWARE
    cfg = m.read(a.sequence / 'calibration/admission/admitted_hardware.json')
    assert cfg['exposure_us'] == 60006. and cfg['gain'] == 1. and cfg['wait_ms'] == 240
    assert cfg['orientation'] == 'flip_v'
    assert digest_file(cfg['response_lut']) == cfg['response_lut_SHA256']
    assert digest_file(cfg['phase_lut']) == cfg['phase_LUT_SHA256']
    protected = {str(p): digest_file(p) for p in protected_paths(cfg, a.sequence)}
    dm = m.read(a.sequence / 'assets/data_manifest.json')
    lineage = m.read(a.sequence / 'assets/TRAIN_lineage.json')
    assert dm['TRAIN'] == 1600 and dm['TEST'] == 411 and dm['VALID'] == 0
    assert digest_file(a.sequence / 'assets/TRAIN_lineage.json') == dm['lineage_SHA256']
    assert [v['label'] for v in lineage['records'][:20]] == list(range(20))
    assert all(v['kind'] == 'original' for v in lineage['records'][:20])
    cases = []
    for depth, stage, parent, digest in CASES:
        session = a.sequence / f'L{depth}_strict8pp_20261020'
        capture = session / 'capture' / stage
        report = m.read(capture / 'report.json')
        audit = m.read(capture / 'independent_capture_audit.json')
        fit = session / 'fits' / parent
        assert digest_file(fit / 'best.pt') == digest == report['PT_SHA256']
        assert m.read(fit / 'independent_audit.json')['passed'] and audit['passed']
        assert report['data_SHA256'] == dm['sha256'] and report['physical_ccd'] == 2011
        assert digest_file(capture / 'phase.bmp') == report['phase_SHA256']
        # Logical mask is reconstructed from the audited capture-input parent.
        model, _ = m.model_for(fit / 'best.pt', digest, depth, session / 'source_reference.ts')
        # Reproduce physical_fields: angle of the coherent-rho complex plane,
        # including expert tiling, followed by the original floor(phase*256).
        plane = model.plane(stage)
        turns = m.torch.remainder(m.torch.angle(plane), 2*np.pi)/(2*np.pi)
        phase = m.torch.floor(turns*256).clamp(0,255)[20:498,20:498].detach().numpy().astype(np.uint8)
        assert np.array_equal(np.asarray(Image.open(capture / 'phase.bmp')), m.phase_native(phase))
        rows = []
        for i in range(20):
            receipt = m.read(capture / 'train' / f'{i:06d}.json')
            path = capture / 'train' / f'{i:06d}_input.png'
            assert receipt['sample_id'] == lineage['records'][i]['sample_id']
            assert receipt['phase_SHA256'] == report['phase_SHA256']
            assert digest_file(path) == receipt['canonical_SHA256']
            rows.append(dict(index=i, sample_id=receipt['sample_id'], canonical=path,
                             canonical_SHA256=receipt['canonical_SHA256']))
        cases.append((depth, stage, phase, rows, report))
    a.out.mkdir(parents=True)
    protocol = dict(source_git=m.git_head(), source_SHA256=digest_file(__file__),
        hardware_SHA256=HARDWARE, protected=protected, exposure_us=60006., gain=1., wait_ms=240,
        TRAIN_each_depth=20, TRAIN_indices=list(range(20)), original_each_class=1,
        TEST_used=False, TEST_inference=False, VALID=0, LUT_modified=False,
        model_modified=False, diagnostic_phase_only=True, production_phase_encoding_unchanged=True,
        phase_LUT_scope='linear voltage; NOT verified linear phase',
        cases=[dict(depth=d, stage=s, parent_PT_SHA256=r['PT_SHA256'],
                    phase_SHA256=r['phase_SHA256']) for d,s,_,_,r in cases],
        phase_conditions=['nominal_before','flat','half_code','global_offset128','nominal_after'],
        checker_levels=[0,64,128,192], checker_TRAIN_indices=[0,1],
        expected_response_frames=216, white_stop_consecutive=2, automatic_retry=False)
    m.save(a.out / 'protocol.json', protocol)
    warp = cv2.getPerspectiveTransform(np.float32(m.corners(Path(cfg['coordinates']))),
        np.float32([[0,0],[477,0],[477,477],[0,477]]))
    drive = np.asarray(m.read(cfg['response_lut'])['drive'], np.uint8)
    with m.LOCK.open('x') as stream:
        json.dump(dict(pid=os.getpid(), kind='TRAIN_phase_response_verification', out=str(a.out)), stream)
    records=[]; consecutive_white=0; nominal=None; started=time.monotonic()
    try:
        with m.Bench(cfg, a.out) as bench:
            try:
                geometry=m.shared_geometry_check(bench,a.out/'geometry',cfg,warp)
                assert geometry['passed']
                for depth, stage, phase, inputs, previous_report in cases:
                    folder=a.out/f'L{depth}';folder.mkdir()
                    for value in inputs:
                        gray=np.asarray(Image.open(value['canonical']))
                        path=folder/f"input_{value['index']:02d}.bmp"
                        Image.fromarray(m.native(drive[gray],'amplitude')).save(path)
                        value['BMP']=path
                    patterns=phase_patterns(phase)+[(f'checker_{v}',checker_phase(v)) for v in (0,64,128,192)]
                    for tag, pixels in patterns:
                        ph=folder/(tag+'_phase.bmp');Image.fromarray(m.phase_native(pixels)).save(ph)
                        if tag=='nominal_before':nominal=ph
                        bench.stage(ph)
                        for value in inputs if not tag.startswith('checker_') else inputs[:2]:
                            raw,meta=bench.capture(value['BMP'])
                            crop=m.orient(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,
                                borderMode=cv2.BORDER_CONSTANT,borderValue=0),cfg['orientation'])
                            stem=folder/f"{tag}_{value['index']:02d}"
                            rawpath=stem.with_name(stem.name+'_raw.png');croppath=stem.with_suffix('.png')
                            Image.fromarray(raw).save(rawpath,compress_level=1)
                            Image.fromarray(crop).save(croppath,compress_level=1)
                            sat=float((crop==255).mean());consecutive_white=consecutive_white+1 if sat>.99 else 0
                            row=dict(depth=depth,stage=stage,condition=tag,index=value['index'],
                                sample_id=value['sample_id'],canonical_SHA256=value['canonical_SHA256'],
                                input_BMP_SHA256=digest_file(value['BMP']),phase_SHA256=digest_file(ph),
                                raw_SHA256=digest_file(rawpath),crop_SHA256=digest_file(croppath),
                                crop=str(croppath),mean=float(crop.mean()),p99=float(np.percentile(crop,99)),
                                saturation=sat,metadata=meta,source_capture_phase_SHA256=previous_report['phase_SHA256'])
                            m.save(stem.with_suffix('.json'),row);records.append(row)
                            m.save(a.out/'progress.json',dict(state='TRAIN_response_verifying',frames=len(records),
                                expected=216,latest=row,elapsed_s=time.monotonic()-started))
                            assert consecutive_white<2,'Two consecutive whole-crop white frames; evidence retained'
                    bench.stage(nominal)
            finally:
                if nominal is not None:bench.stage(nominal)
        assert len(records)==216
        assert protected=={p:digest_file(p) for p in protected}
        m.save(a.out/'report.json',dict(state='complete',response_frames=len(records),TRAIN_unique=40,
            hardware_unchanged=True,protected=protected,SDK_released=True,TEST_used=False,
            source_git=m.git_head(),source_SHA256=digest_file(__file__),
            settings_unchanged=True,LUT_refitted=False,model_modified=False,records=records,
            elapsed_s=time.monotonic()-started,nominal_phase_restored_before_close=True,
            no_model_accuracy_claim=True,phase_response_not_yet_fitted=True))
    except BaseException as error:
        m.save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            frames=len(records),all_evidence_preserved=True,automatic_retry=False))
        raise
    finally:
        assert m.read(m.LOCK)['pid']==os.getpid()
        m.LOCK.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--sequence',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
