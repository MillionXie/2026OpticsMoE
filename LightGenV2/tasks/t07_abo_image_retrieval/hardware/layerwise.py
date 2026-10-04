"""Sealed rank72 whole-layer acquisition with explicit external assets.
Defaults to read-only inspect; does not authorize recapture of the sealed run.
The old completed session refuses restart. Use only a separately authorized run.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import time
import numpy as np
from PIL import Image
from . import geometry as flow
from .ccd_store import CHECKPOINT, CCDStore, STAGES

phase_gray = flow.phase_gray


def digest(path):
    return flow.sha(path)


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2),encoding='utf8')


def protocol_rows(protocol,data_root):
    gallery=[r for r in protocol['rows'] if r['split']=='train']
    query=[r for r in protocol['rows'] if r['split']=='query']
    rows=gallery+query
    if len(gallery)!=1600 or len(query)!=800 or len({r['sample_id'] for r in rows})!=2400:
        raise ValueError('Physical gallery/query identity contract changed')
    if len(protocol['rows'])!=2400:raise ValueError('Unexpected protocol split')
    for row in rows:
        if not re.fullmatch('[0-9a-f]{16}',row['sample_id']):raise ValueError('Unsafe sample identity')
        path=(data_root/row['image_path']).resolve()
        if not path.is_relative_to(data_root.resolve()):raise ValueError('Image escapes data root')
        if not path.is_file():raise FileNotFoundError(path)
    return rows


def inspect_paths(args):
    if args.checkpoint_sha256!=CHECKPOINT or digest(args.checkpoint)!=CHECKPOINT:
        raise ValueError('Require the sealed rank72 checkpoint SHA')
    for name in ('protocol','geometry','machine_config','phase_lut'):
        if not getattr(args,name).is_file():raise FileNotFoundError(getattr(args,name))
    for name in ('processor','data_root','phase_sdk'):
        if not getattr(args,name).is_dir():raise FileNotFoundError(getattr(args,name))
    protocol=json.loads(args.protocol.read_text(encoding='utf8'))
    rows=protocol_rows(protocol,args.data_root)
    geometry=json.loads(args.geometry.read_text(encoding='utf8'))
    if geometry['base_corners_screen_TL_TR_BR_BL']!=[[586,147],[1379,159],[1369,946],[573,932]]:
        raise ValueError('Sealed SHS ROI changed')
    if set(geometry['stage_calibration'])!=set(STAGES):raise ValueError('Require exactly six stage orientations')
    for row in geometry['stage_calibration'].values():
        if (row['phase_candidate'],row['camera_orientation'])!=('hv_inverse','flip_v'):
            raise ValueError('Sealed phase/camera orientation changed')
    if (args.run_dir/'report.json').exists() and args.mode!='inspect':
        raise FileExistsError('Complete run exists; refusing restart')
    return {'read_only':True,'checkpoint_sha256':CHECKPOINT,'protocol_sha256':digest(args.protocol),
            'geometry_sha256':digest(args.geometry),'sample_count':len(rows),
            'gallery':1600,'query':800,'mode':args.mode,'model_loaded':False,'devices_opened':False,
            'source_execution_commit':source_execution_commit()}


def source_execution_commit():
    import subprocess
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[4],
                                       stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):return 'unavailable'


@contextmanager
def keep_awake():
    if os.name!='nt':raise RuntimeError('Real capture requires Windows interactive display')
    import ctypes
    power_state=ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    if not power_state:raise OSError('Could not keep optical displays awake')
    try:yield
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


class Pipeline:
    def __init__(self,data_root,inputs,picture,replay):
        self.data_root,self.inputs,self.picture,self.replay=data_root,inputs,picture,replay
        self.STAGE_CALIBRATION={}
    def selected_phase(self,radians,name):return selected_phase(radians,name)
    def evaluate(self,descriptor,rows,protocol,bank):return evaluate(descriptor,rows,protocol,bank)
    def process_batch(self,model,processor,bench,out,phase_paths,rows):
        ids=[r['sample_id'] for r in rows]
        images=[self.picture(self.data_root/r['image_path'],model.metadata.get('input_preprocessing','contain_white')) for r in rows]
        batch=self.inputs(processor,images,torch.device('cpu'))
        def callback(stage,active,ids):
            return self.capture_stage(bench,out,stage,phase_paths[stage],active,ids,self.STAGE_CALIBRATION[stage][1])
        return self.replay(model,batch,ids,callback)

def phase_planes(model):
    result={}
    for mode in ('vision','language'):
        optics=getattr(model,mode).optics
        router=np.zeros((478,478),np.float32)
        router[127:351,127:351]=(2*math.pi*torch.sigmoid(optics.router.raw_router_phase)).cpu().numpy()
        expert=np.zeros((478,478),np.float32)
        for raw,(y,x) in zip(optics.experts,((0,0),(0,254),(254,0),(254,254))):
            expert[y:y+224,x:x+224]=(2*math.pi*torch.sigmoid(raw)).cpu().numpy()
        global_phase=(2*math.pi*torch.sigmoid(optics.global_phase)).cpu().numpy()
        result.update({f'{mode}_router':router,f'{mode}_expert':expert,f'{mode}_global':global_phase})
    return result

def selected_phase(radians, name):
    spatial = name.split('_')[0]
    return phase_gray(radians, spatial, name.endswith('_inverse'))

def evaluate(descriptor, rows, protocol, bank):
    metadata = {r['sample_id']: r for r in protocol['rows']}
    gallery_indices = [i for i, sid in enumerate(bank['ids']) if metadata[sid]['split'] == 'train']
    gallery_ids = [bank['ids'][i] for i in gallery_indices]
    gallery = F.normalize(bank['vectors'][gallery_indices].float(), dim=-1)
    scores = F.normalize(descriptor.float(), dim=-1) @ gallery.T
    order = scores.argsort(dim=1, descending=True)
    predictions = []
    reciprocal = []
    hit1 = hit5 = hit10 = 0
    for row, ranked in zip(rows, order.tolist()):
        products = [metadata[gallery_ids[i]]['product_id'] for i in ranked]
        ranks = [i + 1 for i, product in enumerate(products) if product == row['product_id']]
        rank = ranks[0]
        hit1 += rank <= 1; hit5 += rank <= 5; hit10 += rank <= 10; reciprocal.append(1.0 / rank)
        top = ranked[0]
        predictions.append({
            'sample_id': row['sample_id'], 'product_id': row['product_id'],
            'top1_sample_id': gallery_ids[top], 'top1_product_id': metadata[gallery_ids[top]]['product_id'],
            'rank_of_first_relevant': rank, 'hit_at_1': int(rank == 1),
        })
    n = len(rows)
    return predictions, {
        'recall_at_1': hit1 / n,
        'recall_at_5': hit5 / n,
        'recall_at_10': hit10 / n,
        'mrr': float(np.mean(reciprocal)),
    }

def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint','processor','protocol','data-root','geometry','run-dir','machine-config','phase-sdk','phase-lut'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--mode', choices=('inspect','selftest','pilot','full'), default='inspect')
    args = parser.parse_args()
    inspected = inspect_paths(args)
    if args.mode == 'inspect':
        print(json.dumps(inspected,indent=2))
        return
    global torch, F
    import torch
    from torch.nn import functional as F
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    weight = args.checkpoint
    if digest(weight) != args.checkpoint_sha256:
        raise ValueError('Checkpoint SHA mismatch')
    from ..standalone.model import OpticalRetrieval
    from ..standalone.bounded_export import quantize
    from ..standalone.io import inputs, picture
    from .replay import replay_batch, snapshot_simulation
    from .bench import SHSBench
    pipeline = Pipeline(args.data_root,inputs,picture,replay_batch)

    payload = torch.load(weight, map_location='cpu', weights_only=True)
    model = OpticalRetrieval(payload['metadata'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval().requires_grad_(False).to('cpu')
    processor = AutoProcessor.from_pretrained(str(args.processor), local_files_only=True)
    geometry = json.loads((args.geometry).read_text(encoding='utf8'))
    flow.BASE_CORNERS = np.asarray(geometry['base_corners_screen_TL_TR_BR_BL'], np.float32)
    pipeline.STAGE_CALIBRATION.clear()
    pipeline.STAGE_CALIBRATION.update({stage: (r['phase_candidate'], r['camera_orientation'])
                                       for stage, r in geometry['stage_calibration'].items()})
    protocol = json.loads((args.protocol).read_text(encoding='utf8'))
    gallery = [r for r in protocol['rows'] if r['split'] == 'train']
    query = [r for r in protocol['rows'] if r['split'] == 'query']
    all_rows = gallery + query
    if len(gallery) != 1600 or len(query) != 800 or len({r['sample_id'] for r in all_rows}) != 2400:
        raise ValueError('Physical gallery/query identity contract changed')
    rows = gallery[:4] if args.mode != 'full' else all_rows
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'report.json'
    if report_path.exists():
        raise FileExistsError('Complete run exists; refusing restart')

    phase_dir = out / 'phase'
    phase_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        phases = phase_planes(model)
    phase_paths = {}
    for stage, radians in phases.items():
        path = phase_dir / (stage + '.bmp')
        expected = pipeline.selected_phase(radians, pipeline.STAGE_CALIBRATION[stage][0])
        if path.exists():
            if not np.array_equal(np.array(Image.open(path)), expected):
                raise ValueError(f'Existing phase changed: {stage}')
        else:
            Image.fromarray(expected).save(path)
        phase_paths[stage] = path
    if set(phase_paths) != set(STAGES):
        raise ValueError('Expected exactly six phase planes')
    contract = dict(checkpoint_sha256=args.checkpoint_sha256, phase_sha256={
        stage: digest(path) for stage, path in phase_paths.items()},
        encoding='bounded tanh(abs/.5) before phase; round255a without peak scaling',
        exposure_us=400, gain='Gain_X4', wait_ms=240,
        geometry=geometry['base_corners_screen_TL_TR_BR_BL'],
        stage_orientation=pipeline.STAGE_CALIBRATION,
        ids=[r['sample_id'] for r in rows])
    cp = out / 'contract.json'
    normalized = json.loads(json.dumps(contract))
    if cp.exists():
        if json.loads(cp.read_text(encoding='utf8')) != normalized:
            raise ValueError('Resume contract changed')
    else:
        write_json(cp, contract)

    store = CCDStore(out,contract_sha256=digest(cp))
    counts = {stage: 0 for stage in STAGES}
    current = [None]
    warmed = set()
    started = time.perf_counter()

    def read(stage, ids):
        return store.read(stage,ids)

    class EndCurrentLayer(Exception):
        pass

    def capture(bench, unused, stage, phase_path, active, ids, orientation):
        if STAGES.index(stage) > STAGES.index(current[0]):
            raise EndCurrentLayer()
        folder = out / 'ccd' / stage
        missing = []
        for j, sid in enumerate(ids):
            image_exists = (folder / (sid + '.png')).is_file()
            receipt_exists = (folder / (sid + '.json')).is_file()
            if image_exists != receipt_exists:
                raise RuntimeError(f'Partial CCD pair needs audit: {stage}/{sid}')
            if not image_exists:
                missing.append(j)
        if stage == current[0] and missing:
            selected = active[missing]
            amp_dir = out / 'amplitude' / stage
            amp_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            selected_ids = [ids[j] for j in missing]
            for sid, bitmap in zip(selected_ids, quantize(selected)):
                path = amp_dir / (sid + '.bmp')
                Image.fromarray(flow.active_to_native(bitmap)).save(path)
                paths.append(path)
            if stage not in warmed:
                for rep in range(2):
                    bench.capture('warmup_' + stage, phase_path, [paths[0]],
                                  [selected_ids[0] + f'_warmup{rep}'], orientation, save=True)
                warmed.add(stage)
            bench.capture(stage, phase_path, paths, selected_ids, orientation, save=True)
            for path in paths:
                path.unlink()
        raw = read(stage, ids)
        return torch.from_numpy(raw).to('cpu').float(), {'reused_or_captured': True}, 1.

    first = rows[:4]
    images = [picture(args.data_root / row['image_path'],
                           model.metadata.get('input_preprocessing', 'contain_white')) for row in first]
    batch = inputs(processor, images, torch.device('cpu'))
    model.vision.optics.router.measured_ccd = None
    model.language.optics.router.measured_ccd = None
    with torch.inference_mode():
        ideal = snapshot_simulation(model, batch)

    def ideal_capture(bench, output, stage, phase, active, ids, orientation):
        return ideal[stage].to('cpu').float(), {}, 1.

    pipeline.capture_stage = ideal_capture
    with torch.inference_mode():
        replay = pipeline.process_batch(model, processor, None, out, phase_paths, first)
    error = float((replay['descriptor'] - ideal['descriptor']).abs().max())
    write_json(out / 'bridge_selftest.json', dict(maximum_descriptor_error=error, samples=4, camera_used=False))
    if error >= 1e-5:
        raise RuntimeError('Ideal six-stage bridge mismatch')
    if args.mode == 'selftest':
        print(json.dumps(dict(status='selftest_complete', bridge_max_error=error)), flush=True)
        return
    pipeline.capture_stage = capture
    chunks = out / 'features'
    chunks.mkdir(exist_ok=True)
    class StageBench:
        """Open SDKs only when a stage actually needs new CCD, after replay."""
        def __init__(self):
            self.inner = None

        def __enter__(self):
            return self

        def release(self):
            if self.inner is not None:
                inner, self.inner = self.inner, None
                inner.__exit__(None, None, None)

        def __exit__(self, *exc):
            if self.inner is not None:
                inner, self.inner = self.inner, None
                inner.__exit__(*exc)

        def capture(self, *capture_args, **capture_kwargs):
            if self.inner is None:
                self.inner = SHSBench(out, 400, 240, phase_paths,
                                      machine_config=args.machine_config,phase_sdk=args.phase_sdk,phase_lut=args.phase_lut)
                self.inner.__enter__()
                print(json.dumps({'hardware_session': 'opened', 'stage': current[0]}), flush=True)
            return self.inner.capture(*capture_args, **capture_kwargs)

    with keep_awake(), StageBench() as bench:
        for stage in STAGES:
            bench.release()
            current[0] = stage
            for index in range(0, len(rows), 4):
                sample = rows[index:index + 4]
                sample_ids = [row['sample_id'] for row in sample]
                folder = out / 'ccd' / stage
                complete_pairs = all((folder / (sid + '.png')).is_file()
                                     and (folder / (sid + '.json')).is_file()
                                     for sid in sample_ids)
                if stage != STAGES[-1] and complete_pairs:
                    # These completed CCD are inputs to later layers. Validate
                    # their contract/signal directly; replaying the model here
                    # would not produce or validate any new physical data.
                    read(stage, sample_ids)
                    result = None
                else:
                    with torch.inference_mode():
                        try:
                            result = pipeline.process_batch(model, processor, bench, out, phase_paths, sample)
                        except EndCurrentLayer:
                            result = None
                if stage == STAGES[-1]:
                    if result is None:
                        raise RuntimeError('Last layer did not yield descriptors')
                    torch.save(result, chunks / f'{index:06d}.pt')
                counts[stage] = index + len(sample)
                progress = dict(status='capturing', stage=stage, stage_completed=counts[stage],
                                stage_total=len(rows), ccd_counts=counts, total_ccd=sum(counts.values()),
                                checkpoint_sha256=args.checkpoint_sha256, elapsed_seconds=time.perf_counter()-started)
                write_json(out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    if args.mode == 'pilot':
        report = dict(status='complete', mode='pilot', checkpoint_sha256=args.checkpoint_sha256,
                      samples=len(rows), ccd_counts=counts, total_ccd=sum(counts.values()),
                      phase_sha256=contract['phase_sha256'], bridge_max_error=error)
        write_json(report_path, report)
        write_json(out / 'progress.json', report)
        print(json.dumps(report), flush=True)
        return
    vectors, ids = [], []
    for index in range(0, len(rows), 4):
        result = torch.load(chunks / f'{index:06d}.pt', map_location='cpu', weights_only=True)
        vectors.append(result['descriptor'])
        ids.extend(result['ids'])
    vectors = torch.cat(vectors)
    if ids != [r['sample_id'] for r in rows]:
        raise ValueError('Descriptor identities changed')
    bank = {'ids': ids, 'vectors': vectors}
    torch.save(bank, out / 'physical_features.pt')
    predictions, metrics = pipeline.evaluate(vectors[len(gallery):], query, protocol, bank)
    report = dict(status='complete', checkpoint_sha256=args.checkpoint_sha256,
                  samples=2400, gallery=1600, query=800, ccd_counts=counts,
                  phase_sha256=contract['phase_sha256'], physical_to_physical=True,
                  metrics=metrics, predictions=predictions, elapsed_seconds=time.perf_counter()-started)
    write_json(report_path, report)
    write_json(out / 'progress.json', dict(status='complete', metrics=metrics, ccd_counts=counts))
    print(json.dumps(dict(status='complete', metrics=metrics)), flush=True)

if __name__=='__main__':main()
