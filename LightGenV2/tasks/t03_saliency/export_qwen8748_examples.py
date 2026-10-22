"""Generate six pinned Qwen baseline prediction arrays for no-GT overlays."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from .export_qwen8748_per_image import PIN, EXAMPLES


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    from PIL import Image
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import SALICONRecord,SALICONSaliencyDataset,collate_salicon
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import SaliencyAccumulator,density_from_logits
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.modeling import preprocess_vision
    from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import FrozenQwenVisionTeacher
    from .aligned_baseline import AlignedReadout
    from .modeling import load_vision_backbone
    from .recheck_aligned import load_hashed_checkpoint
    from .reproduce_baseline import independent_cc
    from .settings import load_settings
    from .run import _seed
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--reference-metrics',type=Path,required=True)
    p.add_argument('--image-source',type=Path,required=True,help='Verified six-image Ours export; only originals and attribution metadata are read')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/qwen_aligned_head_50.yaml')
    a=p.parse_args();out=a.output.resolve();source=a.image_source.resolve()
    if out.exists():raise FileExistsError(out)
    payload,digest=load_hashed_checkpoint(a.baseline_run/'best_checkpoint.pt')
    if digest!=PIN or payload['epoch']!=45 or payload['architecture']!='frozen_qwen24_adapter192_identical_progressive_decoder_v1':
        raise ValueError('Wrong requested baseline')
    reference=read(a.reference_metrics)
    if reference['baseline_checkpoint_sha256']!=PIN:raise ValueError('Wrong metric reference')
    expected={r['image_id']:r for r in reference['six_examples']}
    metadata={r['image_id']:r for r in read(source/'metrics_all.json')['rows']}
    if set(metadata)!=set(EXAMPLES) or set(expected)!=set(EXAMPLES):raise ValueError('Six-image selection mismatch')
    for iid,origin in metadata.items():
        if sha(source/origin['directory']/'original.jpg')!=origin['image_sha256']:
            raise ValueError(f'Reviewed image identity mismatch: {iid}')
    s=load_settings(a.config);s.local_files_only=True;s.download=False
    _seed(42);torch.set_num_threads(4)
    # Use the original baseline manifest rather than scanning all 30,000 target
    # files when only 101 forward inputs and six target maps are needed.
    manifest_path=a.baseline_run/'manifests/samples.csv'
    with manifest_path.open(encoding='utf-8-sig',newline='') as stream:
        records=[SALICONRecord(int(r['sample_index']),r['split'],int(r['image_id']),
                 Path(r['image_path']),Path(r['density_path']),Path(r['fixation_path'])) for r in csv.DictReader(stream)]
    bundle=SimpleNamespace(train_records=[r for r in records if r.split=='train'],
                           validation_records=[r for r in records if r.split=='validation'])
    if [r.sample_id for r in bundle.validation_records]!=[r['sample_id'] for r in reference['rows']]:
        raise ValueError('Original test ordering differs from metric reference')
    loaded=load_vision_backbone(s,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    head=AlignedReadout(s.vision_hidden_size).to(loaded.device);head.load_state_dict(payload['head'],strict=True)
    model=FrozenQwenVisionTeacher(loaded,head).eval();model.requires_grad_(False)
    out.mkdir(parents=True);rows=[]

    def export(records,select_i,context):
        record=records[select_i];iid=record.image_id;origin=metadata[iid]
        reviewed_path=source/origin['directory']/'original.jpg'
        with Image.open(record.image_path) as native, Image.open(reviewed_path) as reviewed:
            native_rgb=np.asarray(native.convert('RGB'));reviewed_rgb=np.asarray(reviewed.convert('RGB'))
            if native_rgb.shape!=reviewed_rgb.shape:raise ValueError('Dataset and reviewed image dimensions differ')
            original_pixel_mae=float(np.abs(native_rgb.astype(np.float32)-reviewed_rgb).mean())
        records=list(records);records[select_i]=replace(record,image_path=reviewed_path)
        dataset=SALICONSaliencyDataset(records,s,training=False)
        batch=collate_salicon([dataset[i] for i in range(len(dataset))])
        inputs=preprocess_vision(loaded.processor,batch['images'],loaded.device)
        all_logits=model(inputs['pixel_values'],inputs['image_grid_thw'])[0]
        logits=all_logits[select_i:select_i+1]
        target=batch['density'][select_i:select_i+1].to(loaded.device)
        fixation=batch['fixation'][select_i:select_i+1].to(loaded.device)
        prediction=density_from_logits(logits)
        acc=SaliencyAccumulator();acc.update(logits,target,fixation)
        metrics=acc.compute();metrics['cc_float64']=float(independent_cc(prediction.cpu().numpy(),target.cpu().numpy())[0])
        delta=max(abs(metrics[k]-expected[iid][k]) for k in ('cc_float64','sim','nss','kld','auc_judd','mae'))
        if original_pixel_mae==0 and delta>2e-4:
            raise ValueError(f'Identical image metrics differ from previous export for {iid}: {delta}')
        origin=metadata[iid];raw=(source/origin['directory']/'original.jpg').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=origin['image_sha256']:raise ValueError('Original JPG identity mismatch')
        folder=out/origin['directory'];folder.mkdir()
        (folder/'original.jpg').write_bytes(raw)
        np.save(folder/'qwen_density_224.npy',prediction.cpu().numpy().squeeze())
        row={k:v for k,v in origin.items() if k not in ('metrics','gt_sha256','fixation_sha256')}
        row.update(model='Qwen baseline 0.8748',metrics=metrics,batch_context=context,
                   model_input_source='reviewed original.jpg',dataset_vs_reviewed_rgb_mae=original_pixel_mae,
                   previous_metrics_max_absolute_difference=delta)
        rows.append(row);(folder/'metrics.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        print('EXPORTED',iid,'CC',metrics['cc_float64'],'max previous metric difference',delta,flush=True)
    try:
        with torch.inference_mode():
            # Preserve the original batch96 context for public-test image 715.
            idx=next(i for i,r in enumerate(bundle.validation_records) if r.image_id==715)
            start=idx//96*96
            export(bundle.validation_records[start:start+96],idx-start,'original test batch96')
            for iid in EXAMPLES[1:]:
                export([next(r for r in bundle.train_records if r.image_id==iid)],0,'single training example, matching prior export')
    finally:model.close()
    (out/'ATTRIBUTION_AND_REVIEW.md').write_bytes((source/'ATTRIBUTION_AND_REVIEW.md').read_bytes())
    report=dict(model='Frozen Qwen baseline, 50-epoch run, best epoch45',checkpoint_sha256=PIN,
        full_test_reference_cc=reference['historical_cc64'],full_test_recent_recheck_cc=reference['means']['cc_float64'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=sha(a.config),reference_metrics_sha256=sha(a.reference_metrics),
        baseline_manifest_sha256=sha(manifest_path),torch=torch.__version__,
        gt_assets_included=False,rows=rows)
    (out/'metrics_all.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={f.relative_to(out).as_posix():sha(f) for f in out.rglob('*') if f.is_file()}
    (out/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('READY_FOR_OVERLAY',out,flush=True)


if __name__=='__main__':main()
