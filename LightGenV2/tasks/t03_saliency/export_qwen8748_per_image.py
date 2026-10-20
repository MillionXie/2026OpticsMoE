"""Export plotting metrics for the pinned 50-epoch frozen-Qwen baseline.

Reuses the original batch-96 forward and SALICON metric implementation. Never
trains, selects samples, or exports ground-truth maps or predictions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

PIN = '689d086b5a390c399df109d07b73185575cd2245998b7bd427f6ccea5b0fe019'
IDS = '625dec6bc15b2d737d39bc252cfa0c354de217fec0266dcda568913f4a3496d0'
METRICS = ('cc', 'kld', 'sim', 'nss', 'auc_judd', 'mae')
EXAMPLES = (715, 508, 450, 6730, 292271, 562382)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    import numpy as np
    import torch
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import prepare_salicon, SALICONSaliencyDataset, collate_salicon
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import SaliencyAccumulator,density_from_logits
    from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import FrozenQwenVisionTeacher
    from .aligned_baseline import AlignedReadout
    from .modeling import load_vision_backbone
    from .recheck_aligned import recheck_loader,load_hashed_checkpoint
    from .reproduce_baseline import independent_cc
    from .settings import load_settings
    from .run import _seed
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.modeling import preprocess_vision
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/qwen_aligned_head_50.yaml')
    p.add_argument('--ours-run',type=Path)
    p.add_argument('--map-cache',type=Path)
    a=p.parse_args();out=a.output.resolve();run=a.baseline_run.resolve()
    if out.exists():raise FileExistsError(out)
    original=run/'independent_test_recheck_original_batch'
    reference=read(original/'reproduction.json')
    checkpoint=run/'best_checkpoint.pt'
    payload,digest=load_hashed_checkpoint(checkpoint)
    if digest!=PIN or reference['checkpoint_sha256']!=PIN or payload['epoch']!=45:
        raise ValueError('Not the requested 0.8748 baseline checkpoint')
    if payload['architecture']!='frozen_qwen24_adapter192_identical_progressive_decoder_v1':
        raise ValueError('Wrong Qwen readout architecture')
    with (original/'per_image_cc.csv').open(encoding='utf-8-sig',newline='') as f:
        original_rows=list(csv.DictReader(f))
    s=load_settings(a.config);s.output_dir=out
    s.local_files_only=True;s.download=False;s.inference_batch_size=96;s.num_workers=2
    if a.map_cache:s.artifact_cache_dir=a.map_cache.resolve()
    _seed(42);torch.set_num_threads(4)
    bundle=prepare_salicon(s,persist=False)
    loader,expected=recheck_loader(bundle,s,'test')
    out.mkdir(parents=True)
    loaded=load_vision_backbone(s,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    head=AlignedReadout(s.vision_hidden_size).to(loaded.device)
    head.load_state_dict(payload['head'],strict=True)
    model=FrozenQwenVisionTeacher(loaded,head).eval()
    model.requires_grad_(False)
    accumulator=SaliencyAccumulator();rows=[];examples=[]

    def process(batch, table, scope):
        inputs=preprocess_vision(loaded.processor,batch['images'],loaded.device)
        logits=model(inputs['pixel_values'],inputs['image_grid_thw'])[0]
        target=batch['density'].to(loaded.device);fixation=batch['fixation'].to(loaded.device)
        cc64=independent_cc(density_from_logits(logits).cpu().numpy(),target.cpu().numpy())
        if scope=='public_test':accumulator.update(logits,target,fixation)
        for i,sid in enumerate(batch['sample_ids']):
            single=SaliencyAccumulator();single.update(logits[i:i+1],target[i:i+1],fixation[i:i+1])
            values=single.compute();values.pop('samples')
            values['cc_float64']=float(cc64[i])
            table.append(dict(sample_id=sid,image_id=int(batch['image_ids'][i]),
                file_name=Path(batch['image_paths'][i]).name,split=sid.split('/')[0],
                scope=scope,model='Qwen_baseline_0.8748',**values))
    try:
        with torch.inference_mode():
            for batch in loader:
                process(batch,rows,'public_test')
                print('EXPORTED',len(rows),'/',expected,'CC64',np.mean([r['cc_float64'] for r in rows]),flush=True)
            for iid in EXAMPLES:
                existing=next((r for r in rows if r['image_id']==iid),None)
                if existing:examples.append(dict(existing));continue
                records=[r for r in bundle.train_records if r.image_id==iid]
                if len(records)!=1:raise ValueError(f'Missing example {iid}')
                dataset=SALICONSaliencyDataset(records,s,training=False)
                process(collate_salicon([dataset[0]]),examples,'training_example')
    finally:
        model.close()
    id_sha=hashlib.sha256('\n'.join(r['sample_id'] for r in rows).encode()).hexdigest()
    if len(rows)!=5000 or len({r['sample_id'] for r in rows})!=5000 or id_sha!=IDS:
        raise ValueError('Test sample identities differ')
    if [r['sample_id'] for r in rows]!=[r['sample_id'] for r in original_rows]:
        raise ValueError('Historical record order differs')
    deltas=np.array([abs(r['cc_float64']-float(old['cc_float64'])) for r,old in zip(rows,original_rows)])
    aggregate=accumulator.compute()
    means={k:float(np.mean([r[k] for r in rows])) for k in (*METRICS,'cc_float64')}
    if abs(means['cc_float64']-reference['independent_float64_cc'])>2e-4:
        raise ValueError('Baseline CC did not reproduce within batch/device tolerance')
    for k in METRICS:
        if abs(means[k]-aggregate[k])>1e-5:raise ValueError('Per-image metric aggregation mismatch: '+k)
    write_csv(out/'qwen8748_test5000_wide.csv',rows)
    long=[dict(sample_id=r['sample_id'],image_id=r['image_id'],model=r['model'],metric=k,
               value=r['cc_float64'] if k=='cc' else r[k],higher_is_better=k not in ('kld','mae'))
          for r in rows for k in METRICS]
    write_csv(out/'qwen8748_test5000_long.csv',long)
    write_csv(out/'qwen8748_six_examples.csv',examples)
    paired=[]
    if a.ours_run:
        ours_report=read(a.ours_run/'candidate_recheck/reproduction.json')
        if ours_report['checkpoint_sha256']!='036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe':
            raise ValueError('Wrong Ours model')
        with (a.ours_run/'candidate_recheck/per_image_cc.csv').open(encoding='utf-8-sig',newline='') as f:
            ours={r['sample_id']:float(r['cc_float64']) for r in csv.DictReader(f)}
        if set(ours)!=set(r['sample_id'] for r in rows):raise ValueError('Ours/test mismatch')
        paired=[dict(sample_id=r['sample_id'],image_id=r['image_id'],qwen_cc=r['cc_float64'],
                     ours_cc=ours[r['sample_id']],ours_minus_qwen=ours[r['sample_id']]-r['cc_float64']) for r in rows]
        write_csv(out/'paired_cc_ours08625_qwen8748.csv',paired)
    source=out/'source_evidence';source.mkdir()
    for name in ('reproduction.json','per_image_cc.csv'):
        shutil.copy2(original/name,source/('historical_'+name))
    report=dict(baseline_checkpoint_sha256=PIN,selected_epoch=45,backbone_frozen=True,
        batch_size=96,test_samples=5000,test_ids_sha256=id_sha,means=means,historical_metrics=reference['metrics'],
        historical_cc64=reference['independent_float64_cc'],max_per_image_cc_difference=float(deltas.max()),
        mean_absolute_per_image_cc_difference=float(deltas.mean()),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),command=sys.argv,
        torch=torch.__version__,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
        paired_cc_count=len(paired),rows=rows,six_examples=examples,
        metric_notes=dict(cc='Pearson; prefer cc_float64 for scatter plots',kld='KL(GT || prediction), eps=1e-8',
            sim='Histogram intersection of sum-normalized densities',nss='Prediction z score at discrete fixation pixels',
            auc_judd='Original deterministic project threshold sweep, no jitter',mae='Original project MAE: each prediction/target divided by own peak; NOT probability-density MAE'),
        protocol='224x224 original SALICON targets; val2014 is public test used for model selection; no independent validation',
        images_and_gt_included=False)
    (out/'metrics_and_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'README.md').write_text('# Qwen baseline 0.8748：逐图画图数据\n\n'
        'qwen8748_test5000_wide.csv：5000张完整public-test，一张一行；散点配对键sample_id/image_id。\n'
        'qwen8748_test5000_long.csv：30000行，按metric分组即可画分布/小提琴图。CC对应float64 Pearson。\n'
        'paired_cc_ours08625_qwen8748.csv：同5000图的Qwen/Ours CC及差值，可直接画配对散点图。\n'
        'qwen8748_six_examples.csv：之前六张展示图。715是测试图，其余五张为训练样例，不并入5000图分布。\n\n'
        '版本：冻结Qwen主干，同规格85412参数解码头，adapter 197184参数，50轮第45轮best。'
        '权重SHA、原始历史分数、本次均值和逐图重现差异见metrics_and_provenance.json。未重新训练。\n\n'
        '指标全部在224×224原始概率图/标签计算，未按显示图或叠加图计算。CC、SIM、NSS、AUC越高越好；KLD、MAE越低越好。'
        'MAE沿用项目峰值归一化口径。未额外增加SSIM等不同协议指标。本包只含数字，不含GT或图片。\n',encoding='utf-8')
    print('COMPLETE',out,means,'max historical delta',deltas.max(),flush=True)


if __name__=='__main__':main()
