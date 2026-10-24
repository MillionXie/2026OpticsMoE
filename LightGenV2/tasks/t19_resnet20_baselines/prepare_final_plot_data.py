"""Freeze the human-confirmed nine runs and export saved observations; no inference.

CPU-only. Checkpoints are copied into a separate archive and SHA-verified, never
overwritten. Original runs, maps and every alternative budget remain intact.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]


def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable_json(path,data):
    raw=(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Refuse overwriting different evidence: '+str(path))
    else:path.write_bytes(raw)


def immutable_copy(source,dest):
    expected=sha(source);dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with source.open('rb') as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
    if sha(dest)!=expected:raise ValueError('Archive SHA mismatch: '+str(dest))
    dest.chmod(0o444)
    return dict(file=str(dest.relative_to(TASK)),sha256=expected,bytes=dest.stat().st_size)


def flatten_metrics(obj,prefix='overall'):
    rows=[]
    for key,value in obj.items():
        if isinstance(value,dict):rows.extend(flatten_metrics(value,prefix+'/'+key))
        elif isinstance(value,(float,int)) and not isinstance(value,bool):rows.append([prefix,key,value])
    return rows


def retrieval_samples(pred):
    n=pred['queries'];assert len(pred['query_ids'])==n
    headers=['query_id','query_SKU','top1_candidate_id','top1_candidate_SKU','first_positive_rank','H@1','H@5','H@10','reciprocal_rank','average_precision','positive_count']
    rows=[]
    for i,(query,sku,order) in enumerate(zip(pred['query_ids'],pred['query_skus'],pred['order'])):
        assert len(order)==pred['candidates'] and len(set(order))==len(order)
        hits=[rank for rank,index in enumerate(order,1) if pred['candidate_skus'][index]==sku]
        assert hits and hits[0]==pred['first_positive_rank'][i]
        first=hits[0];ap=sum(j/rank for j,rank in enumerate(hits,1))/len(hits);best=order[0]
        rows.append([query,sku,pred['candidate_ids'][best],pred['candidate_skus'][best],first,int(first<=1),int(first<=5),int(first<=10),1/first,ap,len(hits)])
    assert len(rows)==n and len({r[0] for r in rows})==n
    return headers,rows


def pose_samples(saved,metrics):
    import torch
    from torch.utils.data import DataLoader
    from .other_data import pose_data
    torch.set_num_threads(2)
    _,test,_,_=pose_data()
    joints=list(metrics['per_joint_pck_at_0.2_torso'])
    headers=['sample_id','PCK@0.2_torso','PCKh@0.5_head','mean_pixel_error_px','normalized_mean_error_torso','visible_joints','torso_scale_px','head_scale_px']
    headers += [j+'_error_px' for j in joints]+[j+'_PCK_hit' for j in joints]
    rows=[];cursor=0
    for batch in DataLoader(test,batch_size=32,shuffle=False,num_workers=0):
        selected=saved[cursor:cursor+len(batch['sample_id'])]
        assert [r['sample_id'] for r in selected]==list(batch['sample_id'])
        xy=torch.tensor([r['prediction_xy'] for r in selected],dtype=torch.float32)
        distance=torch.linalg.vector_norm(xy-torch.nan_to_num(batch['keypoints'],nan=0.),dim=-1)
        for i,r in enumerate(selected):
            visible=batch['visible'][i].bool();error=distance[i];torso=float(batch['torso_scale'][i]);head=float(batch['head_scale'][i])
            assert visible.any() and torso>0 and head>0
            hit=(error<=.2*torso);headh=(error<=.5*head)
            rows.append([r['sample_id'],float(hit[visible].float().mean()),float(headh[visible].float().mean()),
                float(error[visible].mean()),float((error[visible]/torso).mean()),int(visible.sum()),torso,head]
                +[float(error[j]) if visible[j] else None for j in range(14)]
                +[int(hit[j]) if visible[j] else None for j in range(14)])
        cursor+=len(selected)
    assert cursor==len(saved)==1000
    for index,key in [(1,'pck_at_0.2_torso'),(2,'pckh_at_0.5_head'),(3,'mean_pixel_error'),(4,'normalized_mean_error_torso')]:
        observed=sum(row[index]*row[5] for row in rows)/sum(row[5] for row in rows)
        assert math.isclose(observed,metrics[key],abs_tol=2e-6,rel_tol=1e-7),(key,observed,metrics[key])
    return headers,rows


def sample_table(spec,pred,metrics):
    task=spec['task']
    if task.startswith('lgvq_'):
        headers=['sample_id','MOS_target','predicted_score','signed_error','absolute_error','squared_error']
        rows=[[r['sample_id'],r['target'],r['prediction'],r['prediction']-r['target'],abs(r['prediction']-r['target']),(r['prediction']-r['target'])**2] for r in pred]
        assert math.isclose(sum(r[4] for r in rows)/len(rows),metrics['mae'],abs_tol=1e-9)
        assert math.isclose(math.sqrt(sum(r[5] for r in rows)/len(rows)),metrics['rmse'],abs_tol=1e-9)
        return headers,rows
    if task.startswith('abo_') and task!='abo_conditional_edit':
        headers,rows=retrieval_samples(pred)
        for col,key in [(5,'hit_at_1'),(6,'hit_at_5'),(7,'hit_at_10'),(8,'mrr'),(9,'map')]:
            assert math.isclose(sum(r[col] for r in rows)/len(rows),metrics[key],abs_tol=2e-7),(task,key)
        return headers,rows
    if task=='lsp':return pose_samples(pred,metrics)
    if task=='salicon':
        rows=[[r['sample_id'],r['cc']] for r in pred]
        assert math.isclose(sum(r[1] for r in rows)/len(rows),metrics['cc'],abs_tol=1e-7)
        return ['sample_id','CC'],rows
    keys=[k for k,v in pred[0].items() if isinstance(v,(float,int)) and not isinstance(v,bool) and k!='predicted_task_index']
    prefix=['sample_id','task','instruction'] if task=='openmoji' else ['sample_id','mode']
    rows=[[r[k] for k in prefix+keys] for r in pred]
    for key in keys:
        if key in metrics:assert math.isclose(sum(r[key] for r in pred)/len(pred),metrics[key],abs_tol=1e-7)
    return prefix+keys,rows


def main(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only archival/export requires CUDA_VISIBLE_DEVICES empty')
    from .train_easy100 import require_published_source
    require_published_source()
    config_path=TASK/'configs/final_selected9_20261024.json';config=read(config_path)
    assert len(config['rows'])==len({r['task'] for r in config['rows']})==9
    archive=TASK/'runs/simulation/final_selected9_archive_20261024';archive.mkdir(parents=True,exist_ok=True)
    archive_meta=[];exports=[]
    for spec in config['rows']:
        run=(TASK/spec['run']).resolve();assert run.is_relative_to(TASK.resolve())
        result=read(run/'result.json');manifest=read(run/'run_manifest.json');audit=read(TASK/spec['audit'])
        assert result['status']=='complete' and audit['status'].startswith('passed')
        if spec['task']=='abo_image_to_text':assert result['optimizer_updates']==100 and result['completed_epochs']==0
        else:assert result['selected_epoch']==spec['epoch']
        metrics=result['metrics'];assert math.isclose(metrics[spec['primary_metric']],spec['expected_primary'],abs_tol=1e-7)
        weight=run/'last_checkpoint.pt';assert sha(weight)==spec['checkpoint_SHA']
        assert result.get('checkpoint_SHA',result.get('checkpoint_sha256'))==spec['checkpoint_SHA']
        pid=manifest.get('pid');proc=Path('/proc')/str(pid)/'cmdline'
        if proc.exists():assert spec['run'] not in proc.read_bytes().replace(b'\0',b' ').decode()
        commit=manifest['git_commit'];subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=ROOT,check=True)
        pred_path=run/'fixed_weight_predictions.json';pred=read(pred_path)
        headers,rows=sample_table(spec,pred,metrics)
        assert len(rows)==spec['prediction_count'] and len({r[0] for r in rows})==len(rows)
        counts=manifest.get('parameter_counts',manifest.get('parameters'));assert counts['total']-counts['trainable']==11176512
        source_files=[run/n for n in ('last_checkpoint.pt','result.json','run_manifest.json','fixed_weight_predictions.json','history.json','validation_result.json','sampler_and_training.json','sampler_orders.json','train_cache_identity.json') if (run/n).exists()]
        source_files += [TASK/spec['audit']]
        copied=[immutable_copy(p,archive/spec['task']/p.name) for p in source_files] if args.archive else []
        maps=run/'fixed_weight_maps.pt';map_identity=dict(file=str(maps.relative_to(TASK)),sha256=sha(maps),bytes=maps.stat().st_size) if maps.exists() else None
        sources={str(p.relative_to(TASK)):sha(p) for p in source_files}
        aggregate=flatten_metrics(metrics)
        if spec['task']=='abo_conditional_edit':
            for mode,values in audit['TEST_mode_metrics'].items():aggregate+=flatten_metrics(values,'mode/'+mode)
        if spec['task']=='openmoji':
            for operation in sorted({r['task'] for r in pred}):
                subset=[r for r in pred if r['task']==operation]
                aggregate += [[operation,k,sum(r[k] for r in subset)/len(subset)] for k in metrics if k!='samples']
        metadata=[['model','torchvision ResNet-18'],['pretrained_weights','ImageNet-1K IMAGENET1K_V1'],['epoch',spec['epoch']],['budget',spec['budget']],['test_samples',len(rows)],
            ['total_parameters',counts['total']],['trainable_parameters',counts['trainable']],['frozen_visual_parameters',11176512],
            ['training_scope','视觉及BN全部冻结；GRU＋任务头训练' if counts.get('text_encoder') else ('视觉及BN全部冻结；仅任务头训练' if counts['trainable'] else '视觉及BN全部冻结；无任务训练')],
            ['note',spec['note']],['data_notes','固定末轮；历史TEST已观察，用户最终指定版本。'],['code_commit',commit],['checkpoint_SHA256',spec['checkpoint_SHA']]]
        exports.append(dict(**spec,metrics=metrics,overall_rows=aggregate,metadata=metadata,sample_headers=headers,sample_rows=rows,source_sha256=sources,
            sample_notes=('SRCC/KRCC/PLCC是整体相关系数；逐视频表提供原始MOS与预测，以及由两者计算的误差。' if spec['task'].startswith('lgvq_') else
                '逐样本仅有已保存CC；其他显著性指标仅在整体指标表中。' if spec['task']=='salicon' else
                '由保存的预测坐标及原TEST标签计算每样本PCK/PCKh/误差；不新增模型推理。' if spec['task']=='lsp' else
                '由原始保存的完整排名按精确SKU计算逐查询H@K、RR、AP。' if isinstance(pred,dict) else '来自固定末轮已保存逐测试样本指标。')))
        archive_meta.append(dict(task=spec['task'],run=spec['run'],checkpoint_SHA=spec['checkpoint_SHA'],code_commit=commit,source_sha256=manifest.get('source_sha256'),
            original_source_files=sources,archived_files=copied,retained_original_maps=map_identity,parameter_counts=counts,primary_metric=spec['primary_metric'],primary_value=metrics[spec['primary_metric']]))
        print(json.dumps(dict(task=spec['task'],rows=len(rows),primary=metrics[spec['primary_metric']],archived_files=len(copied)),ensure_ascii=False),flush=True)
    payload=dict(profile=config['profile'],selection_config_SHA=sha(config_path),tasks=exports,no_new_neural_forward=True,no_optimization=True)
    output=TASK/'runs/outputs/t19_final_plot_data_20261024';immutable_json(output/'plot_data.json',payload)
    seal=dict(profile=config['profile'],selection_config_SHA=sha(config_path),status='sealed' if args.archive else 'verified',runs=archive_meta,no_original_files_removed=True,no_new_neural_forward=True,
        code_recovery='Recorded published Git commits and per-run source SHA; final selection version is tagged in Git after publication.')
    immutable_json(archive/'archive_manifest.json',seal)
    print(json.dumps(dict(status=seal['status'],tasks=9,payload=str(output/'plot_data.json'),payload_SHA=sha(output/'plot_data.json'),archive_manifest_SHA=sha(archive/'archive_manifest.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',action='store_true');main(parser.parse_args())
