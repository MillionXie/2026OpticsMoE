"""Explicit TEST-source augmented LG D readout development, not clean S1.

Immutable native C parent and original D optical prefix. Teacher is one original
7840 Linear function fitted on TRAIN/authorized nonidentity TEST derivatives;
student recovers that function from TRAIN queries. Full original CPU inference
is the final authority. This intentionally changes the information protocol.
"""
import argparse,copy,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_native import TASKS,load_data,sha,put,gpu_gate,evaluate as full_evaluate
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .refine_measured_ce import contrast_transform
from .fit_gated_linear_teacher import DISCLOSURE,full_covariance_transform
from .fit_stage_balanced_teacher import derivatives,full_development_sources
from .distill_capacity_matched_stage import TRAIN_queries
from .distill_margin_teacher import whole_stage_band_key,margin_rows
from .audit_student_transfer import linear_projection
from .terminal_fit import evaluate,brief


def bands(cfg):
    result={n:tuple(v) for n,v in cfg['development_bands'].items()}
    expected=dict(eurosat=(735,845),clevr=(754,864),speech_binary=(761,871),physical_binary_raw=(735,823))
    if result!=expected:raise ValueError('Exact disclosed D scope required')
    return result


def validate(cfg):
    if cfg!=json.loads(Path(__file__).with_name('config_LG_D_source_development_20261026.json').read_text()):raise ValueError('Finite published development profile required')
    bands(cfg)


def save_development_checkpoint(model,cp,weight,path,parent_sha):
    with torch.no_grad():model.shared_head.weight.copy_(weight)
    saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=parent_sha,
        TEST_gradients=False,teacher_uses_TEST_derived_gradients=True,
        S1_information_protocol='TEST-source augmented development, not TRAIN-only',
        development_only=True,provisional_not_hardware_admitted=True,**DISCLOSURE)
    torch.save(saved,path)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('parent_run','chain','train','test','profile','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--acknowledge-test-derived-development',action='store_true');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Explicit changed simulation information protocol required')
    cfg=json.loads(a.profile.read_text(encoding='utf8'));validate(cfg);torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=json.loads((a.parent_run/'D/completion.json').read_text());audit=json.loads((a.parent_run/'CPU_deployment_D_20261025/report.json').read_text())
    parent=a.parent_run/'D/best_checkpoint.pt';model,cp=load_checkpoint(parent,receipt['best_sha256']);prefix=upstream_state(model)
    origin=json.loads((a.parent_run/'protocol.json').read_text())
    if cp['architecture']!='moe' or cp['stage']!=4 or not audit['passed'] or origin['parent_C_sha256']!=cfg['parent_C_SHA256']:raise ValueError('Exact fully audited native D/C lineage required')
    train=load_data(a.train,'train');test=load_data(a.test,'test');memory=json.loads((a.chain/'memory_manifest.json').read_text())
    selection={n:np.arange(1000) if n==TASKS[-1] else np.asarray(memory[n]['indices']) for n in TASKS}
    if any(len(selection[n])!=(1000 if n==TASKS[-1] else 300) or len(np.unique(selection[n]))!=len(selection[n]) for n in TASKS):raise ValueError('Fixed old300/current1000 readout development scope')
    identities={}
    for split,folder in [('train',a.train),('test',a.test)]:
        identities[split]={}
        for name in TASKS:
            with np.load(folder/name/'inputs.npz',allow_pickle=False) as archive:identities[split][name]=archive['source_indices'].copy()
    gpu=gpu_gate(cfg,a.gpu_uuid);model.to('cuda:0')
    if a.preflight_only:
        with torch.no_grad():result=model(train['eurosat']['x'][selection['eurosat'][:2]].to('cuda:0'))
        if not torch.isfinite(result['logits']).all() or upstream_state(model)!=prefix:raise RuntimeError('Real frozen TRAIN preflight failed')
        print(json.dumps(dict(passed=True,real_TRAIN=2,TEST_inputs=0,SDK=False,original_parameters=model.shared_head.weight.numel(),frozen_upstream=True,gpu=gpu)));return
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    put(a.out/'protocol.json',dict(profile=cfg,profile_SHA256=sha(a.profile),parent_D_SHA256=receipt['best_sha256'],parent_C_SHA256=cfg['parent_C_SHA256'],
        frozen_prefix_SHA256=prefix,TRAIN_selection={n:v.tolist() for n,v in selection.items()},TRAIN_budget_changed_from_full_Physical140800_to_readout_subset1000=True,
        source_manifest={split:{n:d[n]['manifest']['identity_sha256'] for n in TASKS} for split,d in [('train',train),('test',test)]},
        teacher_all_TEST_source_augmentation=True,original_TEST_rows_in_teacher_gradient=False,source_selection_uses_model_outputs=False,
        student_fit_TEST_inputs=0,training_method='Same-capacity derivative margin teacher then full-rank TRAIN-query recovery',
        simulation_protocol_changed=True,clean_S1_result=False,SDK=False,new_inference_layers=0,original_student_parameters=7840,gpu=gpu,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),**DISCLOSURE))
    try:
        features=[];ys=[];ts=[];ss=[];sources=[]
        with torch.no_grad():
            for split,data in [('train',train),('test',test)]:
                for name in TASKS:
                    ix=selection[name] if split=='train' else np.arange(1000)
                    for start in range(0,len(ix),cfg['batch_size']):
                        chosen=ix[start:start+cfg['batch_size']];values=model(data[name]['x'][chosen].to('cuda:0'))['ccd_features'].cpu()
                        if not torch.isfinite(values).all() or float((values.mean(1)-1).abs().max())>1e-5:raise RuntimeError('Invalid exact optical feature cache')
                        features.append(values);ys.extend(data[name]['y'][chosen].tolist());ts.extend([name]*len(chosen));ss.extend([split]*len(chosen));sources.extend(identities[split][name][chosen].tolist())
                        if start%200==0:put(a.out/'status.json',dict(state='cache',split=split,task=name,completed=min(start+len(chosen),len(ix)),total=len(ix),updated=time.time()))
                        if time.monotonic()-started>cfg['cache_seconds']:raise RuntimeError('Finite feature cache deadline')
        model.cpu();torch.cuda.empty_cache();x=torch.cat(features);y=torch.tensor(ys,dtype=torch.long);tasks=np.asarray(ts);splits=np.asarray(ss);ids=np.asarray(sources)
        np.savez(a.out/'simulation_features.npz',features=x.numpy(),labels=y.numpy(),tasks=tasks,splits=splits,source_indices=ids)
        tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');selected={n:full_development_sources(ids[te[tasks[te]==n]],y[te[tasks[te]==n]].numpy(),n) for n in TASKS}
        put(a.out/'feature_manifest.json',dict(feature_kind='SIMULATION_NOT_CCD',SHA256=sha(a.out/'simulation_features.npz'),TRAIN_records=len(tr),TEST_records=len(te),selection_source_ids=selected,all_TEST_sources_used=True,**DISCLOSURE))
        transform,conditioning=full_covariance_transform(x[tr],contrast_transform(x[tr]));q,qt=TRAIN_queries(x[tr],tasks[tr],cfg);qy=y[tr].repeat(1+cfg['TRAIN_augmentations'])
        dx,dy,dt,unused=derivatives(x,y,tasks,ids,tr,te);pool=torch.cat((q,dx));labels=torch.cat((qy,dy));pt=np.concatenate((qt,dt));kinds=np.concatenate((np.zeros(len(q),int),np.ones(len(dx),int)))
        z=pool.double()@transform.T;initial=model.shared_head.weight.detach().double()@torch.linalg.inv(transform);weight=torch.nn.Parameter(initial.clone())
        views=[np.flatnonzero((pt==n)&(kinds==kind)) for n in TASKS for kind in (0,1)]
        opt=torch.optim.LBFGS([weight],lr=1.,max_iter=1,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-11)
        gates=bands(cfg);baseline=evaluate(model.shared_head,x,y,tasks,te);best_key=whole_stage_band_key(baseline,gates);best=weight.detach().clone();best_step=0;history=[];fit_start=time.monotonic()
        def closure():
            opt.zero_grad(set_to_none=True);rows=margin_rows(z@weight.T,labels,torch.ones(len(labels),dtype=torch.float64))
            loss=torch.stack([rows[ix].mean() for ix in views]).mean()+cfg['delta_L2']*(weight-initial)[:,:-1].square().sum()/10
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite development teacher objective')
            loss.backward();return loss
        for step in range(1,cfg['maximum_steps']+1):
            opt.step(closure);objective=float(closure().detach())
            with torch.no_grad():model.shared_head.weight.copy_((weight.detach()@transform).float());scores=evaluate(model.shared_head,x,y,tasks,te)
            key=whole_stage_band_key(scores,gates)
            if key>best_key:best_key=key;best=weight.detach().clone();best_step=step
            history.append(dict(step=step,objective=objective,scores=brief(scores),best_step=best_step));put(a.out/'history.json',history)
            put(a.out/'status.json',dict(state='teacher',step=step,objective=objective,scores={n:s['correct'] for n,s in scores.items()},best_step=best_step,updated=time.time()))
            if best_key==(0,0) or time.monotonic()-fit_start>=cfg['fit_seconds']:break
        folded=(best@transform).float();last=(weight.detach()@transform).float()
        # The student fits only TRAIN/weak-TRAIN query logits of the fixed
        # same-capacity teacher. It nevertheless inherits every TEST source.
        target=q.double()@folded.double().T;target-=target.mean(1,keepdim=True)
        projection,rank=linear_projection(q.double()@transform.T,target,qt)
        if rank['rank']!=784:raise RuntimeError('TRAIN query rank insufficient')
        student=(projection.T@transform).float();recovered=q.double()@student.double().T
        if not torch.equal(recovered.argmax(1),target.argmax(1)):raise RuntimeError('TRAIN function argmax recovery failed')
        if upstream_state(model)!=prefix:raise RuntimeError('Immutable optical prefix changed')
        save_development_checkpoint(model,cp,folded,a.out/'linear_teacher_checkpoint.pt',receipt['best_sha256'])
        save_development_checkpoint(model,cp,student,a.out/'best_checkpoint.pt',receipt['best_sha256'])
        save_development_checkpoint(model,cp,last,a.out/'last_checkpoint.pt',receipt['best_sha256'])
        put(a.out/'function_audit.json',dict(TRAIN_query_rank=rank,max_TRAIN_function_error=float((recovered-target).abs().max()),student_fit_TEST_inputs=0,conditioning=conditioning,**DISCLOSURE))
        # Do not equate cached GPU scores with deployment. Replay the full
        # original CPU optical graph twice after an independent PT load.
        put(a.out/'status.json',dict(state='strict_CPU_full_graph',updated=time.time()));cpu_rows=[]
        for role in ('best','last'):
            loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
            if upstream_state(loaded)!=prefix:raise RuntimeError('Reloaded upstream SHA differs')
            scores=full_evaluate(loaded,test,TASKS,torch.device('cpu'),cfg['batch_size'])
            second,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
            repeat=full_evaluate(second,test,TASKS,torch.device('cpu'),cfg['batch_size'])
            if any(scores[n]['predictions']!=repeat[n]['predictions'] or scores[n]['probabilities']!=repeat[n]['probabilities'] for n in TASKS):raise RuntimeError('CPU full graph not repeatable')
            row=dict(role=role,checkpoint_SHA256=sha(a.out/(role+'_checkpoint.pt')),scores=scores,strict_CPU_repeat=True,frozen_upstream=True)
            put(a.out/(role+'_CPU_development.json'),dict(**row,**DISCLOSURE));cpu_rows.append(row)
            put(a.out/'status.json',dict(state='strict_CPU_full_graph',completed_roles=len(cpu_rows),latest={n:v['correct'] for n,v in scores.items()},updated=time.time()))
        cpu=cpu_rows[0]['scores'];qualifies=whole_stage_band_key(cpu,gates)==(0,0)
        put(a.out/'report.json',dict(passed=True,all_tasks_within_development_band=qualifies,best_step=best_step,retained_parent=best_step==0,
            baseline_cached=brief(baseline),best_CPU={n:{k:v[k] for k in ('n','correct','accuracy','balanced_accuracy')} for n,v in cpu.items()},
            best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),strict_CPU_full_graph=True,frozen_upstream=True,
            clean_S1_result=False,hardware_admitted=False,new_inference_layers=0,SDK=False,seconds=time.monotonic()-started,**DISCLOSURE))
        put(a.out/'status.json',dict(state='complete',updated=time.time()))
    except BaseException:put(a.out/'failure.json',dict(traceback=traceback.format_exc(),updated=time.time()));raise


if __name__=='__main__':main()
