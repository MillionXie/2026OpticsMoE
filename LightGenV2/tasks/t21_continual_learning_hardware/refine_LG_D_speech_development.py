"""One fixed Speech-focused teacher update, TRAIN-query student, full CPU audit.

All TEST-source derivatives were explicitly authorized for this development
branch. No new optical phase, inference layer, GPU, SDK, source or CCD. The
single complete optimization update is fixed before its evaluation, not a scan.
"""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_native import TASKS,put,sha,load_data,evaluate as full_evaluate
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .refine_measured_ce import contrast_transform
from .fit_gated_linear_teacher import DISCLOSURE,full_covariance_transform
from .fit_stage_balanced_teacher import derivatives
from .distill_capacity_matched_stage import TRAIN_queries
from .distill_margin_teacher import whole_stage_band_key,equal_task_retention_KL
from .audit_student_transfer import linear_projection
from .refine_LG_D_source_development import bands,save_development_checkpoint


def validate(cfg):
    if cfg!=json.loads(Path(__file__).with_name('config_LG_D_speech_development_20261026.json').read_text()):raise ValueError('Only one fixed published Speech update')
    if cfg['maximum_steps']!=1 or cfg['SDK'] or cfg['GPU']:raise ValueError('No additional scan, GPU or SDK')


def source_identity():
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();pin=os.environ.get('T21_PUBLISHED_SOURCE_COMMIT',head)
    pin=subprocess.check_output(['git','rev-parse',pin],text=True).strip();folder=Path(__file__).parent
    names=(Path(__file__).name,'config_LG_D_speech_development_20261026.json','distill_margin_teacher.py','fit_stage_balanced_teacher.py','model.py','hardware_bridge.py','refine_LG_D_source_development.py')
    for name in names:
        key='LightGenV2/tasks/t21_continual_learning_hardware/'+name
        current=subprocess.check_output(['git','hash-object','--path',key,str(folder/name)],text=True).strip();expected=subprocess.check_output(['git','rev-parse',pin+':'+key],text=True).strip()
        if current!=expected:raise ValueError('Published source byte identity mismatch: '+name)
    return dict(source_commit=pin,repository_HEAD=head,published_scoped_overlay=head!=pin,source_files={n:sha(folder/n) for n in names})


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('parent_run','test','profile','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Explicit TEST-source development acknowledgment required')
    cfg=json.loads(a.profile.read_text(encoding='utf8'));validate(cfg);source=source_identity();torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    report=json.loads((a.parent_run/'report.json').read_text());protocol=json.loads((a.parent_run/'protocol.json').read_text());fm=json.loads((a.parent_run/'feature_manifest.json').read_text())
    if not report['passed'] or not report['strict_CPU_full_graph'] or report['valid_for_formal_test_table'] is not False or fm['feature_kind']!='SIMULATION_NOT_CCD' or sha(a.parent_run/'simulation_features.npz')!=fm['SHA256']:raise ValueError('Exact completed development parent/cache required')
    model,cp=load_checkpoint(a.parent_run/'best_checkpoint.pt',report['best_sha256']);prefix=upstream_state(model)
    if prefix!=protocol['frozen_prefix_SHA256'] or cp['stage']!=4 or cp['architecture']!='moe' or not cp['development_only']:raise ValueError('Frozen native D graph required')
    with np.load(a.parent_run/'simulation_features.npz',allow_pickle=False) as d:x=torch.tensor(d['features']);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();splits=d['splits'].copy();ids=d['source_indices'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if len(tr)!=1900 or len(te)!=4000 or any(int((tasks[te]==n).sum())!=1000 for n in TASKS):raise ValueError('Exact prior source identities required')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();gates=bands(protocol['profile'])
    put(a.out/'protocol.json',dict(profile=cfg,profile_SHA256=sha(a.profile),parent_D_SHA256=report['best_sha256'],parent_C_SHA256=protocol['parent_C_SHA256'],
        original_optical_parent_D_SHA256=protocol['parent_D_SHA256'],frozen_prefix_SHA256=prefix,feature_manifest_SHA256=sha(a.parent_run/'feature_manifest.json'),
        cached_features=str(a.parent_run/'simulation_features.npz'),cached_features_SHA256=fm['SHA256'],same_prior_TEST_source_ids=True,
        teacher_uses_only_nonidentity_derivatives=True,student_fit_TEST_inputs=0,selection='One fixed complete update, then full CPU all-task comparison to parent; no GPU-cache checkpoint selection',
        loss='TRAIN CE .5 equally across tasks, Speech derivativeCE .5; other authorized derivative queries retain entire parent function with KL2 T2; L2=1e-5',
        **source,source_SHA256=sha(__file__),clean_S1_result=False,SDK=False,GPU=False,**DISCLOSURE))
    try:
        transform,conditioning=full_covariance_transform(x[tr],contrast_transform(x[tr]));q,qt=TRAIN_queries(x[tr],tasks[tr],cfg);qy=y[tr].repeat(1+cfg['TRAIN_augmentations'])
        dx,dy,dt,unused=derivatives(x,y,tasks,ids,tr,te);old=dt!='speech_binary';current=dt=='speech_binary'
        initial=model.shared_head.weight.detach().double()@torch.linalg.inv(transform);weight=torch.nn.Parameter(initial.clone());z=q.double()@transform.T;dz=dx.double()@transform.T
        old_target=(dx[old].double()@model.shared_head.weight.detach().double().T).detach();old_names=[n for n in TASKS if n!='speech_binary']
        opt=torch.optim.LBFGS([weight],lr=1.,max_iter=1,history_size=10,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12)
        def closure():
            opt.zero_grad(set_to_none=True);train_rows=F.cross_entropy(z@weight.T,qy,reduction='none')
            loss=.5*torch.stack([train_rows[qt==n].mean() for n in TASKS]).mean()+.5*F.cross_entropy(dz[current]@weight.T,dy[current])
            loss+=2*equal_task_retention_KL(dz[old]@weight.T,old_target,dt[old],old_names)+1e-5*(weight-initial)[:,:-1].square().sum()/10
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite fixed teacher update')
            loss.backward();return loss
        initial_loss=float(closure().detach());opt.step(closure);final_loss=float(closure().detach());folded=(weight.detach()@transform).float()
        target=q.double()@folded.double().T;target-=target.mean(1,keepdim=True);projection,rank=linear_projection(q.double()@transform.T,target,qt);candidate=(projection.T@transform).float()
        if rank['rank']!=784 or not torch.equal((q.double()@candidate.double().T).argmax(1),target.argmax(1)):raise RuntimeError('TRAIN function recovery failed')
        put(a.out/'fit_report.json',dict(passed=True,complete_updates=1,initial_loss=initial_loss,final_loss=final_loss,TRAIN_query_rank=rank,conditioning=conditioning,
            source_expansion=False,student_fit_TEST_inputs=0,seconds=time.monotonic()-started,**DISCLOSURE))
        for role,w in [('linear_teacher',folded),('best',candidate),('last',candidate)]:save_development_checkpoint(model,cp,w,a.out/(role+'_checkpoint.pt'),report['best_sha256'])
        loaded,_=load_checkpoint(a.out/'best_checkpoint.pt',sha(a.out/'best_checkpoint.pt'));last,_=load_checkpoint(a.out/'last_checkpoint.pt',sha(a.out/'last_checkpoint.pt'))
        if any(not torch.equal(v,last.state_dict()[k]) for k,v in loaded.state_dict().items()) or upstream_state(loaded)!=prefix:raise RuntimeError('Saved best/last/frozen state changed')
        data=load_data(a.test,'test');put(a.out/'status.json',dict(state='strict_CPU_full_graph',updated=time.time()))
        scores=full_evaluate(loaded,data,TASKS,torch.device('cpu'),cfg['batch_size']);put(a.out/'status.json',dict(state='strict_CPU_repeat',first_pass={n:v['correct'] for n,v in scores.items()},updated=time.time()))
        again,_=load_checkpoint(a.out/'best_checkpoint.pt',sha(a.out/'best_checkpoint.pt'));repeat=full_evaluate(again,data,TASKS,torch.device('cpu'),cfg['batch_size'])
        if any(scores[n]['predictions']!=repeat[n]['predictions'] or scores[n]['probabilities']!=repeat[n]['probabilities'] for n in TASKS):raise RuntimeError('Full original CPU graph not repeatable')
        put(a.out/'best_CPU_development.json',dict(scores=scores,checkpoint_SHA256=sha(a.out/'best_checkpoint.pt'),strict_CPU_repeat=True,**DISCLOSURE))
        qualifies=whole_stage_band_key(scores,gates)==(0,0);parent_key=whole_stage_band_key(report['best_CPU'],gates);key=whole_stage_band_key(scores,gates)
        put(a.out/'report.json',dict(passed=True,all_tasks_within_development_band=qualifies,parent_CPU=report['best_CPU'],
            best_CPU={n:{k:v[k] for k in ('n','correct','accuracy','balanced_accuracy')} for n,v in scores.items()},better_than_parent_band_key=key>parent_key,
            best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),strict_CPU_full_graph=True,best_last_model_tensors_identical=True,
            frozen_upstream=True,clean_S1_result=False,hardware_admitted=False,new_inference_layers=0,SDK=False,GPU=False,seconds=time.monotonic()-started,**DISCLOSURE))
        put(a.out/'status.json',dict(state='complete',updated=time.time()))
    except BaseException:put(a.out/'failure.json',dict(traceback=traceback.format_exc(),updated=time.time()));raise


if __name__=='__main__':main()
