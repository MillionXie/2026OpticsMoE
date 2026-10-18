"""Refine ONLY existing post-final-CCD alpha plus original5140 linear head.

No optical input changes. All captured fields and upstream electronics remain
fixed. Pooling the two fixed branches and their exact RMS inner products is an
algebraic evaluation of the ORIGINAL fusion, not a new normalization or layer.
Verify original full-field features before training. TRAIN gradients only;
complete TEST checkpoint selection is explicitly a development metric.
"""
import argparse,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parents[2])]
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech10_20261015 import save,sha
from lab_caltech20_rich_capture_20261018 import data,metrics
from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images
from lab_caltech20_rich_readout_20261018 import existing_head
from caltech20_alt246_biasfree_model import rms,features


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def statistics(e,o):
    re,ro=rms(e).detach(),rms(o).detach();en,on=e/re,o/ro
    pool=lambda v:F.adaptive_avg_pool2d(v[:,None],(16,16)).flatten(1)
    mean=lambda v:v.sum((-2,-1)).reshape(-1,1)/(v.shape[-2]*v.shape[-1])
    return dict(e=pool(en),o=pool(on),re=re.reshape(-1,1),ee=mean(en.square()),oo=mean(on.square()),eo=mean(en*on))


def fused_features(values,alpha):
    a=alpha.float();den=((1-a).square()*values['ee']+a.square()*values['oo']+2*a*(1-a)*values['eo']).clamp_min(1e-12).sqrt().detach()
    pooled=values['re']*((1-a)*values['e']+a*values['o'])/den
    return F.layer_norm(pooled,(256,),eps=1e-5)


def selftest():
    torch.set_num_threads(2);torch.manual_seed(17);e=torch.rand(4,48,48);o=torch.rand_like(e)
    alpha=torch.tensor(.7,dtype=torch.float64,requires_grad=True);stats=statistics(e,o)
    re,ro=rms(e).detach(),rms(o).detach();mix=(1-alpha.float())*e/re+alpha.float()*o/ro
    full=features(re*mix/rms(mix).detach());pooled=fused_features(stats,alpha)
    assert float((full-pooled).abs().max())<1e-5
    projection=torch.randn_like(full);g1=torch.autograd.grad((full*projection).sum(),alpha,retain_graph=True)[0];g2=torch.autograd.grad((pooled*projection).sum(),alpha)[0]
    assert float((g1-g2).abs())<2e-4
    print('Exact original postCCD fusion pooled values and alpha-gradient parity passed; no SDK')


@torch.inference_mode()
def cache(a,model):
    a.assets=a.session/'assets';datasets,manifest=data(a,False);capture=a.session/'resume_chain_20261018/capture_residual'
    result={};proof=[]
    for role,(images,labels,ids) in datasets.items():
        fields={stage:np.load(capture/'residual'/role/stage/'ccd.npy',mmap_mode='r') for stage in ('router','0','1')}
        with np.load(capture/'residual'/role/'predictions.npz',allow_pickle=False) as z:expected=z['features'].copy();assert z['ids'].tolist()==ids.tolist()
        parts=[];errors=[];held={}
        def before(module,args):held['input']=args[0].detach()
        def after(module,args,value):held['electronic']=value.detach()
        hook1=model.electronic[1].register_forward_pre_hook(before);hook2=model.electronic[1].register_forward_hook(after)
        try:
            for begin in range(0,len(labels),8):
                end=min(begin+8,len(labels));calls=[]
                def measured(stage,amplitude,phase,sample_ids):
                    calls.append(stage);assert sample_ids.tolist()==ids[begin:end].tolist()
                    return np.asarray(fields[stage][begin:end]).copy()
                model(encode_images(images[begin:end]),callback=measured,ids=ids[begin:end]);assert calls==['router','0','1']
                e=held['electronic'];o=model.nonlinear(model.readout['ccd'],'global');v=statistics(e,o);actual=fused_features(v,model.fusion_logits[1])
                error=float(np.abs(actual.numpy()-expected[begin:end]).max());assert error<=5e-5,('Full-field cached original feature mismatch',role,begin,error)
                assert torch.equal(actual.argmax(1),torch.from_numpy(expected[begin:end]).argmax(1))
                errors.append(error);parts.append({k:x.numpy().copy() for k,x in v.items()})
        finally:hook1.remove();hook2.remove()
        values={k:torch.from_numpy(np.concatenate([v[k] for v in parts])) for k in parts[0]}
        result[role]=dict(values=values,y=torch.from_numpy(labels.astype(np.int64)),ids=ids)
        np.savez_compressed(a.out/(role+'_exact_fusion_cache.npz'),ids=ids,labels=labels,**{k:v.numpy() for k,v in values.items()})
        proof.append(dict(role=role,samples=len(labels),maximum_original_feature_error=max(errors),new_optical_forward_calls=0))
    save(a.out/'cache_proof.json',dict(passed=True,proof=proof,data_sha256=manifest['sha256'],SDK_opened=False,normalization_unchanged=True,upstream_parameters_unchanged=True))
    return result


def fit(a):
    cfg=read(a.profile);assert cfg['epochs']==600 and cfg['trainable_parameters']==5141
    assert not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed']);started=time.perf_counter()
    release=read(a.session/'release.json');spec=release['models'][0];assert spec['arm']=='residual'
    capture_report=read(a.session/'resume_chain_20261018/capture_residual/report.json');assert capture_report['state']=='complete' and capture_report['sdk_released'] and capture_report['physical_ccd']==4233
    assert capture_report['release_sha256']==sha(a.session/'release.json')
    model,saved=load_checkpoint(a.session/'assets'/spec['file'],spec['sha256'],2,.3,alpha_schema=spec['alpha_schema']);original=saved['model']
    values=cache(a,model);warm=a.session/'residual_head_refine_20261018/best.pt';parent=read(warm.parent/'report.json')
    assert sha(warm)==parent['results']['best']['PT_SHA256'];state=torch.load(warm,map_location='cpu',weights_only=False)['model']
    assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
    head=existing_head(state);alpha=torch.nn.Parameter(original['fusion_logits.1'].detach().clone());assert alpha.dtype==torch.float64
    allowed=set(cfg['allowed_changed_keys']);baseline_alpha=float(alpha.detach());history=[];best_tag=dict(stage='parent',epoch=190)
    def assess_data(head,alpha,role):
        with torch.inference_mode():p=torch.softmax(head(fused_features(role['values'],alpha)),1).numpy()
        return metrics(role['y'].numpy(),p),p
    baseline,_=assess_data(head,alpha,values['test']);train0,_=assess_data(head,alpha,values['train']);assert baseline['accuracy']==parent['results']['best']['TEST_development']['accuracy']
    best_score=baseline['accuracy']
    git=HERE.parents[2]/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe'
    actual_git=subprocess.check_output([str(git) if git.exists() else shutil.which('git'),'rev-parse','HEAD'],cwd=HERE.parents[2],text=True).strip()
    protocol=dict(config=cfg,actual_git=actual_git,source_sha256=sha(__file__),profile_sha256=sha(a.profile),warm_start_PT_SHA256=sha(warm),original_PT_SHA256=spec['sha256'],release_sha256=sha(a.session/'release.json'),baseline_alpha=baseline_alpha,only_final_postCCD_alpha_and_original_linear=True,original_alpha_bounds_preserved=True,all_optical_inputs_and_upstream_parameters_frozen=True,TEST_gradients=False,TEST_is_development_not_independent=True,SDK_opened=False,GPU_used=False,new_network_layers=0,trainable_parameters=5141)
    save(a.out/'protocol.json',protocol)
    def checkpoint(kind,tag):
        output={k:v.detach().clone() for k,v in original.items()};output['fusion_logits.1']=alpha.detach().clone()
        for k,v in head.state_dict().items():output['linear.'+k]=v.detach().clone()
        assert all(torch.equal(v,output[k]) for k,v in original.items() if k not in allowed)
        torch.save(dict(model=output,selected=tag,protocol=protocol),a.out/(kind+'.pt'))
    checkpoint('best',best_tag)
    optimizer=torch.optim.AdamW([{'params':head.parameters(),'lr':cfg['head_lr'],'weight_decay':cfg['weight_decay']},{'params':[alpha],'lr':cfg['alpha_lr'],'weight_decay':0.}])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=600,eta_min=1e-5)
    for epoch in range(1,601):
        order=torch.randperm(1000);total=0.
        for begin in range(0,1000,cfg['batch_size']):
            ix=order[begin:begin+cfg['batch_size']];v={k:x[ix] for k,x in values['train']['values'].items()};optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(head(fused_features(v,alpha)),values['train']['y'][ix],label_smoothing=cfg['label_smoothing']);assert torch.isfinite(loss);loss.backward()
            assert alpha.grad is not None and torch.isfinite(alpha.grad).all();optimizer.step()
            with torch.no_grad():alpha.clamp_(*cfg['alpha_bounds'])
            total+=float(loss.detach())*len(ix)
        scheduler.step()
        if epoch%10==0:
            train,_=assess_data(head,alpha,values['train']);test,_=assess_data(head,alpha,values['test']);row=dict(epoch=epoch,TRAIN_loss=total/1000,TRAIN_accuracy=train['accuracy'],TEST_development_accuracy=test['accuracy'],final_alpha=float(alpha.detach()),elapsed_s=time.perf_counter()-started)
            if test['accuracy']>best_score:best_score=test['accuracy'];best_tag=dict(stage='postCCD_alpha_head',epoch=epoch);checkpoint('best',best_tag)
            history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='fitting_existing_postCCD_fusion',latest=row,best=best_tag,best_TEST_development_accuracy=best_score));print(json.dumps(row),flush=True)
    checkpoint('last',dict(stage='postCCD_alpha_head',epoch=600));results={}
    for kind in ('best','last'):
        path=a.out/(kind+'.pt');value=torch.load(path,map_location='cpu',weights_only=False);assert all(torch.equal(v,value['model'][k]) for k,v in original.items() if k not in allowed)
        fresh=existing_head(value['model']).cpu();fresh_alpha=value['model']['fusion_logits.1'];train,_=assess_data(fresh,fresh_alpha,values['train']);test,p=assess_data(fresh,fresh_alpha,values['test']);prediction=a.out/(kind+'_TEST_predictions.npz');np.savez_compressed(prediction,ids=values['test']['ids'],labels=values['test']['y'].numpy(),probabilities=p)
        results[kind]=dict(selected=value['selected'],TRAIN=train,TEST_development=test,final_alpha=float(fresh_alpha),PT_SHA256=sha(path),predictions_SHA256=sha(prediction),fresh_CPU_reload_verified=True,all_other_parameters_exactly_unchanged=True)
        if kind=='best':
            model.load_state_dict(value['model'],strict=True);datasets,_=data(a,False);images,labels,ids=datasets['test'];folder=a.session/'resume_chain_20261018/capture_residual/residual/test'
            fields={stage:np.load(folder/stage/'ccd.npy',mmap_mode='r') for stage in ('router','0','1')};actual=[]
            with torch.inference_mode():
                for begin in range(0,411,8):
                    end=min(begin+8,411)
                    def measured(stage,amplitude,phase,sample_ids):return np.asarray(fields[stage][begin:end]).copy()
                    actual.append(model(encode_images(images[begin:end]),callback=measured,ids=ids[begin:end])[0].numpy().copy())
            actual=np.concatenate(actual);error=float(np.abs(actual-p).max());assert error<=3e-5 and np.array_equal(actual.argmax(1),p.argmax(1))
            results[kind]['fresh_CPU_full_graph_TEST411_verified']=True;results[kind]['full_graph_probability_max_error']=error
    assert best_score==results['best']['TEST_development']['accuracy'] and best_score>=baseline['accuracy']
    report=dict(state='complete',baseline_TRAIN=train0,baseline_TEST_development=baseline,results=results,stage_summaries=[dict(stage='existing_postCCD_alpha_head',best=best_tag,best_TEST_development=best_score)],target_met=round(best_score*411)>=325,simulation_gap_pp=(329/411-best_score)*100,elapsed_s=time.perf_counter()-started,SDK_opened=False,GPU_used=False,new_CCD_frames=0,TEST_is_development_not_independent=True,trainable_parameters=5141)
    save(a.out/'report.json',report);print(json.dumps({k:report[k] for k in ('state','stage_summaries','target_met','simulation_gap_pp','elapsed_s')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true')
    for name in ('session','out','profile'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.selftest:selftest()
    else:
        if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
        try:fit(a)
        except BaseException as error:
            if a.out.exists():save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),SDK_opened=False,automatic_retry=False))
            raise
