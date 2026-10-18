"""TRAIN100/fullVAL383 reversible stagewise electronic contribution probe."""
import argparse,importlib,inspect,os,subprocess,textwrap
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_L4_prototype_continue as closure
import caltech20_rich_dualview_hosthash_repair as physics
HERE,ROOT,s,base,engine,resources=closure.HERE,closure.ROOT,closure.s,closure.base,closure.engine,closure.resources
rms,features,energies=physics.rms,physics.features,physics.energies
FILES=('caltech20_rich_stagewise_electronic_diagnosis.py','check_caltech20_rich_stagewise_electronic_diagnosis.py','CALTECH20_RICH_STAGEWISE_ELECTRONIC_DIAGNOSIS_PROTOCOL_20261018.md')
OUT=HERE/'runs/smoke/caltech20_rich_stagewise_electronic_contribution_20261018';CPU=HERE/'runs/smoke/caltech20_rich_stagewise_electronic_CPU_20261018'
def sources():
    z=closure.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    z=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
class Probe(physics.Alpha40MoE):
    def __init__(self,depth,rho=.3):super().__init__(depth,rho);self.force=None
text=textwrap.dedent(inspect.getsource(physics.original.Alpha40MoE.forward));site='a=self.fusion_logits[k].float();mixed=';assert text.count(site)==1;text=text.replace(site,"a=self.fusion_logits[k].float();a=torch.tensor(self.force[k],device=amplitude.device,dtype=torch.float32) if self.force is not None and self.force[k] is not None else a;mixed=");ns={};exec(compile(text,__file__+':identical_forward_except_explicit_diagnostic_single_stage_alpha','exec'),globals(),ns);Probe.forward=ns['forward']
def evaluate(model,data,indices):
    probabilities=[]
    with torch.no_grad():
        for off in range(0,len(indices),8):idx=indices[off:off+8];p,_=model(base.encode(data[0][idx].cuda(),None,100));probabilities.append(p.cpu().numpy())
    return engine.metrics(data[1][indices].numpy(),np.concatenate(probabilities))
def run(depth,gpu):
    assert gpu in resources.GPU.values() and os.environ['CUDA_VISIBLE_DEVICES']==gpu;slots=resources.slots();assert slots['total']<=5 and slots['counts'][gpu]==1 and any(j['pid']==os.getpid() and j['gpu_uuid']==gpu for j in slots['jobs']);proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();identity();closure.old.controls()
    dest=OUT/f'L{depth}';assert not dest.exists() and not dest.with_name(dest.name+'.claim.json').exists();OUT.mkdir(exist_ok=True);claim(dest.with_name(dest.name+'.claim.json'),'reversible_stagewise_TRAIN_VAL_probe');dest.mkdir();s.gpu_budget.ALLOWED.update(resources.GPU.values());policy=s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    q,pt,_=closure.incumbent(depth);assert s.sha(Path(q['path']))==q['sha256'];engine.seed(17);model=Probe(depth).cuda().eval();model.load_state_dict(closure.remap(pt['model'],q['alpha_low']));model.requires_grad_(False);before=s.digest(model.state_dict());data={role:base.load(role) for role in ('train','val')};indices={'train':np.array([int(k) for c in range(20) for k in (data['train'][1]==c).nonzero()[:5,0]]),'val':np.arange(383)};assert len(indices['train'])==100
    mod=importlib.import_module(q['module']);reference=getattr(mod,q['class_name'])(depth,.3).cuda().eval();reference.load_state_dict(pt['model'])
    with torch.no_grad():x=base.encode(data['train'][0][indices['train'][:4]].cuda(),None,100);p,c=model(x);a,b=reference(x);pe=float((p-a).abs().max());ce=float((c-b).abs().max());assert pe<=3e-6 and ce<=3e-6
    del reference;torch.cuda.empty_cache();rows={};calls=2
    def collect():return {role:evaluate(model,data[role],indices[role]) for role in ('train','val')}
    rows['retained']=collect();assert abs(rows['retained']['val']['accuracy']-pt['metrics']['val']['accuracy'])<1e-12
    for k in range(depth):
        for alpha in (.85,.75):model.force=[None]*depth;model.force[k]=alpha;rows[f'stage{k}_alpha{alpha:.2f}']=collect()
        model.force=None;weight=model.electronic[k].local.weight;original=weight.detach().clone()
        with torch.no_grad():weight.zero_()
        rows[f'stage{k}_zero_learned9_identity_kept']=collect()
        with torch.no_grad():weight.copy_(original)
        assert torch.equal(weight,original)
    model.force=None;assert before==s.digest(model.state_dict()) and all(p.grad is None for p in model.parameters());closure.old.controls();identity();calls+=len(rows)*sum((len(v)+7)//8 for v in indices.values())
    s.save(dest/'diagnosis.json',dict(identity(),passed=True,depth=depth,parent=q,gpu_uuid=gpu,policy=policy,TRAIN_balanced_subset100=True,VAL_full383=True,original_probability_maxabs=pe,original_capture_maxabs=ce,records=rows,TRAIN_sample_ids=[base.manifest()['split_ids']['train'][i] for i in indices['train']],VAL_sample_ids=base.manifest()['split_ids']['val'],state_unchanged=True,optimizer_steps=0,TEST_images_read=0,TEST_forward_calls=0,actual_model_forward_calls=calls,new_inference_parameters=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--depth',type=int,choices=[2,4,6],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:run(a.depth,a.gpu)
    except BaseException as e:
        dest=OUT/f'L{a.depth}';mark=dest/'failure.json' if dest.exists() else dest.with_name(dest.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
