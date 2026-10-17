"""Explicit newly authorized TEST review of two already-trained, untested heads."""
import argparse,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_robust_precise as protected
import caltech20_rich_pair50_same_head as parent
import audit_caltech20_rich_pair50_alpha as mathcheck
HERE,ROOT,s,engine=parent.HERE,parent.ROOT,parent.s,parent.engine
FILES=('caltech20_rich_existing_head_TESTreview.py','audit_caltech20_rich_existing_head_TESTreview.py','CALTECH20_RICH_EXISTING_HEAD_TESTREVIEW_PROTOCOL_20261018.md')
OUT=HERE/'runs/simulation/caltech20_rich20_existing_samehead_candidates_TESTreview_20261018'

def sources():
    z=protected.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z

def identity():
    z=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)

def inputs(depth):
    key=f'L{depth}_rho03';folder=parent.folder(key);f=s.read(folder/'fit/fit_result.json');proof=s.read(folder/'audit_completed.json')
    assert proof['actual_audit_returncode']==0 and proof['audit_sha256']==s.sha(folder/'science/audit.json') and s.read(folder/'science/audit.json')['passed']
    meta=s.read(folder/'metadata.json');assert meta['sources']==parent.sources();engine.verify_sources(meta['sources'],meta['git_commit'])
    assert not f['candidate_accepted'] and f['candidate_sha256']==s.sha(folder/'fit/candidate.pt')
    oldtest=s.read(folder/'once_test/result.json');assert oldtest['test_reused'] and oldtest['production_head_TEST_evaluations']==0
    pt=torch.load(folder/'fit/candidate.pt',map_location='cpu',weights_only=False)
    optical=parent.parent.OUT/key;old=torch.load(optical/'best.pt',map_location='cpu',weights_only=False)
    assert parent.core_digest(pt['model'])==parent.core_digest(old['model'])
    with np.load(parent.parent.OUT/'once_tests'/key/'predictions.npz',allow_pickle=False) as z:pred={n:z[n] for n in z.files}
    assert pred['ids'].tolist()==parent.parent.base.manifest()['split_ids']['test']
    return folder,f,pt,pred

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    ident=identity();s.save(OUT.with_name(OUT.name+'.claim.json'),dict(ident,pid=os.getpid(),phase='human_new_TEST_PT_selection_two_existing_heads',automatic_retry=False),exclusive=True);OUT.mkdir();torch.set_num_threads(4)
    s.save(OUT/'metadata.json',dict(ident,pid=os.getpid(),depths=[4,6],test_used_for_selection=True,no_new_training=True),exclusive=True)
    results={}
    for depth in (4,6):
        folder,f,pt,pred=inputs(depth);key=f'L{depth}_rho03';dest=OUT/key;dest.mkdir()
        logits=pred['features'].astype(np.float64)@pt['model']['linear.weight'].double().numpy().T+pt['model']['linear.bias'].double().numpy()
        probability=np.exp(logits-logits.max(1,keepdims=True));probability/=probability.sum(1,keepdims=True);pred['probabilities']=probability
        parent.parent.base.save_np(dest/'predictions.npz',pred);metric=mathcheck.metrics(pred['labels'],probability)
        results[key]=dict(metrics=dict(train=f['candidate_metrics']['train'],val=f['candidate_metrics']['val'],test=metric),selected_pt_path=str(folder/'fit/candidate.pt'),pt_sha256=s.sha(folder/'fit/candidate.pt'),model_digest=s.digest(pt['model']),optical_selected_epoch=pt['epoch'],actual_joint_epochs=70,head_iterations=f['head_LBFGS_iterations'],new_cached_head_TEST_evaluations=1,new_optical_TEST_forwards=0,optimizer_steps=0,test_used_for_selection=True,independent_generalization=False)
        s.save(dest/'result.json',results[key],exclusive=True)
    s.save(OUT/'result.json',results,exclusive=True)

if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
        if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
