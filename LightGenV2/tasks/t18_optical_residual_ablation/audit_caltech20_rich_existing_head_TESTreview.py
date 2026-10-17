"""Independent saved NumPy linear/ID/CM/NLL audit; no model/forward."""
import os
import numpy as np
import caltech20_rich_existing_head_TESTreview as m

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (m.OUT/'science').exists()
    ex=m.s.read(m.OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit']
    meta=m.s.read(m.OUT/'metadata.json');assert meta['sources']==m.sources() and meta['test_used_for_selection'];m.engine.verify_sources(meta['sources'],meta['git_commit'])
    receipts={}
    for depth in (4,6):
        folder,f,pt,pred=m.inputs(depth);key=f'L{depth}_rho03';r=m.s.read(m.OUT/key/'result.json')
        assert r['pt_sha256']==m.s.sha(folder/'fit/candidate.pt') and r['model_digest']==m.s.digest(pt['model']) and r['test_used_for_selection'] and not r['independent_generalization']
        met,receipt=m.mathcheck.verify_predictions(m.OUT/key/'predictions.npz',pt['model'],'test',m.parent.parent.base.manifest());m.mathcheck.agree(met,r['metrics']['test'])
        for role in ('train','val'):
            with np.load(folder/'fit'/('selected_'+role+'.npz'),allow_pickle=False) as z:
                assert z['ids'].tolist()==m.parent.parent.base.manifest()['split_ids'][role]
                logits=z['features'].astype(np.float64)@pt['model']['linear.weight'].double().numpy().T+pt['model']['linear.bias'].double().numpy()
                probability=np.exp(logits-logits.max(1,keepdims=True));probability/=probability.sum(1,keepdims=True)
                actual=m.mathcheck.metrics(z['labels'],probability)
            m.mathcheck.agree(actual,r['metrics'][role])
        receipts[key]=receipt
    (m.OUT/'science').mkdir();m.s.save(m.OUT/'science/audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],checks=receipts,model_objects=0,forward_calls=0,TEST_images_read=0,test_used_for_selection=True),exclusive=True)

if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.OUT/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
