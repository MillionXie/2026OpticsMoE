"""Independent intervention selection and exact state-key/parameter proof."""
import torch
import caltech20_rich_VALID_mixer_calibration as m
from check_caltech20_rich_alpha40_fresh246 import unresolved
def run():
    assert m.os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'VALID_single_alpha_CPU')
    for n in ('fit','test','audit','state','decision','ready','configure','identity'):f=getattr(m,n);assert not unresolved(f.__code__,f.__globals__)
    checks={}
    for depth in (4,6):
        key,expected,sha=m.decision(depth);q,p,z=m.state(depth);original=m.diagnosis.closure.remap(p['model'],q['alpha_low']);a=m.config()['actions'][str(depth)];diff=[n for n in z if not torch.equal(original[n],z[n])];assert diff==[f"fusion_logits.{a['stage']}"] and z[diff[0]].dtype==torch.float64 and .4001<float(z[diff[0]])<.95 and sum(v.numel() for n,v in z.items() if n.startswith('linear.'))==5140;checks[str(depth)]=dict(parent_SHA256=q['sha256'],diagnosis_SHA256=sha,one_state_key=diff[0],alpha=float(z[diff[0]]),VAL_accuracy=expected['val']['accuracy'])
    m.diagnosis.closure.old.controls();m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=checks,new_inference_parameters=0,real_model_forward_calls=0,TRAIN_images_read=0,VAL_images_read=0,TEST_images_read=0,optimizer_steps=0),exclusive=True)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
