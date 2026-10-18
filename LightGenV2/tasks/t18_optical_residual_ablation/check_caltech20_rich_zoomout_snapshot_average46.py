"""Original independent interpolation math plus actual preserved-state identity."""
import types,inspect,numpy as np,torch
import caltech20_rich_zoomout_snapshot_average46 as m
import check_caltech20_rich_weight_average as previous
import audit_caltech20_rich_zoomout_snapshot_average46 as audit
text=inspect.getsource(previous.run);assert text.count('for d in (2,4,6):')==1;text=text.replace('for d in (2,4,6):','for d in (4,6):');ns=dict(previous.run.__globals__);ns.update(m=m,audit=audit);exec(compile(text,__file__+':existing_independent_math_only46','exec'),ns);inherited=ns['run']
def run():
    inherited();banks={}
    for d in (4,6):
        states,rows=m.bank(d);raw=torch.load(rows[1]['path'],map_location='cpu',weights_only=False)[rows[1]['state_key']]
        for f in m.config()['mixture_fractions']:
            actual=m.state(d,f)
            for n,v in actual.items():
                if v.is_floating_point():assert np.allclose(v.numpy(),(1-f)*states[0][n].numpy().astype(float)+f*raw[n].numpy().astype(float),atol=2e-7,rtol=1e-7)
                else:assert torch.equal(v,raw[n])
        banks[str(d)]=rows
    proof=m.s.read(m.CPU/'checks.json');proof.update(independent_raw_snapshot_interpolation=True,snapshot_banks=banks,L2_FINAL_untouched=True);m.s.save(m.CPU/'checks.json',proof)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
