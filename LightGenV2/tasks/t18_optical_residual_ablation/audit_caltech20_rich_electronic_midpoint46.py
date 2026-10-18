"""Independent saved interpolation/VALID choice, reuse or once TEST audit."""
import inspect
import caltech20_rich_electronic_midpoint46 as m
import audit_caltech20_rich_weight_average as previous
text=inspect.getsource(previous.verify);site="same=m.s.digest(pt['model'])==m.s.digest(states[0]);";assert text.count(site)==1;text=text.replace(site,"known={m.s.digest(states[0]):m.parent(depth)[2]['test']};oldf=m.original.OUT/f'L{depth}';oldfit=m.s.read(oldf/'fit_result.json');oldpt=torch.load(oldf/'candidate.pt',map_location='cpu',weights_only=False);assert oldfit['candidate_SHA256']==m.s.sha(oldf/'candidate.pt') and oldfit['model_digest']==m.s.digest(oldpt['model']);known[oldfit['model_digest']]=m.previous.load_np(oldf/'candidate_test.npz');same=m.s.digest(pt['model']) in known;");site="old=m.parent(depth)[2]['test'];actual=";assert text.count(site)==1;text=text.replace(site,"old=known[m.s.digest(pt['model'])];actual=");ns=dict(previous.verify.__globals__);ns.update(m=m);exec(compile(text,__file__+':same_independent_math_plus_actual_known_digest_reuse','exec'),ns);verify=ns['verify']
text=inspect.getsource(previous.run);assert text.count('for d in (2,4,6)')==1;text=text.replace('for d in (2,4,6)','for d in (4,6)');ns=dict(previous.run.__globals__);ns.update(m=m,verify=verify);exec(compile(text,__file__+':same_independent_science_only46','exec'),ns);run=ns['run']
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:run(a.phase)
    except BaseException as e:
        p=m.OUT/(a.phase+'_audit_failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
