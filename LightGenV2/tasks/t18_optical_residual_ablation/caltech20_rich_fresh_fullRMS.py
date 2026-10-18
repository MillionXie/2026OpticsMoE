"""Fresh alpha.45 training with full RMS derivatives and TRAIN prototypes."""
import types
import caltech20_rich_prototype_consistency as baseline
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('original','baseline','closure'):globals()[n]=v
original=baseline.original
import caltech20_rich_weight_average as closure
FILES=('caltech20_rich_fresh_fullRMS.py','caltech20_rich_fresh_fullRMS_profile.json','check_caltech20_rich_fresh_fullRMS.py','audit_caltech20_rich_fresh_fullRMS.py','CALTECH20_RICH_FRESH_FULLRMS_PROTOCOL_20261018.md')
def config():
    c=baseline.config();c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_fresh_fullRMS_CPU_20261018'
def sources():
    z=closure.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
class Alpha40MoE(baseline.Alpha40MoE):
    def __init__(self,depth,rho=.3):super().__init__(depth,rho);self.full_rms_gradient=True
    def contract(self):
        c=super().contract();c.update(fresh_alpha45_full_RMS_and_TRAIN_prototypes=True,fresh_optimizer=True,additional_inference_parameters=0);return c
@lru_cache(None)
def incumbent(depth):
    if depth!=2:return closure.parent(depth)
    f=closure.OUT/'L2';rec=s.read(closure.OUT/'test_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(closure.OUT/'test_science/audit.json');fit=s.read(f/'fit_result.json');test=s.read(f/'test_result.json');assert test['metrics']['accuracy']>engine.metrics(closure.parent(2)[2]['test']['labels'],closure.parent(2)[2]['test']['probabilities'])['accuracy'];assert fit['candidate_SHA256']==s.sha(f/'candidate.pt');pt=torch.load(f/'candidate.pt',map_location='cpu',weights_only=False);meta=s.read(f/'metadata.json');q=dict(path=str(f/'candidate.pt'),sha256=fit['candidate_SHA256'],source_git=meta['git_commit'],module=original.__name__,class_name='Alpha40MoE',alpha_low=None,joint_budget_epochs=None,selected_optical_epoch=None,parameter_average=True,contributing_optical_epochs=[75,90,90],new_weight_average_epochs=0);pred={r:load_predictions(f/('candidate_'+r+'.npz')) for r in ('train','val','test')};return q,pt,pred
def initialize(depth,fresh):
    assert fresh;engine.seed(17);model=Alpha40MoE(depth);model.set_alpha(.45);model.anchor();return model
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','fit','mixed','gradients','campaign','feature_loss','target_data'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=baseline.prototype_data.__wrapped__;prototype_data=lru_cache(None)(types.FunctionType(f.__code__,globals(),'prototype_data',f.__defaults__,f.__closure__))
def gate(depth,fresh,data,case):
    assert fresh;proof=numeric_gate_kernel(depth,True,data,case);fresh_model=initialize(depth,True);expected=s.read(CPU/'checks.json')['fresh_states'][str(depth)]['initial_digest'];assert s.digest(fresh_model.state_dict())==expected;proof.update(fresh_initial_digest=expected,initial_alpha=fresh_model.alphas(),full_RMS_gradient=True,TRAIN_class_prototype=prototype_data(depth)[3],TEST_forward_calls=0);return proof
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['fresh_initial_states_reproduced'] and p['independent_RMS_JS_prototype_derivatives'];identity();old.controls()
    for d in (2,4,6):prototype_data(d)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
