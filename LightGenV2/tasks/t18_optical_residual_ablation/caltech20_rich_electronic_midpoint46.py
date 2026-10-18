"""One additional VALID midpoint per depth, all prior TEST digests reused."""
import inspect,types
import caltech20_rich_electronic_weight_average46 as original
for n,v in vars(original).items():
    if not n.startswith('__') and n not in ('original',):globals()[n]=v
FILES=('caltech20_rich_electronic_midpoint46.py','check_caltech20_rich_electronic_midpoint46.py','audit_caltech20_rich_electronic_midpoint46.py','CALTECH20_RICH_ELECTRONIC_MIDPOINT46_PROTOCOL_20261018.md')
def config():
    c=original.config();c.update(run='caltech20_rich20_electronic_VALID_midpoint46_20261018',mixture_fractions=[0.,.5,.75,1.],only_new_VALID_fraction=.75,known_TEST_parameter_digests_reused=True);return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_electronic_midpoint46_CPU_20261018'
def sources():
    z=original.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','state','initialize','configure','score'):
    f=getattr(original,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources() and q['optical_phase_bytes_identical_for_all_fractions'];identity();recent.closure.old.controls()
    for phase in ('fit','test'):
        rec=s.read(original.OUT/(phase+'_audit_completed.json'));assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(original.OUT/(phase+'_science/audit.json')) and s.read(original.OUT/(phase+'_science/audit.json'))['passed']
    for depth in (4,6):bank(depth)
def validation_choice(depth,fraction,model,val):
    old=original.OUT/f'L{depth}';fit=s.read(old/'fit_result.json')
    for i,row in enumerate(fit['records']):
        if row['fraction']==fraction:
            assert row['model_digest']==s.digest(model.state_dict());p=previous.load_np(old/f'VALID_choice{i}.npz');return row['val'],p
    assert fraction==.75;return base.evaluate(model,val)
def training_choice(depth,fraction,model,data):
    if fraction==0:
        p=parent(depth)[2]['train'];return metrics(p['labels'],p['probabilities']),p
    old=original.OUT/f'L{depth}';fit=s.read(old/'fit_result.json')
    if fraction==fit['selected_fraction']:
        assert fit['model_digest']==s.digest(model.state_dict());return fit['metrics']['train'],previous.load_np(old/'candidate_train.npz')
    return base.evaluate(model,data)
text=inspect.getsource(original.fit);site='metric,p=base.evaluate(model,val)';assert text.count(site)==1;text=text.replace(site,'metric,p=validation_choice(depth,fraction,model,val)');site='train,p=base.evaluate(model,data)';assert text.count(site)==1;text=text.replace(site,"train,p=training_choice(depth,rows[selected]['fraction'],model,data)");exec(compile(text,__file__+':same_fit_reuse_existing_VALID_choices_only_new_midpoint','exec'),globals())
def known_test(depth):
    known={s.digest(previous.initial_state(depth)):parent(depth)[2]['test']};f=original.OUT/f'L{depth}';fit=s.read(f/'fit_result.json');pt=torch.load(f/'candidate.pt',map_location='cpu',weights_only=False);assert fit['candidate_SHA256']==s.sha(f/'candidate.pt') and fit['model_digest']==s.digest(pt['model']);known[fit['model_digest']]=previous.load_np(f/'candidate_test.npz');return known
def test(depth,gpu):
    ready();rec=s.read(OUT/'fit_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(OUT/'fit_science/audit.json');f=OUT/f'L{depth}';claim(f/'once_test_claim.json','one_new_VALID_midpoint_or_known_TEST_reuse');fit=s.read(f/'fit_result.json');assert fit['candidate_SHA256']==s.sha(f/'candidate.pt');configure(gpu);torch.set_num_threads(4);known=known_test(depth);same=fit['model_digest'] in known
    if same:p=known[fit['model_digest']];metric=metrics(p['labels'],p['probabilities'])
    else:model=initialize(depth,fit['selected_fraction']).cuda();metric,p=base.evaluate(model,base.load('test'))
    base.save_np(f/'candidate_test.npz',p);s.save(f/'test_result.json',dict(passed=True,metrics=metric,model_digest=fit['model_digest'],new_TEST_models=0 if same else 1,parent_TEST_reused=same,known_TEST_cache_reused=same,TEST_used_to_select_fraction=False,TEST_informed_development=True,independent_generalization=False),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);p.add_argument('--depth',type=int,choices=[4,6],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:globals()[a.phase](a.depth,a.gpu)
    except BaseException as e:
        mark=OUT/(f'L{a.depth}.'+a.phase+'_failure.json') if OUT.exists() else OUT.with_name(OUT.name+'.'+a.phase+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
