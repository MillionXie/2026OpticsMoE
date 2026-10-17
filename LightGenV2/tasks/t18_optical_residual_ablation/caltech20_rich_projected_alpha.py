"""Evidence-driven alpha curriculum with direct bounded coefficient learning."""
import ast,inspect,types
import caltech20_rich_alpha40_rebalance as core
import caltech20_rich_alpha40_fresh246 as previous
for n,v in vars(core).items():
    if not n.startswith('__'):globals()[n]=v
from caltech20_rich_projected_alpha_model import ProjectedMoE as Alpha40MoE,convert as remap
FILES=('caltech20_rich_projected_alpha.py','caltech20_rich_projected_alpha_model.py','caltech20_rich_projected_alpha_profile.json','check_caltech20_rich_projected_alpha.py','audit_caltech20_rich_projected_alpha.py','CALTECH20_RICH_PROJECTED_ALPHA_PROTOCOL_20261018.md')
def config():
    c=core.config();c.update(s.read(HERE/FILES[2]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_projected_alpha_cpu_20261018'
def sources():
    z=previous.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def alpha_schedule(epoch):
    c=config();return c['alpha_restart']+(c['alpha_curriculum_end']-c['alpha_restart'])*(epoch-1)/(c['alpha_curriculum_epochs']-1)
for node in ast.parse(inspect.getsource(core)).body:
    if isinstance(node,ast.FunctionDef) and node.name not in ('config','sources','ready','campaign','fit','gradients'):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),globals())
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    model.train();model.zero_grad(set_to_none=True);ps=[];cs=[];losses=[]
    for off in range(0,len(x),micro):
        model.offset=off;model.linear.mask=mask[off:off+micro];p,c=model(x[off:off+micro])
        loss=base.loss(model,p,c,y[off:off+micro],weights,cfg)+cfg['KD_weight']*old.kd.KD_loss(p,q[off:off+micro],mass[off:off+micro],cfg)
        (loss*len(p)/len(x)).backward();ps.append(p.detach());cs.append(c.detach());losses.append(float(loss.detach())*len(p)/len(x))
    model.linear.mask=None;return torch.cat(ps),torch.cat(cs),sum(losses)
# Preserve the reviewed fit kernel; explicit, asserted training-policy changes.
text=inspect.getsource(core.fit)
site="for n,p in model.named_parameters():p.requires_grad_(not warm or not (n=='raw_router' or n.startswith('phase_sets.') or n.startswith('fusion_logits.')))"
assert text.count(site)==1
text=text.replace(site,"for n,p in model.named_parameters():p.requires_grad_(not n.startswith('fusion_logits.') or tune>cfg['alpha_curriculum_epochs'])\n        if tune<=cfg['alpha_curriculum_epochs']:model.set_alpha(alpha_schedule(tune))")
assert text.count('opt.step();steps+=1')==1;text=text.replace('opt.step();steps+=1','opt.step();model.project();steps+=1')
site="electronic_only=warm,adam_updates=steps";assert text.count(site)==1;text=text.replace(site,"electronic_only=warm,alpha_trainable=tune>cfg['alpha_curriculum_epochs'],adam_updates=steps")
exec(compile(ast.parse(text),__file__+':explicit_projected_fit','exec'),globals())
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['projection_analytic_proof']['passed'] and p['all_kernel_and_audit_global_symbols_resolved'];identity();old.controls();return p
def campaign():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'direct_alpha_curriculum_three_depths');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),frozen_controls=old.controls()),exclusive=True)
    pending=list(config()['cases']);jobs=[];complete=[];failures=[];waiting=time.monotonic()
    while pending or jobs:
        slots=resources.slots();used={q['gpu'] for q in jobs}
        for index in config()['gpu_order']:
            gpu=resources.GPU[index]
            if not pending or gpu in used or slots['counts'][gpu]>0:continue
            key=pending.pop(0);s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            log=(OUT/(key+'.log')).open('x');p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','fit','--key',key,'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append(dict(key=key,gpu=gpu,pid=p.pid,process=p,log=log));used.add(gpu)
        for job in list(jobs):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();ex={k:v for k,v in job.items() if k not in ('process','log')};ex.update(returncode=rc,actual_exit=True);s.save(OUT/(job['key']+'.producer_exit.json'),ex,exclusive=True);jobs.remove(job)
            if rc:failures.append(ex);pending=[];continue
            f=OUT/job['key'];claim(f/'audit_claim.json','independent_projected_Adam_EMA_and_selection')
            with (f/'audit.log').open('x') as log:arc=subprocess.run([os.sys.executable,str(HERE/FILES[4]),'--key',job['key']],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
            if arc:failures.append(dict(key=job['key'],audit_returncode=arc));pending=[];continue
            assert s.read(f/'science/audit.json')['passed'];s.save(f/'audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(f/'science/audit.json')),exclusive=True);complete.append(ex)
        assert time.monotonic()-waiting<18000,'Queue deadline protection, no automatic extension'
        s.save(OUT/'status.json',dict(state='failed' if failures else 'projected_alpha_training_or_waiting' if jobs or pending else 'projected_alpha_all_reviewed',active=[{k:v for k,v in q.items() if k not in ('process','log')} for q in jobs],pending=pending,complete=complete,failures=failures))
        if pending or jobs:time.sleep(2)
    assert not failures
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        folder=OUT/a.key if a.key else OUT;mark=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
