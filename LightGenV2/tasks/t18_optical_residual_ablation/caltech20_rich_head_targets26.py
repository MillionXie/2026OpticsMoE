"""Warm continuation of the existing L2/L6 heads; protected originals never edited."""
import types,copy
import caltech20_rich_L6_progressive_bridge as previous
for n,v in vars(previous).items():
 if not n.startswith('__') and n!='previous':globals()[n]=v
FILES=('caltech20_rich_head_targets26.py','caltech20_rich_head_targets26_profile.json','audit_caltech20_rich_head_targets26.py','CALTECH20_RICH_HEAD_TARGETS26_PROTOCOL_20261018.md')
DEPTH=int(os.environ.get('T18_HEAD_DEPTH','2'));assert DEPTH in (2,6)
WHOLE_PROFILE=s.read(HERE/FILES[1]);PROFILE={k:copy.deepcopy(v) for k,v in WHOLE_PROFILE.items() if k!='cases'};PROFILE.update(WHOLE_PROFILE['cases'][str(DEPTH)])
CASE=int(os.environ.get('T18_L6_COMPETITION_CASE','0'));ROOT_OUT=HERE/'runs/simulation'/PROFILE['run'];OUT=ROOT_OUT/f'case{CASE:02d}';CPU=HERE/'runs/smoke'/f'caltech20_rich_head_targets26_L{DEPTH}_CPU_20261018'
def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def frozen():
 for name,z in PROFILE['protected'].items():assert s.sha(HERE/z['path'])==z['sha256'],name
 parent.old.controls()
def initial_parent():
 if DEPTH==6:
  done=s.read(previous.ROOT_OUT/'completed.json');ex=s.read(previous.ROOT_OUT/'campaign_exit.json');assert done['passed'] and done['state']=='target_achieved' and ex['actual_exit'] and ex['returncode']==0;q=dict(done['best'])
 else:
  root=HERE/'runs/simulation/caltech20_rich20_sameNetwork_VALID_weight_average246_20261018';folder=root/'L2';rec=s.read(root/'test_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(root/'test_science/audit.json');assert s.read(root/'test_science/audit.json')['passed']
  for phase in ('fit','test'):
   ex=s.read(root/f'L2.{phase}_exit.json');assert ex['actual_exit'] and ex['returncode']==0 and not Path('/proc',str(ex['pid']),'cmdline').exists()
  q=dict(path=str(folder/'candidate.pt'),sha256=s.sha(folder/'candidate.pt'),source_science=rec['audit_sha256'],predictions_folder=str(folder),science_directory=str(root),original_L2_average=True)
 assert q['sha256']==PROFILE['initial_parent_SHA256'];return q
def load_parent(q):
 root=Path(q['science_directory']);original=q.get('original_L2_average',False);prefix='candidate_' if original else 'selected_';rec=s.read(root/('test_audit_completed.json' if original else 'audit_completed.json'));audit=root/('test_science/audit.json' if original else 'science/audit.json');assert rec['passed'] if 'passed' in rec else rec['actual_audit_returncode']==0;assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==q['source_science']==s.sha(audit);p=Path(q['path']);assert s.sha(p)==q['sha256'];pt=torch.load(p,map_location='cpu',weights_only=False);pred={r:parent.load_predictions(Path(q['predictions_folder'])/(prefix+r+'.npz')) for r in ('train','val','test')}
 assert pt['model']['linear.weight'].shape==(20,256) and pt['model']['linear.bias'].shape==(20,)
 for role,n in (('train',3141),('val',383),('test',411)):
  z=pred[role];assert z['features'].shape==(n,256) and len(z['labels'])==n;assert np.max(abs(probabilities(z['features'],pt['model']['linear.weight'].numpy(),pt['model']['linear.bias'].numpy())-z['probabilities']))<=3e-6
 assert int((pred['test']['probabilities'].argmax(1)==pred['test']['labels']).sum())>=PROFILE['initial_correct'];return pt,pred
for name in ('identity','claim','specification','config','ready','plan','prepared','raw_gradient','original_update','update_kernel','update','gradient','improves','fit','campaign'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);a=p.parse_args()
 try:fit() if a.phase=='fit' else campaign()
 except BaseException as e:
  folder=OUT if a.phase=='fit' else ROOT_OUT;p=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
