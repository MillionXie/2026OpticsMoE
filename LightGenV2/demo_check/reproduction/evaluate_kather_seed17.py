"""Lock completed seed17 checkpoints and evaluate on each original training GPU."""
import argparse,csv,json,os,subprocess,sys,time,hashlib
from pathlib import Path

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)

def evaluate(a):
    import kather_coverage as c
    from kather_coverage import b,m,k,r,torch,np
    from verify_bloodmnist import predictions
    lock=read(a.out/'lock.json');entry=lock['entry'];x=entry['result'];folder=Path(entry['folder']);cfg=lock['config']
    assert c.sources()==lock['sources'];assert sha(a.data)==lock['data_sha256'];assert sha(folder/'best_checkpoint.pt')==x['checkpoint_sha256']
    torch.set_num_threads(4);r.setseed(17);model=m.build(x['arch'],x['depth'],cfg);ck=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False);assert ck['config']==cfg and ck['epoch']==x['selected_epoch'];model.load_state_dict(ck['model']);scores={};ids={};max_error=0.
    # Both non-test splits are replayed before opening the test arrays.
    for split in ['val','train']:
        scores[split],rows=b.evaluate(model,k.load_data(a.data,split),x['arch'],cfg['batch_size'])
        with (folder/(split+'_predictions.csv')).open() as f:old=list(csv.DictReader(f))
        assert [z['sample_id'] for z in rows]==[z['sample_id'] for z in old]
        assert [z['label_pred'] for z in rows]==[int(z['label_pred']) for z in old]
        err=max(abs(z[f'score{i}']-float(o[f'score{i}'])) for z,o in zip(rows,old) for i in range(8));assert err<=2e-6,(x['name'],split,err);max_error=max(max_error,err)
        for key in ['accuracy','balanced_accuracy','macro_f1','macro_ovr_auroc','balanced_nll','detector_capture']:assert abs(scores[split][key]-x['metrics'][split][key])<=2e-6
        r.csvwrite(a.out/(split+'_predictions.csv'),rows);ids[split]=set(predictions(a.out/(split+'_predictions.csv'),scores[split])[0])
    save(a.out/'validation_replay.json',dict(passed=True,maximum_score_error=max_error,lock_sha256=sha(a.out/'lock.json'),gpu=m.environment()))
    scores['test'],rows=b.evaluate(model,k.load_data(a.data,'test'),x['arch'],cfg['batch_size']);r.csvwrite(a.out/'test_predictions.csv',rows);ids['test']=set(predictions(a.out/'test_predictions.csv',scores['test'])[0]);assert not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
    save(a.out/'result.json',dict(arch=x['arch'],depth=x['depth'],seed=17,selected_epoch=x['selected_epoch'],epochs_completed=x['epochs_completed'],metrics=scores,checkpoint_sha256=x['checkpoint_sha256'],source_folder=str(folder),validation_replay_maximum_error=max_error,test_csv_sha256=sha(a.out/'test_predictions.csv'),independent_numpy_verification=True,environment=m.environment()))

def watch(a):
    a.out.mkdir(parents=True,exist_ok=False);sel=read(a.pilot/'candidate_selection.json');chosen=sel['chosen'];entries=[e for e in sel['entries'] if e['candidate']==chosen];cfg=read(Path(entries[0]['folder']).parent/'metadata.json')['config'];sources=sel['sources'];done=set();started=time.time()
    save(a.out/'protocol_lock.json',dict(seed=17,depths=[2,4,6],architectures=['moe','moe_nooeo','d2nn_wide','d2nn_wide_nooeo'],config=cfg,sources=sources,data_sha256=sel['data_sha256'],pilot_selection_sha256=sha(a.pilot/'candidate_selection.json'),selection='Existing validation balanced NLL rule, no further tuning or test-based changes',test_previously_observed=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    while len(done)<12:
        available=list(entries)
        for p in (a.run/'jobs').glob('*/result.json'):
            x=read(p)
            if x['seed']==17:available.append(dict(result=x,folder=str((p.parent/x['name']).resolve()),candidate=chosen,reused=False))
        for entry in sorted(available,key=lambda e:(-e['result']['depth'],e['result']['arch'])):
            x=entry['result'];name=x['name']
            if name in done:continue
            folder=Path(entry['folder']);meta=read(folder.parent/'metadata.json');assert meta['config']==cfg and meta['sources']==sources and meta['data_sha256']==sel['data_sha256'];assert sha(folder/'best_checkpoint.pt')==x['checkpoint_sha256']
            dest=a.out/name;dest.mkdir();save(dest/'lock.json',dict(entry=entry,config=cfg,sources=sources,data_sha256=sel['data_sha256'],time=time.time()))
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=meta['environment']['visible_devices'];assert env['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
            save(a.out/'status.json',dict(state='evaluating',model=name,completed=len(done),total=12,time=time.time()))
            with (dest/'evaluation.log').open('w') as log:subprocess.run([sys.executable,'-u',__file__,'--phase','evaluate','--data',str(a.data),'--out',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            done.add(name);save(a.out/'results.json',[read(a.out/n/'result.json') for n in sorted(done)])
        if len(done)<12:
            save(a.out/'status.json',dict(state='waiting_for_remaining_training',completed=len(done),total=12,time=time.time()));assert time.time()-started<3600;time.sleep(5)
    save(a.out/'status.json',dict(state='complete',completed=12,total=12,time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['watch','evaluate'],required=True);p.add_argument('--run',type=Path);p.add_argument('--pilot',type=Path);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();{'watch':watch,'evaluate':evaluate}[a.phase](a)
