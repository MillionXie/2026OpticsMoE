"""Recover locked evaluation on original GPU UUIDs without changing tolerances.

Training sources/weights stay immutable. Preserve failed default evaluation;
replay every validation result exactly on its training device, then open test.
"""
import argparse,concurrent.futures,json,os,subprocess,sys,time,traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--dataset',choices=['bloodmnist','kather2016'],required=True);p.add_argument('--gpu-map',type=Path,required=True);p.add_argument('--phase',choices=['coordinate','validation','test'],default='coordinate');p.add_argument('--gpu',type=int);a=p.parse_args()
    import bloodmnist_multiseed as m
    if a.dataset=='kather2016':import kather2016_experiment as k
    b,r,torch,np=m.b,m.r,m.torch,m.np
    lock=r.read(a.run/'selection_lock.json');assert lock['sources']==m.sources() and lock['data_sha256']==r.sha(a.data);cfg=lock.get('config',r.read(a.run/'metadata.json').get('config'));mapping=r.read(a.gpu_map)['mapping'];dest=a.run/'evaluation_native_device'
    expected=54 if a.dataset=='bloodmnist' else 36;assert len(lock['entries'])==expected
    if a.phase=='coordinate':
        assert not (a.run/'results.json').exists(),'Never replace an already evaluated suite';prior=r.read(a.run/'status.json');assert prior['state']=='failed','Inspect the original evaluation failure before recovery';dest.mkdir(exist_ok=False);assignments={}
        for e in lock['entries']:
            x=e['result'];folder=Path(e['folder']);meta=r.read(folder.parent/'metadata.json');env=meta.get('environment',{});gpu=int(env.get('visible_devices','0'));name=env.get('gpu',meta.get('gpu'));assert mapping[str(gpu)]['name']==name,(x['name'],gpu,name);assert r.sha(folder/'best_checkpoint.pt')==x['checkpoint_sha256'];assignments[x['name']]=gpu
        r.save(dest/'assignments.json',assignments);r.save(dest/'metadata.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),evaluator_sha256=r.sha(__file__),sources=m.sources(),selection_lock_sha256=r.sha(a.run/'selection_lock.json'),data_sha256=r.sha(a.data),prior_failed_status=prior,gpu_identity=mapping,scope=__doc__,time=r.now()))
        gpus=sorted(set(assignments.values()));assert len(gpus)<=5
        def worker(gpu,phase):
            uuid=mapping[str(gpu)]['uuid']
            for attempt in range(10):
                occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).split()
                if uuid not in occupied:break
                time.sleep(.5)
            assert uuid not in occupied,('GPU busy; do not displace another process',gpu,uuid)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=uuid;cmd=[sys.executable,'-u',__file__,'--run',str(a.run),'--data',str(a.data),'--dataset',a.dataset,'--gpu-map',str(a.gpu_map),'--phase',phase,'--gpu',str(gpu)]
            with (dest/f'{phase}_gpu{gpu}.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        try:
            r.save(a.run/'status.json',dict(state='native_validation',time=r.now()))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as ex:list(ex.map(lambda g:worker(g,'validation'),gpus))
            validation=[x for g in gpus for x in r.read(dest/f'validation_gpu{g}.json')];assert len(validation)==expected and {x['model'] for x in validation}==set(assignments);r.save(dest/'validation_barrier.json',dict(all_validation_replayed_exactly=True,models=validation,time=r.now(),test_read=False))
            if a.dataset=='bloodmnist':
                import hashlib
                with np.load(a.data,allow_pickle=False) as z:
                    seen={hashlib.sha256(im.tobytes()).hexdigest() for split in ['train','val'] for im in z[split+'_images']};clean=[]
                    for i,im in enumerate(z['test_images']):
                        h=hashlib.sha256(im.tobytes()).hexdigest()
                        if h not in seen:clean.append(i);seen.add(h)
                r.save(a.run/'test_clean_indices.json',clean)
            r.save(a.run/'status.json',dict(state='native_test_evaluation',time=r.now()))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as ex:list(ex.map(lambda g:worker(g,'test'),gpus))
            results=[r.read(dest/e['result']['name']/'result.json') for e in lock['entries']];assert len(results)==expected;r.save(a.run/'results.json',results);r.save(a.run/'evaluation_directory.json',dict(directory=dest.name,reason='Exact original-device validation replay; default failed evaluation preserved',metadata_sha256=r.sha(dest/'metadata.json')));r.save(a.run/'evaluation_identity.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),evaluator_sha256=r.sha(__file__),sources=m.sources(),selection_lock_sha256=r.sha(a.run/'selection_lock.json'),native_validation_barrier_sha256=r.sha(dest/'validation_barrier.json'),time=r.now()));r.save(a.run/'status.json',dict(state='complete',evaluation_mode='original_gpu_uuid',time=r.now()))
        except BaseException:
            r.save(a.run/'status.json',dict(state='failed',traceback=traceback.format_exc(),recovery_evidence=str(dest),time=r.now()));raise
        return
    torch.set_num_threads(4);assert os.environ['CUDA_VISIBLE_DEVICES']==mapping[str(a.gpu)]['uuid'];assert torch.cuda.get_device_name()==mapping[str(a.gpu)]['name'];assignments=r.read(dest/'assignments.json');entries=[e for e in lock['entries'] if assignments[e['result']['name']]==a.gpu];r.save(dest/f'{a.phase}_gpu{a.gpu}_environment.json',m.environment())
    if a.phase=='validation':
        val=b.load_data(a.data,'val');data=b.load_data(a.data,'train');records=[]
        for e in entries:
            x=e['result'];folder=Path(e['folder']);r.setseed(x['seed']);model=m.build(x['arch'],x['depth'],cfg);initial={n:p.detach().clone() for n,p in model.named_parameters()};ck=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);vm,rows=b.evaluate(model,val,x['arch'],cfg['batch_size']);assert vm==x['metrics']['val'],(x['name'],vm,x['metrics']['val'])
            import csv
            with (folder/'val_predictions.csv').open() as f:old=list(csv.DictReader(f))
            assert len(rows)==len(old) and all(z['sample_id']==o['sample_id'] and z['label_pred']==int(o['label_pred']) and all(z[f'score{k}']==float(o[f'score{k}']) for k in range(8)) for z,o in zip(rows,old)),x['name']
            audit,maps=m.pixel_audit(model,x['arch'],data,val)
            for n,q in model.named_modules():
                if isinstance(q,m.PhaseLayer):
                    delta=q.raw_phase.detach()-initial[n+'.raw_phase'];audit[n].update(changed_fraction=float((delta.abs()>1e-7).float().mean()),rms_parameter_update=float(delta.square().mean().sqrt()))
            out=dest/x['name'];out.mkdir();r.save(out/'phase_audit.json',audit);np.savez_compressed(out/'phase_maps.npz',**maps);records.append(dict(model=x['name'],gpu_uuid=mapping[str(a.gpu)]['uuid'],exact_validation_metrics=True,exact_validation_scores=True));del model,initial,ck;torch.cuda.empty_cache()
        r.save(dest/f'validation_gpu{a.gpu}.json',records)
    else:
        assert r.read(dest/'validation_barrier.json')['all_validation_replayed_exactly'];test=b.load_data(a.data,'test');clean=r.read(a.run/'test_clean_indices.json') if a.dataset=='bloodmnist' else None
        for e in entries:
            x=e['result'];model=m.build(x['arch'],x['depth'],cfg);ck=torch.load(Path(e['folder'])/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);tm,rows=b.evaluate(model,test,x['arch'],cfg['batch_size']);out=dest/x['name'];r.csvwrite(out/'test_predictions.csv',rows);result=dict(e,test=tm,validation_replayed=True,evaluation_gpu_uuid=mapping[str(a.gpu)]['uuid'])
            if clean is not None:
                cr=[rows[i] for i in clean];r.csvwrite(out/'test_clean_predictions.csv',cr);result['clean_test']=b.metrics(np.array([v['label_true'] for v in cr]),np.array([[v[f'score{k}'] for k in range(8)] for v in cr]))
            r.save(out/'result.json',result);del model,ck;torch.cuda.empty_cache()
if __name__=='__main__':main()
