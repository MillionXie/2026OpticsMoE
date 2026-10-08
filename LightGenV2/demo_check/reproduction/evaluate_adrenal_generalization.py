"""Locked checkpoint replay and test audit, with no model/threshold selection."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import adrenal_generalization as g
from adrenal_generalization import r,np,torch,F,old

@torch.no_grad()
def route_audit(model,train,val,frontend):
    prompt=model.net.prompt;original=prompt.routing;had_instance_override='routing' in prompt.__dict__;probabilities={}
    for name,data in [('train',train),('val',val)]:
        allp=[]
        for x,y,_ in r.batches(data):
            if frontend is not None:x=frontend.encode(x)
            allp.append(original(x)['probabilities'].cpu())
        probabilities[name]=torch.cat(allp)
    mean=probabilities['train'].mean(0).cuda()
    def fixed(images):
        out=original(images);weights=mean.expand(len(images),-1)
        out.update(probabilities=weights,weights=weights,transmission=prompt.transmission(weights),prompt_amplitude=prompt.amplitude_map(weights))
        return out
    prompt.routing=fixed
    try:fixed_metrics,_=g.evaluate(model,val)
    finally:
        if had_instance_override:prompt.routing=original
        else:delattr(prompt,'routing')
    p=probabilities['val'];power=p.square()/p.square().sum(1,keepdim=True);labels=val[1].cpu()
    return dict(scope='Validation diagnostic only; replace input-dependent routing by mean probabilities estimated on training inputs, preserving phases and nine branches',fixed_train_mean_route_validation=fixed_metrics,train_mean_probabilities=mean.cpu().tolist(),validation_class_mean_power={str(c):power[labels==c].mean(0).tolist() for c in [0,1]},validation_power_std=power.std(0,unbiased=False).tolist(),mean_normalized_power_entropy=float(-(power*power.clamp_min(1e-12).log()).sum(1).mean()/np.log(9)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--selection-lock',type=Path,required=True);p.add_argument('--validation-only',action='store_true');p.add_argument('--historical-reference',action='store_true');a=p.parse_args();root=a.run
    selection=r.read(a.selection_lock)
    if a.historical_reference:
        assert not a.validation_only
        reference=selection['historical_validation_only_references'][root.name]
        for name,key in [('metadata.json','metadata_sha256'),('validation_results.json','validation_results_sha256'),('test_lock.json','test_lock_sha256')]:assert r.sha(root/name)==reference[key]
        cmd=[sys.executable,*r.read(root/'metadata.json')['command']]
        cmd[cmd.index('--phase')+1]='test';cmd[cmd.index('--out')+1]=str(root);cmd[cmd.index('--data')+1]=str(a.data)
        subprocess.run(cmd,check=True)
        r.EXP.update(batch_size=r.read(root/'metadata.json')['protocol']['batch_size'],data_npz=str(a.data.resolve()));r.setup();r.setseed(17);val=r.getdata('val');diagnostics=[]
        for entry in r.read(root/'test_lock.json')['models']:
            dest=root/entry['directory'];ck=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False);model=r.build(ck['variant']['architecture'],r.read(dest/'config.json')).cuda();model.load_state_dict(ck['model']);metrics,_=g.evaluate(model,val)
            assert metrics['auroc']==r.read(dest/'completed.json')['val']['auroc']
            diagnostics.append(dict(variant=ck['variant']['id'],validation=metrics));del model,ck;torch.cuda.empty_cache()
        r.save(root/'validation_classification_capture_diagnostics.json',diagnostics)
        r.save(root/'reference_evaluation_lock.json',dict(selection_lock_sha256=r.sha(a.selection_lock),evaluator_sha256=r.sha(__file__),command=cmd,time=r.now()))
        return
    assert root.name in selection['runs']
    assert r.sha(root/'test_lock.json')==selection['runs'][root.name]['test_lock_sha256']
    assert r.sha(root/'metadata.json')==selection['runs'][root.name]['metadata_sha256']
    assert r.sha(root/'validation_results.json')==selection['runs'][root.name]['validation_results_sha256']
    lock=r.read(root/'test_lock.json');metadata=r.read(root/'metadata.json');cfg=metadata['config'];sources=lock['sources'];assert sources==metadata['sources']
    for rel,h in sources.items():
        if rel.startswith('adrenal_softsign_'):assert r.sha(old.TASK/rel)==h,rel
    assert r.sha(g.__file__)==sources.get('training',sources['runner'])
    assert r.sha(old.__file__)==sources['augmentation']
    if 'router_temperature' in cfg:
        for config in r.CONFIGS.values():config['optical_router']['temperature']=cfg['router_temperature']
    if 'training' in sources:
        import adrenal_shared_frontend as f
        assert r.sha(f.__file__)==sources['runner']
    r.EXP.update(batch_size=cfg['batch_size'],data_npz=str(a.data.resolve()));r.setup();r.setseed(metadata['seeds'][0]);assert r.sha(a.data)==metadata['data_sha256']
    for rel,h in lock['files'].items():assert r.sha(root/rel)==h,rel
    if 'frontend_checkpoint' in sources:
        import adrenal_shared_frontend as f
        assert r.sha(f.__file__)==sources['runner'];path=Path(metadata['frontend_file']);assert r.sha(path)==sources['frontend_checkpoint']
        front=f.TinyFrontend().cuda();front.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);front.eval()
        for p in front.parameters():p.requires_grad_(False)
        f.install_frontend(front)
    # First independently reload each selected checkpoint and reproduce validation.
    val=r.getdata('val');replay=[];phase_audit=[];routing_audits=[];train=r.getdata('train')
    for rel in lock['models']:
        dest=root/rel;ck=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False);model=g.build(ck['variant'],cfg);model.load_state_dict(ck['model']);vm,rows=g.evaluate(model,val);expected=r.read(dest/'summary.json')['metrics']['val'];assert vm==expected,(rel,vm,expected);replay.append(dict(model=rel,validation_identical=True))
        x=front.encode(val[0][:8]) if 'frontend_checkpoint' in sources else val[0][:8]
        weights=torch.tensor([1188/(2*929),1188/(2*259)],device='cuda')
        loss,_=g.loss_terms(model,model(x),val[1][:8],weights,cfg);loss.backward()
        diagnostics={}
        for name,param in model.named_parameters():
            assert name.endswith('raw_phase') and torch.isfinite(param).all()
            diagnostics[name]=dict(rms_from_zero_initialization=float(param.detach().square().mean().sqrt()),gradient_norm=float(param.grad.norm()),sigmoid_saturation_fraction=float(((param.detach().sigmoid()<.01)|(param.detach().sigmoid()>.99)).float().mean()))
            assert diagnostics[name]['rms_from_zero_initialization']>0 and np.isfinite(diagnostics[name]['gradient_norm']) and diagnostics[name]['gradient_norm']>0
        phase_audit.append(dict(model=rel,selected_epoch=ck['epoch'],parameters=diagnostics))
        if ck['variant']['architecture']=='moe':routing_audits.append(dict(model=rel,dynamic_validation=vm,audit=route_audit(model,train,val,front if 'frontend_checkpoint' in sources else None)))
        del model,ck,loss;torch.cuda.empty_cache()
    r.save(root/'validation_replay.json',replay)
    r.save(root/'selected_phase_audit.json',phase_audit)
    r.save(root/'routing_audit.json',routing_audits)
    if a.validation_only:
        r.save(root/'validation_audit_execution.json',dict(command=sys.argv,evaluator_sha256=r.sha(__file__),time=r.now(),test_read=False));return
    # Only now read the test arrays, after all selected files have been checked.
    with np.load(a.data,allow_pickle=False) as z:
        x=z['test_images'].copy();y=z['test_labels'].reshape(-1).copy();ids=z['test_ids'].copy()
        image_hashes={split:{hashlib.sha256(img.tobytes()).hexdigest() for img in z[split+'_images']} for split in ['train','val','test']}
    r.save(root/'split_overlap_audit.json',dict(exact_projection_overlap_counts={a+'_'+b:len(image_hashes[a]&image_hashes[b]) for a,b in [('train','val'),('train','test'),('val','test')]},unique_projections={k:len(v) for k,v in image_hashes.items()},scope='Exact projected-array identity only; split-local row IDs do not establish patient-level independence'))
    assert np.bincount(y).tolist()==[229,69] and len(set(ids))==298
    test=(F.interpolate(torch.from_numpy(x[:,None]),size=(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1).cuda(),torch.from_numpy(y).long().cuda(),ids);results=[]
    for rel in lock['models']:
        dest=root/rel;ck=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False);model=g.build(ck['variant'],cfg);model.load_state_dict(ck['model']);m,rows=g.evaluate(model,test);r.csvwrite(dest/'test_predictions.csv',rows)
        t=r.read(dest/'thresholds.json')['policies']['val_balanced']['threshold'];prob=np.array([[a['score0'],a['score1']] for a in rows]);cal=g.full_metrics(y,prob,t,None);cal.pop('detector_plane_mse')
        results.append(dict(variant=ck['variant']['id'],seed=ck['seed'],test=m,val_threshold=cal));del model,ck;torch.cuda.empty_cache()
    r.save(root/'test_results.json',results);r.save(root/'test_execution.json',dict(command=sys.argv,selection_lock_sha256=r.sha(a.selection_lock),sources=sources,evaluator_sha256=r.sha(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),time=r.now()));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
