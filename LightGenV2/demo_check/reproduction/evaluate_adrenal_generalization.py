"""Locked checkpoint replay and test audit, with no model/threshold selection."""
import argparse,json,subprocess,sys
from pathlib import Path
import adrenal_generalization as g
from adrenal_generalization import r,np,torch,F,old

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--selection-lock',type=Path,required=True);a=p.parse_args();root=a.run
    selection=r.read(a.selection_lock);assert root.name in selection['runs']
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
    r.EXP.update(batch_size=cfg['batch_size'],data_npz=str(a.data.resolve()));r.setup();assert r.sha(a.data)==metadata['data_sha256']
    for rel,h in lock['files'].items():assert r.sha(root/rel)==h,rel
    if 'frontend_checkpoint' in sources:
        import adrenal_shared_frontend as f
        assert r.sha(f.__file__)==sources['runner'];path=Path(metadata['frontend_file']);assert r.sha(path)==sources['frontend_checkpoint']
        front=f.TinyFrontend().cuda();front.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);front.eval()
        for p in front.parameters():p.requires_grad_(False)
        f.install_frontend(front)
    # First independently reload each selected checkpoint and reproduce validation.
    val=r.getdata('val');replay=[];phase_audit=[]
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
        phase_audit.append(dict(model=rel,selected_epoch=ck['epoch'],parameters=diagnostics));del model,ck,loss;torch.cuda.empty_cache()
    r.save(root/'validation_replay.json',replay)
    r.save(root/'selected_phase_audit.json',phase_audit)
    # Only now read the test arrays, after all selected files have been checked.
    with np.load(a.data,allow_pickle=False) as z:x=z['test_images'].copy();y=z['test_labels'].reshape(-1).copy();ids=z['test_ids'].copy()
    assert np.bincount(y).tolist()==[229,69] and len(set(ids))==298
    test=(F.interpolate(torch.from_numpy(x[:,None]),size=(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1).cuda(),torch.from_numpy(y).long().cuda(),ids);results=[]
    for rel in lock['models']:
        dest=root/rel;ck=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False);model=g.build(ck['variant'],cfg);model.load_state_dict(ck['model']);m,rows=g.evaluate(model,test);r.csvwrite(dest/'test_predictions.csv',rows)
        t=r.read(dest/'thresholds.json')['policies']['val_balanced']['threshold'];prob=np.array([[a['score0'],a['score1']] for a in rows]);cal=g.full_metrics(y,prob,t,None);cal.pop('detector_plane_mse')
        results.append(dict(variant=ck['variant']['id'],seed=ck['seed'],test=m,val_threshold=cal));del model,ck;torch.cuda.empty_cache()
    r.save(root/'test_results.json',results);r.save(root/'test_execution.json',dict(command=sys.argv,selection_lock_sha256=r.sha(a.selection_lock),sources=sources,evaluator_sha256=r.sha(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),time=r.now()));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
