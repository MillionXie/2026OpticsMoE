"""Guarded selected depth2/3 native BMP parity; synthetic probes, no dataset/test."""
import argparse,math,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from PIL import Image
import caltech_repair23_validate as v
r=v.r
export=v.export
reverse=v.reverse

def admitted(study,base):
    ref=v.reference(study)
    meta=r.read(base/'metadata.json');status=r.read(base/'status.json');proof=r.read(base/'cpu_report/audit.json')
    assert v.audit_core.exited(meta['pid']) and status['active']==status['pending']==0 and not status['failures'] and len(status['completed'])==2
    assert all(v.audit_core.exited(j['pid']) and j['exit_code']==0 for j in status['completed'])
    assert meta['parent_L4_validation']==ref==proof['parent_L4_validation']
    assert proof['passed'] and proof['all_depths_eligible'] and proof['model_forward_calls']==proof['test_arrays_read']==proof['test_evaluations']==0
    assert set(proof['fits'])==set(proof['gates'])=={'L2_rho03','L3_rho03'}
    r.engine.verify_sources(meta['sources'],meta['git_commit']);assert meta['sources']==v.sources()
    for key,depth in [('L2_rho03',2),('L3_rho03',3)]:
        arm=base/key;fit=r.read(arm/'fit_result.json');assert fit==proof['fits'][key]
        assert fit['depth']==depth and fit['rho']==.3 and fit['test_evaluations']==0 and not fit['test_used_for_selection']
        assert fit['config']==r.read(arm/'config.json') and fit['config']['parent_L4_validation']==ref and fit['config']['mixup_alpha']==ref['mixup_alpha']
        assert fit['sources']==meta['sources'] and fit['config']['cache_sha256']==ref['data_sha256']
        assert r.sha(arm/'best_checkpoint.pt')==fit['checkpoint_sha256'] and r.sha(arm/'history.json')==fit['history_sha256']
        hist=r.read(arm/'history.json');assert min(hist,key=lambda h:h['val']['balanced_nll'])['epoch']==fit['selected_epoch']
        assert proof['gates'][key]==v.eligibility(fit,hist) and proof['gates'][key]['passed']
    return ref,proof

def probes(model,config,out,checkpoint,checkpoint_sha,classes,sources):
    depth=model.depth if hasattr(model,'depth') else config['depth']
    assert depth in (2,3) and config['depth']==depth
    package=export.export_selected(model,config,out/'static',checkpoint,checkpoint_sha,classes,sources)
    for stage in package['stages']:
        s=stage['stage'];assert np.array_equal(reverse(out/'static'/f'phase_{s}_native.bmp','phase'),np.asarray(Image.open(out/'static'/f'phase_{s}_canonical.bmp')))
    with np.load(out/'static'/'amplitude_multipliers.npz') as z:
        for stage in package['stages']:
            s=stage['stage'];assert np.array_equal(z[s],model.plane(s).detach().abs().numpy()[20:498,20:498])
    torch.manual_seed(20261014)
    images={'random':torch.randint(0,256,(1,3,150,150),dtype=torch.uint8),'dim':torch.full((1,3,150,150),4,dtype=torch.uint8),'black':torch.zeros((1,3,150,150),dtype=torch.uint8)}
    checks=[]
    for name,image in images.items():
        amp=r.input_amplitude(image)
        with torch.no_grad():expected=model(amp)[0]
        current=[amp];weights=[None];calls=[];brightness=[]
        def replay(stage,a,p,ids):
            qa,qp=export.encode_stage(model,current[0],stage,weights[0]);assert np.array_equal(qa,np.rint(a*255).astype(np.uint8)) and np.array_equal(qp,p)
            dest=out/(name+'_'+stage);receipt=export.save_sample_bmps(model,current[0],stage,dest,weights[0]);gray=reverse(dest/'amplitude_000000_native.bmp','amplitude');phase=reverse(dest/'phase_native.bmp','phase')
            assert np.array_equal(gray,qa[0]) and np.array_equal(phase,qp);calls.append(stage);brightness.append(receipt['brightness'][0])
            field=torch.from_numpy(gray.astype(np.float32)/255)[None]*torch.exp(2j*math.pi*torch.from_numpy(phase.astype(np.float32)/256))[None];field=torch.nn.functional.pad(field,(20,20,20,20));intensity=model.propagator(field).abs().square()[:,20:498,20:498]
            if stage=='router':weights[0]=export.routing_from_ccd(model,intensity)
            elif int(stage)<depth-1:current[0]=export.amplitude_from_ccd(model,intensity,model.stages[int(stage)+1])
            return intensity.numpy()
        with torch.no_grad():actual=model(amp,callback=replay,ids=[name])[0]
        error=float((actual-expected).abs().max());assert error<=1e-6 and calls==['router']+[str(k) for k in range(depth)] and torch.equal(weights[0],model.routing_weights)
        if name=='black':assert all(b['active']['maximum']==0 for b in brightness) and torch.allclose(actual,torch.full_like(actual,.05),atol=1e-8,rtol=0)
        checks.append(dict(probe=name,native_exact=True,probability_max_error=error,captures=depth+1,routing_and_ccd_helpers_match=True,brightness=brightness))
    return checks

def native(study,base,out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists() and out==base/'selected_native_bmp'
    ref,proof=admitted(study,base)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip();tool_sources=dict(v.sources());tool_sources[Path(__file__).relative_to(r.ROOT).as_posix()]=r.sha(__file__);r.engine.verify_sources(tool_sources,commit)
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);results={}
    try:
        for key in ('L2_rho03','L3_rho03'):
            fit=proof['fits'][key];cp=base/key/'best_checkpoint.pt';state=torch.load(cp,map_location='cpu',weights_only=False)
            assert state['config']==fit['config'] and state['epoch']==fit['selected_epoch'] and state['sources']==fit['sources'] and state['selection_split']=='val' and state['selected_state_kind']=='ema'
            model=r.CorrectedMoE(fit['depth'],.3).eval();model.load_state_dict(state['model'],strict=True);dest=out/key;dest.mkdir()
            checks=probes(model,fit['config'],dest,cp,fit['checkpoint_sha256'],fit['config']['classes'],fit['sources']);assert r.sha(cp)==fit['checkpoint_sha256']
            results[key]=dict(selected_epoch=fit['selected_epoch'],checkpoint_sha256=fit['checkpoint_sha256'],config_sha256=r.sha(base/key/'config.json'),checks=checks,selected_trained_ema=True)
        admitted(study,base);r.engine.verify_sources(tool_sources,commit)
        receipt=dict(passed=True,results=results,parent_L4_validation=ref,depth_cpu_audit_sha256=r.sha(base/'cpu_report/audit.json'),tool_git_commit=commit,tool_sources=tool_sources,model_forward_calls=12,test_arrays_read=0,test_evaluations=0,devices_opened=0,gpu_used=False,physical_acceptance=False,scope='Selected depth2/3 EMA synthetic native BMP parity only; no train/val accuracy recomputation, formal fits or test')
        r.save(out/'checks.json',receipt,exclusive=True);return receipt
    except BaseException as err:r.save(out/'failure.json',dict(error=repr(err),automatic_retry=False),exclusive=True);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();native(a.study,a.base,a.out)
