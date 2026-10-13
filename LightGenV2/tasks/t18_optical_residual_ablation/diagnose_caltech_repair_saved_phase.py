"""Saved CPU tensor diagnostics only: no model construction, inference, dataset or GPU."""
import argparse,hashlib,json,math,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech_repair234 as r

def exited(pid):
    p=Path('/proc')/str(pid);return not p.exists() or not (p/'cmdline').read_bytes()

def array_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def tensor_stats(raw,rho):
    raw=raw.detach().cpu().float();phi=2*math.pi*torch.sigmoid(raw);modulation=(1-rho)*torch.exp(1j*phi)+rho
    tv=[float((phi[1:]-phi[:-1]).square().mean()),float((phi[:,1:]-phi[:,:-1]).square().mean())]
    z=modulation.numpy().astype(np.complex128);spectrum=np.abs(np.fft.fft2(z-z.mean()))**2
    fy=np.fft.fftfreq(z.shape[0])[:,None];fx=np.fft.fftfreq(z.shape[1])[None,:];frequency=np.sqrt(fx**2+fy**2);denom=float(spectrum.sum())
    phase=torch.remainder(torch.angle(modulation),2*math.pi)/(2*math.pi);gray=torch.floor(phase*256).clamp(0,255);qphi=gray/256*2*math.pi
    error=torch.angle(torch.exp(1j*(qphi-torch.angle(modulation)))).abs()
    return dict(shape=list(raw.shape),parameters=raw.numel(),raw_min=float(raw.min()),raw_max=float(raw.max()),raw_std=float(raw.std(unbiased=False)),phase_standard_deviation_rad=float(phi.std(unbiased=False)),phase_tv_by_axis=tv,phase_tv_mean=sum(tv)/2,modulation_amplitude_mean=float(modulation.abs().mean()),modulation_amplitude_min=float(modulation.abs().min()),modulation_amplitude_max=float(modulation.abs().max()),quantized_phase_levels=int(gray.unique().numel()),phase_floor_error_max_rad=float(error.max()),phase_floor_error_mean_rad=float(error.mean()),modulation_high_frequency_fraction_above_1over8_cycle_per_pixel=float(spectrum[frequency>1/8].sum()/denom) if denom else 0,modulation_variance=float(np.mean(np.abs(z-z.mean())**2)))

def diagnose(base,out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();torch.set_num_threads(2)
    meta=r.read(base/'metadata.json');status=r.read(base/'status.json');proof=r.read(base/'cpu_report/audit.json')
    assert exited(meta['pid']) and len(status['completed'])==3 and status['active']==status['pending']==0 and not status['failures'] and all(exited(j['pid']) for j in status['completed'])
    assert proof['passed'] and proof['model_forward_calls']==proof['test_evaluations']==proof['test_arrays_read']==0
    r.engine.verify_sources(meta['sources'],meta['git_commit']);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip();own={Path(__file__).relative_to(r.ROOT).as_posix():r.sha(__file__)};r.engine.verify_sources(own,commit)
    out.mkdir(parents=True,exist_ok=False)
    try:
        checks=[]
        for key in ('L4_rho03_mixup02','L4_rho03_mixup04','L4_rho03_mixup06'):
            fit=r.read(base/key/'fit_result.json');assert fit==proof['fits'][key] and r.sha(base/key/'best_checkpoint.pt')==fit['checkpoint_sha256'];best=torch.load(base/key/'best_checkpoint.pt',map_location='cpu',weights_only=False);last_bytes=(base/key/'last_checkpoint.pt').read_bytes();last=torch.load(__import__('io').BytesIO(last_bytes),map_location='cpu',weights_only=False)
            assert best['epoch']==fit['selected_epoch'] and best['selected_state_kind']=='ema' and best['selection_split']=='val';assert best['sources']==last['sources']==meta['sources'] and best['config']==last['config']==fit['config']
            ordered=['raw_router']+[f'phase_sets.{k}' for k in range(10)];assert set(ordered)==set(best['model'])==set(last['ema'])
            g=torch.Generator().manual_seed(17);initial={n:torch.empty(best['model'][n].shape).uniform_(-.1,.1,generator=g) for n in ordered};assert array_digest(initial)==last['initial_parameter_sha256']==fit['initial_parameter_sha256']
            states={};cfg=fit['config']
            for label,state in [('initial',initial),('selected_ema',best['model']),('last_ema',last['ema'])]:
                tensors={n:tensor_stats(state[n],0 if n=='raw_router' else .3) for n in ordered};tv=sum(sum(v['phase_tv_by_axis']) for v in tensors.values())/22
                states[label]=dict(tensors=tensors,parameter_count=sum(v.numel() for v in state.values()),phase_smoothness_actual_formula=tv,phase_regularizer_coefficient=cfg['phase_smooth_weight'],weighted_phase_regularizer=cfg['phase_smooth_weight']*tv)
            hist=r.read(base/key/'history.json');selected=hist[fit['selected_epoch']-1]
            checks.append(dict(case=key,selected_epoch=fit['selected_epoch'],checkpoint_sha256=fit['checkpoint_sha256'],last_snapshot_sha256=hashlib.sha256(last_bytes).hexdigest(),initial_parameter_sha256=array_digest(initial),phase_tensors=11,main_stages=4,captures_including_router=5,parameters_per_training_image=states['selected_ema']['parameter_count']/1246,states=states,selected_train_balanced_nll=fit['metrics']['train']['balanced_nll'],selected_val_balanced_nll=fit['metrics']['val']['balanced_nll'],saved_mixed_training_mean_objective_at_selected_epoch=selected['train_loss'],selected_gap_pp=100*(fit['metrics']['train']['accuracy']-fit['metrics']['val']['accuracy'])))
        assert len({c['initial_parameter_sha256'] for c in checks})==1
        receipt=dict(passed=True,checks=checks,training_source_commit=meta['git_commit'],tool_git_commit=commit,tool_sources=own,CPU_audit_sha256=r.sha(base/'cpu_report/audit.json'),sources_current_original_match=True,model_constructed=0,model_forward_calls=0,data_arrays_read=0,test_evaluations=0,gpu_used=False,scope='Saved tensor properties only; capacity/regularizer magnitude and spectrum are descriptive, not causal or predictive of new configuration accuracy. Epoch mixed objective is not a component decomposition of selected EMA validation loss.')
        r.save(out/'checks.json',receipt,exclusive=True);return receipt
    except BaseException as e:r.save(out/'failure.json',dict(error=repr(e),automatic_retry=False),exclusive=True);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();result=diagnose(a.base,a.out);print(json.dumps(dict(passed=result['passed'],cases=[dict(case=c['case'],parameters=c['states']['selected_ema']['parameter_count'],phase_penalty=c['states']['selected_ema']['weighted_phase_regularizer']) for c in result['checks']],model_forward_calls=0,test_evaluations=0)))
