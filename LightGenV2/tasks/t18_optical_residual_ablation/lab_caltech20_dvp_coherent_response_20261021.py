"""TRAIN-only complex-response surrogate; NEVER a deployed LUT or model.

Fit all existing phase-diagnostic TRAIN frames; assess on other TRAIN images
and a different mask. No TEST, no camera, no geometry/SLM-LUT mutation. This
only tests whether a more faithful digital derivative is supportable by data.
"""
import argparse
import json
import hashlib
import os
import time
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2),encoding='utf-8')


def pack(a):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    from PIL import Image
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_TRAIN_response_20261020 import CASES,phase_patterns,checker_phase,HARDWARE
    torch.set_num_threads(2);assert not a.out.exists();a.out.mkdir()
    source=a.sequence/'TRAIN_response_verification_desktop_20261020';r=m.read(source/'report.json');proof=m.read(source/'independent_audit.json')
    assert proof['passed'] and proof['report_SHA256']==sha(source/'report.json') and r['response_frames']==216 and not r['TEST_used']
    old_fit=a.sequence/'TRAIN_response_fit_20261020/report.json';assert sha(old_fit)=='eb028992c05ff64d12b6adf3e37373f0ab9716ed17d8090685aa530707b834e2'
    assert m.session_hardware_digest(a.sequence)==HARDWARE and r['protected']=={p:sha(p) for p in r['protected']}
    data,dm=m.data_for(a.sequence);rows=[];transfer=None
    for depth,stage,parent,digest in CASES:
        session=a.sequence/f'L{depth}_strict8pp_20261020';cap=session/'capture'/stage
        model,_=m.model_for(session/'fits'/parent/'best.pt',digest,depth,session/'source_reference.ts')
        h=model.propagator.transfer.detach().numpy();assert transfer is None or np.array_equal(h,transfer);transfer=h.copy()
        def phase_of(current):
            plane=current.plane(stage);g=torch.floor(torch.remainder(torch.angle(plane),2*torch.pi)/(2*torch.pi)*256).clamp(0,255)[20:498,20:498].detach().numpy().astype(np.uint8)
            return g
        patterns=dict(phase_patterns(phase_of(model)));patterns.update({f'checker_{v}':checker_phase(v) for v in (0,64,128,192)})
        for record in [v for v in r['records'] if v['depth']==depth]:
            i=record['index'];amp=cap/'train'/f'{i:06d}_input.png';crop=Path(record['crop']);phase=source/f'L{depth}'/(record['condition']+'_phase.bmp')
            assert sha(amp)==record['canonical_SHA256'] and sha(crop)==record['crop_SHA256'] and sha(phase)==record['phase_SHA256']
            g=patterns[record['condition']];assert np.array_equal(m.phase_native(g),np.asarray(Image.open(phase)))
            assert record['sample_id']==str(data['train'][2][i])
            rows.append(dict(role='fit_TRAIN',depth=depth,index=i,sample_id=record['sample_id'],amp=np.asarray(Image.open(amp)).copy(),phase=g,observed=np.asarray(Image.open(crop)).copy(),receipt=sha(source/f'L{depth}'/f"{record['condition']}_{i:02d}.json")))
        fresh=a.sequence/'TRAIN_fullbatch_closedloop_20261021'/f'L{depth}_round1';cr=m.read(fresh/'capture/report.json');ca=m.read(fresh/'capture/independent_audit.json')
        assert ca['passed'] and ca['report_SHA256']==sha(fresh/'capture/report.json') and cr['data_SHA256']==dm['sha256']
        candidate=fresh/'proposal/candidate.pt';assert sha(candidate)==cr['PT_SHA256']
        model,_=m.model_for(candidate,sha(candidate),depth,session/'source_reference.ts');g=phase_of(model)
        assert np.array_equal(m.phase_native(g),np.asarray(Image.open(fresh/'capture/phase.bmp')))
        array_path=fresh/'capture/train/ccd.npy';assert sha(array_path)==cr['arrays']['train']['SHA256'];array=np.load(array_path,mmap_mode='r')
        for i in range(20,60):
            stem=fresh/'capture/train'/f'{i:06d}';record=m.read(stem.with_suffix('.json'));amp=stem.with_name(stem.name+'_input.png');crop=stem.with_suffix('.png')
            assert record['role']=='train' and record['index']==i and record['sample_id']==str(data['train'][2][i]) and record['PT_SHA256']==cr['PT_SHA256']
            assert sha(amp)==record['canonical_SHA256'] and record['phase_SHA256']==cr['phase_SHA256']
            assert np.array_equal(array[i],np.asarray(Image.open(crop)))
            rows.append(dict(role='other_TRAIN_mask_check',depth=depth,index=i,sample_id=record['sample_id'],amp=np.asarray(Image.open(amp)).copy(),phase=g,observed=np.asarray(Image.open(crop)).copy(),receipt=sha(stem.with_suffix('.json'))))
    assert len(rows)==296 and sum(v['role']=='fit_TRAIN' for v in rows)==216
    fit_ids={v['sample_id'] for v in rows if v['role']=='fit_TRAIN'};other_ids={v['sample_id'] for v in rows if v['role']!='fit_TRAIN'};assert not fit_ids&other_ids
    path=a.out/'observations.npz';np.savez_compressed(path,amplitude=np.stack([v['amp'] for v in rows]),phase=np.stack([v['phase'] for v in rows]),observed=np.stack([v['observed'] for v in rows]),fit=np.array([v['role']=='fit_TRAIN' for v in rows]),depth=np.array([v['depth'] for v in rows]),transfer=transfer)
    manifest=dict(schema='TRAIN_coherent_response_v1',file=path.name,SHA256=sha(path),bytes=path.stat().st_size,source_git=m.git_head(),source_SHA256=sha(__file__),data_SHA256=dm['sha256'],hardware_SHA256=HARDWARE,source_probe_report_SHA256=sha(source/'report.json'),source_probe_audit_SHA256=sha(source/'independent_audit.json'),initial_four_parameter_TRAIN_report_SHA256=sha(old_fit),fit_observations=216,fit_distinct_TRAIN_ids=len(fit_ids),other_TRAIN_observations=80,other_distinct_TRAIN_ids=len(other_ids),TEST_used=False,VALID=0,labels_used=False,SDK_used=False,hardware_mutated=False,rows=[{k:v for k,v in row.items() if k not in ('amp','phase','observed')} for row in rows])
    save(a.out/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='rows'}),flush=True)


def response(amplitude,turns,transfer,raw):
    import torch
    from torch.nn import functional as F
    span=.25+1.75*raw['span'].sigmoid()
    knots=torch.cat([raw['shape'].new_zeros(1),raw['shape'].softmax(0).cumsum(0)])*span
    coord=(turns*16).clamp(0,16-1e-5);i=coord.floor().long();f=coord-i
    phi=knots[i]*(1-f)+knots[i+1]*f
    mag=.5*raw['mag'].tanh();log_mag=mag[i]*(1-f)+mag[i+1]*f
    amplitude_phase=torch.pi*(raw['amplitude_phase'][0].tanh()*amplitude+raw['amplitude_phase'][1].tanh()*amplitude.square())
    field=amplitude*log_mag.exp()*torch.exp(1j*(2*torch.pi*phi+amplitude_phase))
    intensity=torch.fft.ifft2(torch.fft.fft2(F.pad(field,(20,20,20,20)))*transfer).abs().square()
    freq=torch.fft.fftfreq(518,device=field.device);sigma=12*raw['sigma'].sigmoid()
    blur=torch.exp(-2*torch.pi**2*sigma.square()*(freq[:,None].square()+freq[None,:].square()))
    out=torch.fft.ifft2(torch.fft.fft2(intensity)*blur).real.clamp_min(0)[:,20:498,20:498]
    return out+3*raw['background'].sigmoid()*out.mean((-2,-1),keepdim=True)


def initial(device,nominal=False):
    import torch
    q=torch.linspace(0,1,17,device=device);shape=q if nominal else q.pow(.7571307420730591)
    logit=lambda x:torch.logit(torch.tensor(x,device=device))
    return torch.nn.ParameterDict(dict(shape=torch.nn.Parameter((shape[1:]-shape[:-1]).log()),mag=torch.nn.Parameter(torch.zeros(17,device=device)),amplitude_phase=torch.nn.Parameter(torch.zeros(2,device=device)),span=torch.nn.Parameter(logit(((1. if nominal else .7914485931396484)-.25)/1.75)),sigma=torch.nn.Parameter(logit(1e-9 if nominal else 8.823136329650879/12)),background=torch.nn.Parameter(logit(1e-9 if nominal else 2.4428398609161377/3))))


def normalized_loss(pred,obs):
    import torch
    from torch.nn import functional as F
    p=pred/pred.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-8);o=obs/obs.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-8)
    return .5*(p-o).square().mean()+.5*(F.adaptive_avg_pool2d(p[:,None],(16,16))-F.adaptive_avg_pool2d(o[:,None],(16,16))).square().mean()


def selftest():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73);amp=torch.rand(2,478,478);phase=torch.rand_like(amp);freq=torch.fft.fftfreq(518);transfer=torch.exp(3j*(freq[:,None].square()+freq[None,:].square()))
    expected=torch.fft.ifft2(torch.fft.fft2(F.pad(amp*torch.exp(2j*torch.pi*phase),(20,20,20,20)))*transfer).abs().square()[:,20:498,20:498]
    raw=initial('cpu',True);actual=response(amp,phase,transfer,raw);assert torch.allclose(actual,expected,atol=5e-6,rtol=5e-6)
    raw=initial('cpu');loss=normalized_loss(response(amp,phase,transfer,raw),expected);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in raw.values())
    gradient=float(raw['span'].grad);h=.001;center=raw['span'].detach().clone()
    with torch.no_grad():
        raw['span'].copy_(center+h);plus=normalized_loss(response(amp,phase,transfer,raw),expected)
        raw['span'].copy_(center-h);minus=normalized_loss(response(amp,phase,transfer,raw),expected);raw['span'].copy_(center)
    finite=float((plus-minus)/(2*h));assert abs(finite-gradient)<=max(1e-4,.05*abs(gradient))
    print(dict(passed=True,nominal_maxabs=float((actual-expected).abs().max()),parameters=sum(v.numel() for v in raw.values()),span_derivative=gradient,span_finite_difference=finite,SDK_used=False,inference_changed=False),flush=True)


def fit(a):
    cfg=json.loads(a.profile.read_text());assert cfg['schema']=='TRAIN_coherent_response_fit_v1';os.environ['CUDA_VISIBLE_DEVICES']=cfg['GPU_UUID']
    import subprocess
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(73)
    assert not a.out.exists() and torch.cuda.is_available() and 'A100' not in torch.cuda.get_device_name(0)
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert all(v.split(',')[0].strip()!=cfg['GPU_UUID'] or int(v.split(',')[1])==os.getpid() for v in owners.splitlines())
    torch.cuda.set_per_process_memory_fraction(.6,0)
    manifest=json.loads(a.manifest.read_text());path=a.manifest.parent/manifest['file'];assert sha(path)==manifest['SHA256'] and not manifest['TEST_used'] and not manifest['labels_used']
    with np.load(path) as z:
        amplitude=torch.from_numpy(z['amplitude'].astype(np.float32)/255).cuda();turns=torch.from_numpy(z['phase'].astype(np.float32)/256).cuda();observed=torch.from_numpy(z['observed'].astype(np.float32)).cuda();transfer=torch.from_numpy(z['transfer']).cuda();mask=z['fit'];depth=z['depth']
    train=np.flatnonzero(mask);other=np.flatnonzero(~mask);assert len(train)==216 and len(other)==80
    a.out.mkdir();raw=initial('cuda');history=[];start=time.monotonic()
    protocol=dict(source_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),config=cfg,manifest_SHA256=sha(a.manifest),observations_SHA256=sha(path),TRAIN_response_frames=216,distinct_TRAIN_images=manifest['fit_distinct_TRAIN_ids'],other_TRAIN_response_frames=80,TEST_used=False,VALID=0,labels_used=False,new_classifier_PT=False,hardware_mutated=False,training_surrogate_only=True,original_deployed_graph_unchanged=True)
    save(a.out/'protocol.json',protocol)
    def metrics(ix):
        vals=[];pcc=[];coarse=[]
        with torch.no_grad():
            for begin in range(0,len(ix),4):
                j=ix[begin:begin+4];p=response(amplitude[j],turns[j],transfer,raw);o=observed[j];vals.append(float(normalized_loss(p,o))*len(j))
                for destination,x,y in ((pcc,p,o),(coarse,F.adaptive_avg_pool2d(p[:,None],(16,16))[:,0],F.adaptive_avg_pool2d(o[:,None],(16,16))[:,0])):
                    x=x.flatten(1);y=y.flatten(1);x=x-x.mean(1,keepdim=True);y=y-y.mean(1,keepdim=True);v=(x*y).sum(1)/(x.norm(dim=1)*y.norm(dim=1)).clamp_min(1e-12)
                    destination.extend(v.cpu().tolist())
        assert len(pcc)==len(coarse)==len(ix)
        return dict(normalized_loss=sum(vals)/len(ix),pixel_Pearson_median=float(np.median(pcc)),coarse16_Pearson_median=float(np.median(coarse)))
    baseline={str(d):metrics(other[depth[other]==d]) for d in (4,6)};best=float('inf');best_state=None;opt=torch.optim.Adam(raw.parameters(),lr=cfg['learning_rate'])
    for step in range(1,cfg['maximum_full_updates']+1):
        opt.zero_grad(set_to_none=True);value=0.
        for begin in range(0,216,cfg['microbatch']):
            j=train[begin:begin+cfg['microbatch']];loss=normalized_loss(response(amplitude[j],turns[j],transfer,raw),observed[j]);(loss*len(j)/216).backward();value+=float(loss.detach())*len(j)/216
        if value<best:best=value;best_state={k:v.detach().cpu().clone() for k,v in raw.items()}
        penalty=.002*(raw['mag'].square().mean()+raw['amplitude_phase'].square().mean());penalty.backward();torch.nn.utils.clip_grad_norm_(raw.parameters(),2);opt.step()
        row=dict(step=step,TRAIN_response_records=216,TRAIN_loss=value,elapsed_s=time.monotonic()-start);history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='fitting_TRAIN_complex_response',latest=row));print(json.dumps(row),flush=True)
        if time.monotonic()-start>=cfg['maximum_fit_seconds']:break
    final_train=metrics(train)
    if final_train['normalized_loss']<best:best=final_train['normalized_loss'];best_state={k:v.detach().cpu().clone() for k,v in raw.items()}
    np.savez_compressed(a.out/'last_response.npz',**{k:v.detach().cpu().numpy() for k,v in raw.items()})
    # Choose fit loss only, never use the other TRAIN-mask check to select a step.
    raw.load_state_dict(best_state);results={str(d):metrics(other[depth[other]==d]) for d in (4,6)}
    admitted=all(results[k]['normalized_loss']<=cfg['maximum_relative_other_TRAIN_loss']*baseline[k]['normalized_loss'] and results[k]['pixel_Pearson_median']>=max(cfg['minimum_pixel_PCC'],baseline[k]['pixel_Pearson_median']+cfg['minimum_PCC_gain']) and results[k]['coarse16_Pearson_median']>=max(cfg['minimum_coarse_PCC'],baseline[k]['coarse16_Pearson_median']+cfg['minimum_PCC_gain']) for k in baseline)
    np.savez_compressed(a.out/'best_response.npz',**{k:v.detach().cpu().numpy() for k,v in raw.items()})
    save(a.out/'report.json',dict(state='complete',TRAIN_fit_best_loss=best,baseline_other_TRAIN=baseline,fitted_other_TRAIN=results,response_admitted=admitted,admission_is_not_mask_or_model_accuracy=True,actual_updates=step,TEST_used=False,labels_used=False,SDK_used=False,hardware_mutated=False,original_deployed_graph_unchanged=True,response_PT_SHA256=sha(a.out/'best_response.npz'),protocol_SHA256=sha(a.out/'protocol.json'),next_mask_or_capture_started=False))
    print(json.dumps(dict(complete=True,response_admitted=admitted,baseline=baseline,fitted=results)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('pack','fit','selftest'),required=True);p.add_argument('--sequence',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);a=p.parse_args();selftest() if a.mode=='selftest' else pack(a) if a.mode=='pack' else fit(a)
