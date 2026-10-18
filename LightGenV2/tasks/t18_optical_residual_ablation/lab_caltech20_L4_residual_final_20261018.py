"""Deploy the explicitly frozen346/411 L4 residual model, TRAIN60/class.

Same existing inference, layer-major capture and original5140 readout. No optical
mask update in this first complete deployment; never reuse another model's CCD.
The first24 TRAIN frames are retained and signal-gated as part of full capture.
"""
import argparse,json,os,sys,traceback,time,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE),str(ROOT)]
import numpy as np
import torch
from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images,sha,DIRECT
from lab_caltech20_rich_layerwise_20261018 import load_data,MeasuredPrefix,allowed_keys,state_cpu,acquire_layer
from lab_caltech20_rich_capture_20261018 import layer_forward,metrics
from lab_caltech20_controls46_20261018 import capture_group,adapt_head,LOCK
from lab_caltech10_20261015 import save
PT_SHA='7b0d7f055a16f7f6bbf9858c9ecaea572ecd0b0d6e96c618630f4a4e077c09a5'


@torch.inference_mode()
def gate(a):
    torch.set_num_threads(2);datasets,dm=load_data(a.session)
    model,_=load_checkpoint(a.session/'residual_L4.pt',PT_SHA,4,.3,alpha_schema=DIRECT)
    reference_gate=json.loads((a.session/'portable_reference_gate.json').read_text());assert reference_gate['passed'] and reference_gate['PT_SHA256']==PT_SHA
    source_path=a.session/'source_reference.ts';assert sha(source_path)==reference_gate['TorchScript_SHA256']
    source_model=torch.jit.load(str(source_path),map_location='cpu').eval()
    # This non-persistent buffer is absent from the checkpoint. Windows/Linux
    # complex exp differ by one float32 ulp; quantized cascades amplify it.
    # Freeze the original server constant for simulation audit only. During
    # physical layer-major capture every propagation is replaced by real CCD.
    canonical=dict(source_model.named_buffers());buffer_differences={}
    for name,value in model.named_buffers():
        assert name in canonical and value.shape==canonical[name].shape
        difference=float((value-canonical[name]).abs().max());buffer_differences[name]=difference
        assert difference<1e-6 if name=='propagator.transfer' else difference==0.
        value.copy_(canonical[name])
    proof=json.loads((a.session/'server_bridge_gate.json').read_text());assert proof['passed'] and proof['correct']==346 and proof['PT_SHA256']==PT_SHA
    reference=a.session/'simulation_TEST_reference.npz';assert sha(reference)==proof['reference_SHA256']
    images,y,ids=datasets['test'];parts=[];source_error=0.
    for b in range(0,411,8):
        x=encode_images(images[b:b+8]);n=len(x)
        if n<8:x=torch.cat([x,torch.zeros(8-n,224,224)],0)
        value=model(x)[0];original=source_model(x)[0];source_error=max(source_error,float((value-original).abs().max()));parts.append(value[:n].numpy())
    assert source_error<=3e-6,'Full TEST same-host original-model/bridge parity is required'
    p=np.concatenate(parts)
    with np.load(reference,allow_pickle=False) as z:
        assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['labels']);err=float(np.abs(p-z['probabilities']).max());cross_host_disagreements=int((p.argmax(1)!=z['probabilities'].argmax(1)).sum())
    assert int((p.argmax(1)==y).sum())==346
    # Exercise the exact layer-major replay for the first TRAIN sample, no SDK.
    images,_,ids=datasets['train'];images=images[:1];ids=ids[:1];expected=model(encode_images(images))[0].numpy();previous={}
    for stage in ['router','0','1','2','3']:
        def simulated(current,amplitude,phase,sample_ids):
            field=torch.from_numpy(amplitude.copy())*torch.exp(2j*np.pi*torch.from_numpy(phase.astype(np.float32)/256))[None]
            value=model.propagator(torch.nn.functional.pad(field,(20,20,20,20))).abs().square()[:,20:498,20:498].numpy();previous[current]=value;return value
        actual=layer_forward(model,images,ids,stage,dict(previous),simulated)
    replay_error=float(np.abs(actual['probabilities']-expected).max());assert replay_error<=3e-6
    result=dict(passed=True,PT_SHA256=PT_SHA,TEST=411,correct=346,accuracy=346/411,full_TEST_same_host_source_model_error=source_error,portable_source_reference_SHA256=sha(source_path),reference_only_not_capture_model=True,canonical_propagation_buffer_used_for_audit_only=True,original_buffer_max_differences=buffer_differences,padded_samples_excluded=True,cross_host_probability_error=err,cross_host_prediction_disagreements=cross_host_disagreements,cross_host_bitwise_equivalence_claimed=False,layer_major_replay_error=replay_error,data_SHA256=dm['sha256'],TRAIN=1200,VALID=0,planned_CCD=8055,source_SHA256=sha(__file__),capture_source_SHA256=sha(HERE/'lab_caltech20_controls46_20261018.py'),SDK_opened=False,GPU_used=False)
    save(a.session/'lab_bridge_gate.json',result);print(json.dumps(result),flush=True)


def run(a):
    torch.set_num_threads(4);gateproof=json.loads((a.session/'lab_bridge_gate.json').read_text());assert gateproof['passed'] and gateproof['source_SHA256']==sha(__file__) and gateproof['capture_source_SHA256']==sha(HERE/'lab_caltech20_controls46_20261018.py')
    assert not LOCK.exists() and not (a.session/'controller.json').exists()
    datasets,dm=load_data(a.session);assert dm['sha256']==gateproof['data_SHA256']
    a.residual_datasets=datasets;a.residual_data_manifest=dm;a.dependency_kind='latest human: locked L4 residual346/411, L2 paused, no scheduled monitoring'
    save(a.session/'controller.json',dict(pid=os.getpid(),source_SHA256=sha(__file__),PT_SHA256=PT_SHA,planned_CCD=8055,TRAIN=1200,TEST=411,VALID=0,mode='fixed_optics_full_capture_then_existing_head_CPU',recurring_monitor=False))
    spec=dict(depth=4,rho=.3,alpha_schema=DIRECT,file='residual_L4.pt',sha256=PT_SHA,simulation_correct=346,simulation_total=411)
    model,data=capture_group(a,spec)
    fit=adapt_head(a.session/'L4',model,data)
    save(a.session/'report.json',dict(state='complete',capture=json.loads((a.session/'L4/capture_report.json').read_text()),readout=fit,SDK_released=not LOCK.exists(),TEST_is_development=True))


def layerwise_selftest():
    torch.set_num_threads(2);torch.manual_seed(17);ref=MeasuredPrefix(4).eval()
    with torch.no_grad():
        for p in ref.fusion_logits:p.fill_(.8)
        x=torch.rand(1,224,224);expected=ref(x)[0];fields={k:v['intensity'].clone() for k,v in ref.last.items()}
    records=[]
    for prefix in range(1,6):
        m=MeasuredPrefix(4).eval();m.load_state_dict(ref.state_dict());m.measured={k:fields[k] for k in ('router','0','1','2','3')[:prefix]};m.requires_grad_(False);keys=allowed_keys(m,prefix)
        for n,p in m.named_parameters():p.requires_grad_(n in keys)
        p=m(x)[0];assert torch.equal(p,expected)
        if prefix in (1,3,5):
            loss=-p[:,0].clamp_min(1e-12).log().mean();loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for n,v in m.named_parameters() if n in keys)
            assert all(v.grad is None for n,v in m.named_parameters() if n not in keys)
        assert 'raw_router' not in keys
        if prefix==5:assert keys=={'electronic.3.local.weight','fusion_logits.3','linear.weight','linear.bias'}
        records.append(dict(prefix=prefix,trainable=sum(v.numel() for n,v in m.named_parameters() if n in keys),allowed_keys=sorted(keys)))
    print(json.dumps(dict(passed=True,depth=4,exact_measured_prefix_replay=True,gradient_and_frozen_upstream=True,proof=records,SDK_opened=False,GPU_used=False)),flush=True)


def adopt_router(a,dm,datasets):
    out=a.session/'capture/router';assert not out.exists();handoff=json.loads((a.session/'layerwise_handoff_20261018.json').read_text());assert handoff['router_valid_frames']==1611 and handoff['SDK_owner_process_exited']
    phase=a.session/'L4/phase_router.bmp';assert sha(phase)==handoff['phase_SHA256'];out.mkdir(parents=True);os.link(phase,out/'phase.bmp');arrays={};receipt_hashes={}
    for role,n in (('train',1200),('test',411)):
        source=a.session/'L4/capture'/role/'router';dest=out/role;dest.mkdir();assert sha(source/'ccd.npy')==handoff['router_arrays_SHA256'][role];assert len(list(source.glob('[0-9]*.json')))==n
        for index in range(n):
            p=source/f'{index:06d}.json';r=json.loads(p.read_text());assert r['sample_id']==str(datasets[role][2][index]) and r['phase_SHA256']==sha(phase) and r['checkpoint_SHA256']==PT_SHA and r['p99']>=15 and r['saturation_fraction']<=.15;receipt_hashes[role+'/'+str(index)]=sha(p)
        os.link(source/'ccd.npy',dest/'ccd.npy');arrays[role]=dict(sha256=handoff['router_arrays_SHA256'][role],samples=n,original_receipts=str(source))
    save(out/'adoption_receipts.json',receipt_hashes)
    save(out/'report.json',dict(state='complete',stage='router',physical_ccd=1611,TRAIN=1200,TEST=411,VALID=0,sdk_released=True,arrays=arrays,data_sha256=dm['sha256'],checkpoint_sha256=PT_SHA,phase_sha256=sha(phase),reused_exact_original_router=True,new_capture_frames=0,original_data_retained=True,receipt_manifest_SHA256=sha(out/'adoption_receipts.json'),handoff_SHA256=sha(a.session/'layerwise_handoff_20261018.json')))


def layerwise(a):
    import psutil
    state=a.session/'layerwise_20261018';assert not state.exists() and not LOCK.exists();cfg=json.loads(a.profile.read_text());assert cfg['depth']==4 and cfg['TRAIN']==1200 and cfg['TEST']==411 and cfg['VALID']==0
    torch.set_num_threads(cfg['cpu_threads']);datasets,dm=load_data(a.session);gateproof=json.loads((a.session/'lab_bridge_gate.json').read_text());assert gateproof['passed'] and gateproof['PT_SHA256']==PT_SHA
    state.mkdir();receipt=state/'controller.json';generic=HERE/'lab_caltech20_rich_layerwise_20261018.py'
    save(receipt,dict(pid=os.getpid(),session=str(a.session.resolve()),source_SHA256=sha(generic),entry_SHA256=sha(__file__),profile_SHA256=sha(a.profile),actual_git=subprocess.check_output([str(ROOT/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe'),'rev-parse','HEAD'],cwd=ROOT,text=True).strip(),TRAIN=1200,TEST=411,VALID=0,depth=4,selection='First highest full CPU TEST development; TRAIN gradients only',order='Complete each TRAIN+TEST stage -> freeze captured upstream -> optimize remaining masks/electronics -> strictCPU best -> next fresh stage',reused_router_frames=1611,planned_new_frames=6444,recurring_monitor=False))
    adopt_router(a,dm,datasets);checkpoint=a.session/'residual_L4.pt';digest=PT_SHA;records=[];a.depth=4
    for index,stage in enumerate(('router','0','1','2','3'),1):
        if stage!='router':acquire_layer(a,stage,checkpoint,digest)
        target=state/'fits'/stage;target.parent.mkdir(exist_ok=True);save(state/'progress.json',dict(state='suffix_fitting',prefix=index,after_stage=stage,reused_router=1611,completed_capture_frames=index*1611))
        cli=[str(Path(sys.executable).with_name('python.exe')),str(generic),'--mode','fit','--depth','4','--device',a.device,'--session',str(a.session),'--parent',str(a.parent),'--profile',str(a.profile),'--prefix',str(index),'--checkpoint',str(checkpoint),'--checkpoint-sha',digest,'--out',str(target),'--controller-pid',str(os.getpid()),'--controller-receipt',str(receipt)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg['GPU_UUID'] if a.device=='cuda' else '')
        with target.with_suffix('.log').open('x',encoding='utf-8',buffering=1) as log:child=subprocess.run(cli,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        assert child.returncode==0,('Layerwise suffix failed; stop before next capture',stage,child.returncode)
        report=json.loads((target/'report.json').read_text());assert report['state']=='complete' and report['frozen_parameters_unchanged'] and report['results']['best']['strict_CPU']
        checkpoint=target/'best.pt';digest=report['results']['best']['PT_SHA256'];assert sha(checkpoint)==digest
        workers=[p.info['pid'] for p in psutil.process_iter(['pid','cmdline']) if p.info['cmdline'] and str(target) in ' '.join(p.info['cmdline']) and '--mode' in p.info['cmdline']];assert not workers,('Training child must exit before SDK',workers)
        records.append(dict(stage=stage,selected_PT_SHA256=digest,selected_epoch=report['selected_epoch'],TEST_development=report['results']['best']['TEST_development'],scope=report['evaluation_kind'],capture_report_SHA256=sha(a.session/'capture'/stage/'report.json'),fit_report_SHA256=sha(target/'report.json')));save(state/'stage_history.json',records)
    score=records[-1]['TEST_development']['accuracy'];save(state/'report.json',dict(state='complete',stages=records,physical_ccd=8055,reused_router=1611,new_CCD=6444,TRAIN=1200,TEST=411,VALID=0,final_PT=str(checkpoint),final_PT_SHA256=digest,final_full_physical_TEST_development=score,simulation_reference=346/411,gap_pp=100*(346/411-score),SDK_released=not LOCK.exists(),GPU_children_exited=True,TEST_gradients=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','run','layerwise','layerwise-test'),required=True)
    for n in ('session','parent','profile','coordinates','machine-config','phase-lut','response-lut','phase-sdk','amplitude-sdk'):p.add_argument('--'+n,type=Path)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    a=p.parse_args()
    if sys.stdout is None:sys.stdout=(a.session/('layerwise_controller.log' if a.mode=='layerwise' else 'controller.log')).open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    try:layerwise_selftest() if a.mode=='layerwise-test' else layerwise(a) if a.mode=='layerwise' else gate(a) if a.mode=='gate' else run(a)
    except BaseException as e:
        folder=a.session/'layerwise_20261018' if a.mode=='layerwise' else a.session
        if folder is not None and folder.exists():save(folder/('gate_failure.json' if a.mode=='gate' else 'failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
