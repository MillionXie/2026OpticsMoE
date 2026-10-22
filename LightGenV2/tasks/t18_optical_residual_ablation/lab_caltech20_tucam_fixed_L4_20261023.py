"""One-shot L6 terminal CPU / fixed-weight fresh L4 optical experiment workflow.

Not a scheduled monitor. No changes to the already-running L6 capture source.
"""
import argparse,os,sys,time,traceback,subprocess,shutil
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
import lab_caltech20_dvp_layerwise_20261019 as m
import lab_caltech20_dvp_closedloop_20261021 as loop
from lab_caltech20_tucam_capture_20261023 import acquire,validate_hardware
from lab_caltech20_tucam_L6_20261022 import check_connection
from lab_caltech20_tucam_session_20261022 import Bench,logical
from lab_caltech20_tucam_exposure_20261023 import set_exposure
from lab_caltech20_tucam_terminal_20261023 import optimize

ROOT=Path(__file__).resolve().parents[3]


def original(a,cfg):
    assert m.sha(ROOT/cfg['L4_initial'])==cfg['L4_initial_SHA256'] and m.sha(ROOT/cfg['L4_reference'])==cfg['L4_reference_SHA256']
    data,dm=m.data_for(ROOT/cfg['sequence']);assert dm['sha256']==cfg['data_SHA256']
    model,_=m.model_for(ROOT/cfg['L4_initial'],cfg['L4_initial_SHA256'],4,ROOT/cfg['L4_reference'])
    return model,data,dm


def gate(a,cfg):
    out=a.out/'L4/admission';assert not out.exists();out.mkdir(parents=True)
    model,data,dm=original(a,cfg);reference=torch.jit.load(str(ROOT/cfg['L4_reference']),map_location='cpu').eval();parts=[];error=0
    with torch.inference_mode():
        for begin in range(0,411,8):
            images,n,end=m.padded(data,'test',begin);model.measured={};x=m.encode_images(images);p=model(x)[0];q=reference(x)[0];error=max(error,float((p-q).abs().max()));parts.append(p[:n].numpy());m.save(out/'progress.json',dict(TEST_completed=end,total=411,SDK_opened=False))
    score=int((np.concatenate(parts).argmax(1)==data['test'][1]).sum());assert score==cfg['L4_simulation_correct'] and error<=3e-6
    m.save(out/'report.json',dict(passed=True,simulation_correct=score,TEST=411,source_bridge_maxabs=error,PT_SHA256=cfg['L4_initial_SHA256'],reference_SHA256=cfg['L4_reference_SHA256'],data_SHA256=dm['sha256'],source_git=m.git_head(),SDK_opened=False,GPU_used=False,TEST_gradient=False))


def calibrate(model,data,dm,fields,paths,cfg,stage,out,parent,hardware):
    assert not out.exists() and not m.LOCK.exists();out.mkdir(parents=True);hw=validate_hardware(hardware);drive=np.array(m.read(hw['response_lut'])['drive'],np.uint8);patterns=out/'TRAIN40';patterns.mkdir();model.measured={};phase=None;rows=[]
    for begin in range(0,40,8):
        images,n,end=m.padded(data,'train',begin);cache={k:np.asarray(v['train'][begin:end]).copy() for k,v in fields.items()}
        def export(current,amplitudes,pixels,ids):
            nonlocal phase
            assert current==stage
            if phase is None:phase=m.phase_native(pixels);Image.fromarray(phase).save(out/'phase.bmp')
            assert np.array_equal(phase,m.phase_native(pixels))
            for j in range(8):
                i=begin+j;gray=np.rint(amplitudes[j].clip(0,1)*255).astype(np.uint8);p=patterns/f'{i:03d}.bmp';Image.fromarray(m.native(drive[gray],'amplitude')).save(p)
                rows.append(dict(index=i,id=str(ids[j]),label=int(data['train'][1][i]),BMP_SHA256=m.sha(p)))
            return np.zeros((8,478,478),np.uint8)
        m.layer_forward(model,images,data['train'][2][begin:end],stage,cache,export)
    assert len(rows)==40 and np.bincount(data['train'][1][:40],minlength=20).tolist()==[2]*20
    m.save(out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),TRAIN=40,TEST=0,VALID=0,rows=rows,PT_SHA256=m.sha(parent),data_SHA256=dm['sha256'],phase_SHA256=m.sha(out/'phase.bmp'),prefix_SHA256={k+'/'+r:m.sha(paths[k]/r/'ccd.npy') for k in fields for r in ('train','test')},parent_hardware_SHA256=m.sha(hardware),optical_parameters_unchanged=True,exposure_selection_not_accuracy_based=True))
    warp=cv2.getPerspectiveTransform(np.float32(m.corners(Path(hw['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]));m.save(m.LOCK,dict(pid=os.getpid(),kind='fixed_L4_TRAIN_exposure',out=str(out)));ladder=[];baseline=None;selected=None
    def capture(bench,bmp,folder,tag):
        raw,meta=bench.capture(bmp);v=logical(raw,warp,hw['sensor_full_scale'],hw['orientation']);native=cv2.warpPerspective(raw.astype(np.float32),warp,(478,478),flags=cv2.INTER_AREA);p=folder/(tag+'_raw.png');Image.fromarray(raw).save(p,compress_level=1)
        result=dict(p99=float(np.percentile(v,99)),saturation_fraction=float((v==255).mean()),native_mean=float(native.mean()),raw_SHA256=m.sha(p),metadata=meta);m.save(folder/(tag+'.json'),result);return result
    try:
        with Bench(hw,out) as bench:
            check_connection(bench,hw,warp,out);bench.stage(out/'phase.bmp');zero=out/'zero.bmp';Image.fromarray(m.native(np.zeros((478,478),np.uint8),'amplitude')).save(zero)
            for requested in cfg['exposure_probe_us']:
                actual=set_exposure(bench,requested);folder=out/f'exposure_{requested}';folder.mkdir();bg=capture(bench,zero,folder,'background');anchors=[capture(bench,patterns/f'{i:03d}.bmp',folder,str(i)) for i in cfg['TRAIN_exposure_anchors']];row=dict(requested_us=requested,actual_us=actual,background=bg,anchors=anchors)
                if baseline is None and min(v['p99'] for v in anchors)>=5:baseline=row
                if baseline is not None:
                    row['native_exposure_ratios']=[(v['native_mean']-bg['native_mean'])/max(b['native_mean']-baseline['background']['native_mean'],1e-9)/(actual/baseline['actual_us']) for v,b in zip(anchors,baseline['anchors'])]
                ladder.append(row);m.save(out/'exposure_ladder.json',ladder);m.save(out/'progress.json',dict(state='TRAIN_exposure_calibration',exposure_us=actual,p99=[v['p99'] for v in anchors]))
                if min(v['p99'] for v in anchors)>=40 and max(v['saturation_fraction'] for v in anchors)<=.001:
                    assert baseline is not None and max(abs(v-1) for v in row['native_exposure_ratios'])<=.2,'Exposure response outside registered tolerance';selected=row;break
                assert max(v['saturation_fraction'] for v in anchors)<=.15,'Clipping before acceptable exposure; preserve, no scan escalation'
            assert selected is not None,'No admissible bounded TRAIN exposure'
            validation=out/'validation40';validation.mkdir();records=[]
            for i in range(40):
                r=capture(bench,patterns/f'{i:03d}.bmp',validation,f'{i:03d}');records.append(r);m.save(out/'progress.json',dict(state='TRAIN40_validation',samples=i+1,p99=r['p99']))
                assert r['p99']>=15 and r['saturation_fraction']<=.15,'TRAIN calibration signal guard; no automatic retry'
            inherited={k:v for k,v in hw.items() if not k.startswith('same_phase_TRAIN_')}
            accepted={**inherited,'exposure_us':selected['actual_us'],'camera_device_info':{**bench.settings,'Exposure':selected['actual_us']},'stage_TRAIN_exposure_calibrated':True,'calibration_stage':stage,'calibration_PT_SHA256':m.sha(parent),'calibration_phase_SHA256':m.sha(out/'phase.bmp'),'connection_spot_reference_exposure_us':hw.get('connection_spot_reference_exposure_us',1999.8)}
        m.save(out/'admitted_hardware.json',accepted);m.save(out/'spatial_parent_proof.json',dict(parent_hardware=str(Path(hardware).resolve()),SHA256=m.sha(hardware),passed=True,geometry_not_refitted=True))
        m.save(out/'report.json',dict(passed=True,SDK_released=True,hardware_SHA256=m.sha(out/'admitted_hardware.json'),TRAIN=40,TEST_used=False,min_p99=min(r['p99'] for r in records),max_saturation=max(r['saturation_fraction'] for r in records),exposure_us=selected['actual_us'],coordinates_unchanged=True,device_LUTs_unchanged=True,network_background_subtraction=False,physical_calibration_frames=sum(1+len(v['anchors']) for v in ladder)+40+5))
    except BaseException as e:
        m.save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),all_data_preserved=True,automatic_retry=False));raise
    finally:assert m.read(m.LOCK)['pid']==os.getpid();m.LOCK.unlink()
    return out/'admitted_hardware.json'


def capture_L4(a,cfg):
    model,data,dm=original(a,cfg);frozen=m.state_cpu(model);fields={};paths={};parent=ROOT/cfg['L4_initial'];base_hw=ROOT/cfg['hardware'];assert m.sha(base_hw)==cfg['hardware_SHA256'];out=a.out/'L4';out.mkdir(exist_ok=True)
    assert m.read(out/'admission/report.json')['passed'] and not (out/'protocol.json').exists();assert shutil.disk_usage(out).free>cfg['minimum_storage_GiB']*1024**3
    m.save(out/'protocol.json',dict(config=cfg,source_git=m.git_head(),source_SHA256=m.sha(__file__),PT_SHA256=m.sha(parent),data_SHA256=dm['sha256'],interstage_fit=False,all_capture_parameters_frozen=True,new_physical_frames=10055,TRAIN=1600,TEST=411,VALID=0))
    for stage in cfg['L4_stages']:
        stage_out=out/('stage_'+stage);stage_out.mkdir();m.save(out/'progress.json',dict(state='calibrating_frozen_stage',stage=stage));hw=calibrate(model,data,dm,fields,paths,cfg,stage,stage_out/'TRAIN_calibration',parent,base_hw)
        m.save(out/'progress.json',dict(state='capturing_frozen_stage',stage=stage));cap=stage_out/'capture';fields=acquire(model,data,dm,fields,paths,hw,stage,cap,parent);paths[stage]=cap
        assert all(torch.equal(v,m.state_cpu(model)[k]) for k,v in frozen.items()),'Any parameter changed during fixed capture'
        m.save(stage_out/'frozen_state_audit.json',dict(passed=True,PT_SHA256=m.sha(parent),all_parameters_bitwise_frozen=True,interstage_updates=0,report_SHA256=m.sha(cap/'report.json')))
    m.save(out/'capture_report.json',dict(state='complete',physical_frames=10055,all_five_layers_audited=True,PT_SHA256=m.sha(parent),interstage_updates=0,SDK_released=not m.LOCK.exists()))
    fit={**cfg,'target':cfg['L4_target']};m.save(out/'progress.json',dict(state='all_captured_CPU_terminal_optimization'));optimize(model,data,fields,paths,fit,out/'terminal',ROOT/cfg['L4_reference'])
    m.save(out/'report.json',dict(state='complete',full_physical=True,terminal=m.read(out/'terminal/report.json'),SDK_released=not m.LOCK.exists()))


def optimize_L6(a,cfg):
    old=ROOT/cfg['L6_run'];assert m.read(old.with_suffix('.return.json'))['returncode']==0;full=m.read(old/'final_readout/report.json');assert full['strict_original_CPU'] and full['full_physical']
    data,dm=m.data_for(ROOT/cfg['sequence']);assert dm['sha256']==cfg['data_SHA256'];paths={k:Path(v['path']) for k,v in m.read(old/'L6/stage_5/capture/protocol.json')['prefix_reports'].items()};paths['5']=old/'L6/stage_5/capture';fields={}
    for stage,cap in paths.items():
        report=m.read(cap/'report.json');assert report['data_SHA256']==dm['sha256'];fields[stage]={}
        if (cap/'independent_audit.json').exists():assert m.read(cap/'independent_audit.json')['passed']
        for role,n in (('train',1600),('test',411)):
            p=cap/role/'ccd.npy';assert m.sha(p)==report['arrays'][role]['SHA256'];v=np.load(p,mmap_mode='r');assert v.shape==(n,478,478);fields[stage][role]=v
    assert list(fields)==['router','0','1','2','3','4','5'];parent=old/'final_readout/best.pt';reference=ROOT/cfg['sequence']/'L6_response_20261020/source_reference.ts';model,_=m.model_for(parent,full['results']['best']['PT_SHA256'],6,reference);captured,_=m.model_for(ROOT/cfg['sequence']/cfg['L6_capture_parent_relative'],cfg['L6_capture_parent_SHA256'],6,reference);loop.validate_state(m,model,m.state_cpu(captured),{'linear.weight','linear.bias'})
    optimize(model,data,fields,paths,{**cfg,'target':cfg['L6_target']},a.out/'L6_terminal',reference)


def workflow(a,cfg):
    assert not a.out.exists();a.out.mkdir(parents=True);m.save(a.out/'protocol.json',dict(config=cfg,source_git=m.git_head(),source_SHA256=m.sha(__file__),scheduler=False,automatic_restarts=False,CPU_threads_per_child=2));gate(a,cfg)
    old=ROOT/cfg['L6_run'];start=time.monotonic()
    while not (old/'L6/stage_5/capture/independent_audit.json').exists():
        if old.with_suffix('.return.json').exists():assert m.read(old.with_suffix('.return.json'))['returncode']==0,'Existing L6 failed; do not steal SDK/restart'
        assert time.monotonic()-start<7200,'Bounded dependency wait expired';m.save(a.out/'progress.json',dict(state='waiting_existing_L6_capture_audit',elapsed_s=time.monotonic()-start));time.sleep(15)
    audit=m.read(old/'L6/stage_5/capture/independent_audit.json');assert audit['passed'] and audit['matched_samples']==2011 and not m.LOCK.exists()
    def spawn(mode):
        log=(a.out/(mode+'.log')).open('x',encoding='utf-8');cmd=[sys.executable,str(Path(__file__).resolve()),'--mode',mode,'--profile',str(a.profile.resolve()),'--out',str(a.out.resolve())];p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0);m.save(a.out/(mode+'.launch.json'),dict(pid=p.pid,args=cmd,source_git=m.git_head(),SDK=mode=='capture',GPU=False));return p,log
    capture,clog=spawn('capture');m.save(a.out/'progress.json',dict(state='L4_capture_started_waiting_L6_base_head'))
    while not old.with_suffix('.return.json').exists():
        assert capture.poll() in (None,0),'Fresh L4 capture failed; preserve and diagnose';assert time.monotonic()-start<10800;time.sleep(10)
    assert m.read(old.with_suffix('.return.json'))['returncode']==0;fit,flog=spawn('optimize6');m.save(a.out/'progress.json',dict(state='L4_capture_parallel_L6_CPU_optimization'))
    while capture.poll() is None or fit.poll() is None:
        for p in (capture,fit):assert p.poll() in (None,0),'A bounded child failed; no restart, other independent child is preserved'
        time.sleep(15)
    clog.close();flog.close();assert capture.returncode==fit.returncode==0
    m.save(a.out/'report.json',dict(state='complete',L4=m.read(a.out/'L4/report.json'),L6=m.read(a.out/'L6_terminal/report.json'),SDK_released=not m.LOCK.exists(),all_old_data_preserved=True))


def selftest():
    from lab_caltech20_rich_tail_refine_20261018 import selftest as tail_test
    tail_test();assert ROOT.name=='2026OpticsMoE'
    from lab_caltech20_tucam_capture_20261023 import name
    assert [name(k) for k in ('router','0','1','2','3','4','5')]==['router','E1','G1','E2','G2','E3','G3']
    from lab_caltech20_tucam_terminal_20261023 import tail,allowed
    from torch.nn import functional as F
    for depth in (4,6):
        model=m.MeasuredPrefix(depth).eval();model.requires_grad_(False)
        with torch.no_grad():
            for p in model.fusion_logits:p.fill_(.8)
        held={};hook=model.electronic[-1].register_forward_pre_hook(lambda module,args:held.update(x=args[0].detach()))
        try:
            with torch.inference_mode():expected=model(torch.rand(8,224,224)*.25)[0]
        finally:hook.remove()
        inp=held['x'].clone();ccd=model.readout['ccd'].clone();expected=expected.clone();optical=model.nonlinear(ccd,'global')
        with torch.no_grad():actual=tail(model.electronic[-1],model.linear,model.effective_alpha(depth-1),inp,optical).softmax(1)
        assert float((expected-actual).abs().max())<=3e-6
        for n,p in model.named_parameters():p.requires_grad_(n in allowed(model))
        F.cross_entropy(tail(model.electronic[-1],model.linear,model.effective_alpha(depth-1),inp,optical),torch.arange(8)).backward()
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==5150
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    print('Original terminal gradient and depth-independent stage contracts passed; SDK unopened')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('workflow','capture','optimize6','selftest'),required=True);p.add_argument('--profile',type=Path,default=Path(__file__).with_name('lab_caltech20_tucam_fixed_L4_profile_20261023.json'));p.add_argument('--out',type=Path);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(73)
    if a.mode=='selftest':selftest();sys.exit(0)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';cfg=m.read(a.profile);assert cfg['schema']=='TUCam_fixed_simulation_L4_after_L6_v1' and cfg['interstage_fit'] is False
    try:
        {'workflow':workflow,'capture':capture_L4,'optimize6':optimize_L6}[a.mode](a,cfg);m.save(a.out/(a.mode+'.return.json'),dict(returncode=0))
    except BaseException as e:
        if a.out.exists():m.save(a.out/(a.mode+'.return.json'),dict(returncode=1,error=repr(e),traceback=traceback.format_exc(),all_data_preserved=True,automatic_retry=False))
        raise
