"""Check the approved geometry, OEO semantics, gradients, and paired inputs."""
import time,json,yaml
import numpy as np
import torch
from torch.nn import functional as F
from experiments import ROOT,EXP,VARIANTS,CONFIGS,parameters
from models import build,objective
from optical_reference.activations import reference_activate, CANDIDATES
from optical_reference.experts import StageGlobalOEO
from run_experiment import setup,getdata,setseed,snapshot,save,now,evaluate,sha_tensor,epoch_order

def preflight():
    setup();train=getdata('train');val=getdata('val');x,y=train[0][:8],train[1][:8]
    report={'time':now(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'cuda':torch.version.cuda,'source_hashes':snapshot(),'data_sha256':EXP['data_sha256'],
            'train_n':len(train[1]),'val_n':len(val[1]),'test_arrays_read':False,'variants':{}}
    masks=[];init_pairs={};expected_work_seconds=0.
    for v in VARIANTS:
        cfg=CONFIGS[v['id']];depth=v['depth'];cycles=depth//2
        side=min(range(2,499,2),key=lambda s:abs(depth*s*s-cfg['experiment']['expected_parameters']['moe']))
        assert side==cfg['experiment']['baseline_phase_size']
        setseed(17);model=build(v['architecture'],cfg).cuda();masks.append(model.masks.detach().cpu())
        nparam=sum(p.numel() for p in model.parameters() if p.requires_grad);assert nparam==parameters(v)
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        assert names and all('raw_phase' in n for n in names)
        assert len(names)==(1+10*cycles if v['architecture']=='moe' else depth)
        initial={n:sha_tensor(p) for n,p in model.named_parameters()};key=(v['architecture'],depth)
        reference=yaml.safe_load((ROOT/'reference_initialization.yaml').read_text())
        assert initial==reference[f"{v['architecture']}_L{depth}_seed17"],'Initial phase tensors differ from the historical same-depth ReLU run'
        if key in init_pairs:assert initial==init_pairs[key]
        else:init_pairs[key]=initial
        panel=torch.linspace(0,2,498*498,device='cuda').reshape(1,498,498).to(torch.complex64)
        block=StageGlobalOEO(cfg['nonlinearity'],0,498,1).cuda()
        encoded,detail=block([panel],[True],capture_fields=True)
        intensity=panel.abs().square()
        mean=intensity.mean((-2,-1),keepdim=True)
        var=(intensity-mean).square().mean((-2,-1),keepdim=True)
        expected=reference_activate((intensity-mean)/torch.sqrt(var+1e-6),cfg['nonlinearity']['activation']['type'])
        assert torch.allclose(encoded[0].real,expected,rtol=1e-5,atol=1e-6)
        assert torch.count_nonzero(encoded[0].imag)==0
        assert torch.isfinite(encoded[0]).all() and (encoded[0].real>=0).all()
        zero=block([torch.zeros_like(panel)],[True])[0][0]
        expected_zero=reference_activate(torch.zeros_like(zero.real),cfg['nonlinearity']['activation']['type'])
        assert torch.allclose(zero.real,expected_zero)
        del panel,block,encoded,detail,intensity,mean,var,expected,zero
        records=[];handles=[]
        def hook(module,args,output):records.append(output[1])
        for module in model.modules():
            if isinstance(module,StageGlobalOEO):handles.append(module.register_forward_hook(hook))
        with torch.no_grad():
            original=model(x)
            if v['architecture']=='moe':
                assert len(records)==(depth if v['oeo'] else 0)
                for detail in records:
                    assert detail['elementwise_affine'] is False
                    assert detail['routing_amplitude_reapplied'] is False
                assert original['stage_input_power'].shape==(8,cycles,9)
            else:
                # Independently replay D2NN with the exact MoE OEO module.
                field=F.pad(x[:,0],(model.pad,)*4).to(torch.complex64)
                ref_oeo=StageGlobalOEO(cfg['nonlinearity'],0,498,1).cuda()
                for phase,prop in zip(model.phases,model.propagators):
                    field=prop(phase(field))
                    if v['oeo']:
                        active=field[:,model.a:model.b,model.a:model.b]
                        output,detail=ref_oeo([active],[True],capture_fields=True)
                        assert torch.equal(output[0].real,detail['activation'][:,0])
                        assert torch.allclose(output[0].real,reference_activate(F.layer_norm(active.abs().square(),(498,498),eps=1e-6),cfg['nonlinearity']['activation']['type']),rtol=1e-6,atol=1e-7)
                        assert torch.count_nonzero(output[0].imag)==0
                        field=field.clone();field[:,model.a:model.b,model.a:model.b]=output[0]
                field=model.camera(field)
                assert torch.allclose(original['intensity'],field.abs().square(),rtol=1e-6,atol=1e-7)
                if not v['oeo']:assert torch.count_nonzero(field.imag)>0
                del ref_oeo,field
        for h in handles:h.remove()
        opt=torch.optim.Adam(model.parameters(),lr=EXP['lr']);times=[];norms=None
        torch.cuda.reset_peak_memory_stats()
        for step in range(5):
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            out=model(x);loss=objective(out,y,model.masks);assert torch.isfinite(loss);loss.backward()
            norms={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters() if p.requires_grad}
            assert all(value is not None and np.isfinite(value) and value>0 for value in norms.values()),(v,norms)
            opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
        result,_=evaluate(model,val)
        item={'parameters':nparam,'phase_planes':len(names),'main_depth':depth,'cycles':cycles if v['architecture']=='moe' else None,
              'oeo_enabled':v['oeo'],'oeo_math_or_bypass_verified':True,'gradient_norms':norms,
              'seconds_per_batch':float(np.mean(times[2:])),'peak_memory_bytes':torch.cuda.max_memory_allocated(),
              'smoke_val':result,'identical_initialization_to_historical_same_depth_relu':True}
        if v['architecture']=='moe':
            assert out['routing'].shape==(8,9) and out['routing'].all()
            assert result['routing']['hard_counts']==[98]*9
            with torch.no_grad():
                _,physical=model.net(x,return_intermediates=True,capture_expert_outputs=False)
                fractions=physical['expert_entrance_energy_ratios']
                assert (fractions>0).all() and torch.allclose(fractions.sum(1),torch.ones(8,device=x.device),atol=1e-6)
                assert torch.allclose(physical['expert_entrance'].abs().square().sum((1,2)),x.square().sum((1,2,3)),rtol=1e-5)
                assert len(physical['expert_stage_details'])==cycles
                for stage in physical['expert_stage_details']:
                    assert (stage['linear_input_power']>0).all() and (stage['linear_output_power']>0).all()
                item.update(entrance_power_fractions=fractions.mean(0).cpu().tolist(),fanout_power_conserved=True,
                            all_cycles_all_expert_input_power_positive=True)
                del physical
        report['variants'][v['id']]=item
        expected_work_seconds+=len(EXP['seeds'])*EXP['epochs']*149*item['seconds_per_batch']
        print(json.dumps({'preflight':v['id'],'parameters':nparam,'seconds_per_batch':item['seconds_per_batch'],
                          'all_phase_gradients_positive':True,'peak_memory_gb':item['peak_memory_bytes']/1e9}),flush=True)
        del model,opt,out,original,loss,records;torch.cuda.empty_cache()
    assert all(torch.equal(masks[0],m) for m in masks)
    assert masks[0].sum((1,2)).tolist()==[1024.,1024.] and (masks[0].sum(0)<=1).all()
    assert torch.equal(epoch_order(17,1,len(train[1])),epoch_order(17,1,len(train[1])))
    report['estimated_training_only_seconds']=expected_work_seconds
    save(ROOT/'protocol/preflight.json',report)
    print(json.dumps({'preflight_completed':len(VARIANTS),'estimated_training_only_hours':expected_work_seconds/3600}),flush=True)
