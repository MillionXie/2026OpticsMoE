"""Pure CPU arithmetic/guard contracts; no Torch, model or GPU loads."""
import json
import csv
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from measure_lgvq_cost import field_plan,costs,proof_contract,live
from audit_lgvq_cost import expected_fields
from run_lgvq_cost import point_plan,previous_cost_finished


@pytest.mark.parametrize('target,fields,slots', [('temporal',13,208),('spatial',200,200)])
def test_independent_full_field_and_padding_contract(target,fields,slots):
    plan=field_plan(target)
    assert plan==expected_fields(target) and len(plan)==fields
    assert sum(sum(valid) for _,valid in plan)==200
    assert sum(len(valid) for _,valid in plan)==slots
    if target=='temporal':assert plan[-1]==(list(range(192,200))+list(range(8)),[True]*8+[False]*8)


def test_entire_front_counted_once_no_detail_double_count_or_residual_charge():
    plan=field_plan('temporal')
    fields=[dict(valid=valid,frontend_wall_ms=10,internal_gap_ms=0 if i==0 else 1) for i,(_,valid) in enumerate(plan)]
    stages=[dict(region='frontend',charged=True,synchronized_wall_ms=99999),
        dict(region='body',charged=True,synchronized_wall_ms=26),dict(region='body',charged=False,synchronized_wall_ms=900)]
    cost=costs('temporal',fields,stages)
    assert cost['charged_electronic_ms_total']==168
    assert cost['optical_ms_total']==pytest.approx(13*6*1.0447)
    assert cost['equivalent_frame_ms']==pytest.approx(cost['sample_ms']/4)
    assert cost['joules200']==pytest.approx((168*104.230+13*6*1.0447*121.746)/1000)
    with pytest.raises(ValueError):costs('temporal',fields[:-1],stages)


def test_missing_only_points_do_not_change_repeat_identity():
    assert point_plan(['spatial_repeat03','temporal_repeat02'])==[('spatial_repeat03','spatial',3),('temporal_repeat02','temporal',2)]
    for bad in ([],['temporal_repeat01']*2,['spatial_repeat04'],['qwen_repeat01']):
        with pytest.raises(ValueError):point_plan(bad)


def test_previous_cost_requires_exit_not_only_a_status_word():
    assert not previous_cost_finished(dict(status='running'),False)
    assert not previous_cost_finished(dict(status='complete'),True)
    assert previous_cost_finished(dict(status='complete'),False)
    assert previous_cost_finished(dict(status='stopped'),False)


def test_actual_cpu_proof_requires_all_four_original_inputs_and_models(tmp_path):
    checks=[dict(target=t,field_index=i,body_sha256=live.PINS[t][1],bitwise_equal=True,
        full_original_output_bitwise_equal=True,selected_NN_stage_instrumentation_tested=True,
        frontend_transformer_layers_executed=0,six_passes=list(range(6)))
        for t,i in [('temporal',13),('temporal',14),('spatial',200),('spatial',557)]]
    proof=dict(status='passed',no_gpu=True,cuda_initialized=False,manifest_sha256=live.MANIFEST_SHA,
        original_front_parameter_binding_sha256=live.BINDING_SHA,checks=checks,sources={})
    path=tmp_path/'contract.json';path.write_text(json.dumps(proof))
    assert proof_contract(path,check_sources=False)['status']=='passed'
    checks[0]['bitwise_equal']=False;path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):proof_contract(path,check_sources=False)


def test_worker_preserves_original_producer_full_forward_and_window_hash_order():
    source=(HERE/'measure_lgvq_cost.py').read_text()
    assert 'front.build(rows,args.target,meter)' in source and 'original.forward(model,moved)' in source
    assert source.index('after=file_records(tests,True)')>source.index("sampler.request('barrier')")
    assert 'original_FrameStem192' in source and 'explicit_warmup_forwards=0' in source
    assert 'common_input_tokens200=None' in source
    assert 'OpticalBoundary(' not in source and '.replay(' not in source


def test_independent_auditor_replays_schema_power_and_rejects_tail_changes(tmp_path,monkeypatch):
    """Synthetic unit fixture only: never a neural proof or measurement result."""
    import numpy as np
    import audit_lgvq_cost as audit
    from measure_lgvq_cost import sha,GPU,PROTOCOL
    root=tmp_path/'point';root.mkdir()
    def write(path,value):path.write_text(json.dumps(value),encoding='utf8')
    def csv_write(path,values):
        with path.open('w',newline='',encoding='utf8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    def seal():write(root/'files_manifest.json',[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))
        for p in root.iterdir() if p.name!='files_manifest.json'])
    ids=[f'video{i}' for i in range(200)];plan=field_plan('temporal');pin=dict(target='temporal',input_ids=[[1]],attention_mask=[[1]])
    binding=tmp_path/'binding.json';write(binding,dict(matched_front_tensors=[],matched_prompt_checks=[pin]))
    original_manifest=tmp_path/'test.csv'
    csv_write(original_manifest,[dict(sample_id=f'video{i}',split='test') for i in range(558)])
    monkeypatch.setattr(live,'BINDING',binding);monkeypatch.setattr(live,'BINDING_SHA',sha(binding))
    monkeypatch.setattr(live,'MANIFEST',original_manifest);monkeypatch.setattr(live,'MANIFEST_SHA',sha(original_manifest))
    monkeypatch.setattr(audit,'source_blobs',lambda *_:None);monkeypatch.setattr(audit,'mapped',lambda p:Path(p))
    shape=dict(vision_tokens=dict(shape=[1,16,4,49,1024],dtype='torch.float16',sha256='e'*64),
        quality_tokens=dict(shape=[1,16,4,49,14],dtype='torch.float16',sha256='f'*64))
    proofpath=tmp_path/'cpu.json';write(proofpath,dict(unit_fixture_not_actual_neural_proof=True))
    monkeypatch.setattr(audit,'proof_contract',lambda *_,**__:dict(checks=[dict(target='temporal',inputs=shape,source_ledger=[])]))
    refdir=tmp_path/'qualified';refdir.mkdir();refpath=refdir/'report.json'
    video_metadata={sid:dict(width=64,height=64,reported_frame_count=10) for sid in ids}
    write(refdir/'input_audit.json',dict(original_video_metadata=video_metadata))
    write(refpath,dict(input_files={sid:dict(sha256='a'*64,size_bytes=1) for sid in ids},input_audit_sha256=sha(refdir/'input_audit.json')))
    files={sid:dict(path='/fixture/'+sid,bytes=1,mtime_ns=0,ctime_ns=0,inode=0,sha256='a'*64) for sid in ids}
    identity=dict(target='temporal',protocol_id=PROTOCOL,system='ours',body_sha256=live.PINS['temporal'][1],
        manifest_sha256=live.MANIFEST_SHA,front_binding_sha256=live.BINDING_SHA,gpu_uuid=GPU,visible_multiprocessors=108,
        power_limit_w=250,mps_percent=100,batch=1,explicit_warmup_forwards=0,first_field_retained=True,frontend_transformer_layers_executed=0,
        cpu_contract=str(proofpath),cpu_contract_sha256=sha(proofpath),parameter_dtypes={'body':['torch.float32'],'native_patch_position':['torch.bfloat16']},
        original_front_tensors=[],native_prompt=pin,qualified_Qwen_input_reference=str(refpath),qualified_Qwen_report_sha256=sha(refpath),
        video_metadata_before={sid:{k:v for k,v in item.items() if k!='sha256'} for sid,item in files.items()},
        clock_anchor=dict(unix=0,monotonic=0),pid=1,repeat=1)
    views={sid:dict(**video_metadata[sid],Qwen_front_source_selected_positions=[round(9*(.1+i*.8/3)) for i in range(4)],
        Qwen_front_crop=.65,Qwen_front_size=448,metadata_read_after_T1=True) for sid in ids}
    fields=[];stages=[];csv_fields=[];predictions=[]
    for i,(indices,valid) in enumerate(plan):
        start=100+i*.1;front_end=start+.01;tick=front_end
        stages.append(dict(sample_index=i,start_monotonic=start,end_monotonic=front_end,synchronized_wall_ms=10,
            stage='frontend.native_patch',family='frontend_neural',charged=True,cuda_event_ms=1,region='frontend'))
        for name in ['input.original_CPU_features_H2D',*[f'prop{j}' for j in range(6)],'frame_merger','output.full_field_CPU_MOS']:
            end=tick+.001;fft=name.startswith('prop');output=name.startswith('output')
            stages.append(dict(sample_index=i,start_monotonic=tick,end_monotonic=end,synchronized_wall_ms=(end-tick)*1000,
                stage=name,family='optical_simulation_FFT' if fft else 'output' if output else 'neural',
                charged=not fft,cuda_event_ms='' if output else .1,region='body'));tick=end
        gap=0 if not fields else (start-fields[-1]['cpu_result_return'])*1000
        record=dict(field=i,test_indices=indices,valid=valid,source_ids=[ids[j] for j in indices],input_start=start,frontend_end=front_end,
            cpu_result_return=tick,frontend_wall_ms=(front_end-start)*1000,internal_gap_ms=gap,
            original_native_prompt=dict(input_ids=[[1]],attention_mask=[[1]],source_ids=[ids[j] for j in indices]),predictions=[1.]*16)
        fields.append(record);csv_fields.append({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in record.items()});predictions.extend([1.]*16)
    cost=costs('temporal',fields,stages);elapsed=fields[-1]['cpu_result_return']-fields[0]['input_start']
    write(root/'identity.json',identity);write(root/'status.json',dict(status='complete',valid_samples=200,fields=13))
    write(root/'summary.json',dict(status='complete',**cost,actual_software_wall_seconds=elapsed,diagnostic_software_board_j200=60*elapsed,
        weighted_software_board_w=60,common_input_tokens200=None))
    write(root/'input_files.json',files);write(root/'video_views.json',views);write(root/'prepared_inputs.json',[shape]*13)
    write(root/'stage_ledger.json',[]);write(root/'nn_module_shapes.json',[dict(field=i) for i in range(13)])
    platform=dict(uuid=GPU,name='NVIDIA A100-PCIE-40GB',power_limit_w=250,memory_total_mib=40960)
    write(root/'process_audit.json',[dict(unix=t,platform=platform,clients=[1],servers=[2],processes=[[2,GPU,'1']]) for t in (99.9,102.)])
    csv_write(root/'fields.csv',csv_fields);csv_write(root/'stages.csv',stages)
    csv_write(root/'telemetry.csv',[dict(host_monotonic_s=99.9+i*.05,watts=60) for i in range(50)])
    np.savez(root/'field_predictions.npz',predictions=np.asarray(predictions,np.float32),
        test_indices=np.asarray([j for indices,_ in plan for j in indices]),valid=np.asarray([v for _,valid in plan for v in valid]))
    seal();assert audit.audit(root)['valid_videos']==200
    csv_fields[-1]['valid']=json.dumps([True]*16);csv_write(root/'fields.csv',csv_fields);seal()
    with pytest.raises(ValueError,match='tail membership'):audit.audit(root)
