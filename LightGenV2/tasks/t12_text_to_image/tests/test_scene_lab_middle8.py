"""Synthetic contracts only: no new production PT, camera, SDK or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from LightGenV2.tasks.t12_text_to_image import scene_lab_middle8 as m
from LightGenV2.tasks.t12_text_to_image.scene_lab_middle8_runtime import ComposedObserver,matched_training_inputs


def binding():
    config=dict(schema='scene_expanded_middle_v1',prompt_profile=m.PROFILE['prompt_profile'],
        vision_keep=6,language_keep=4,UNet_deep_keep=9,student_steps=8,
        parent_generator_sha256='a'*64,parent_frontend_sha256=m.F6,
        electronic_teacher_sha256='b'*64,pairs_sha256=m.PAIRS,TRAIN_contexts=624)
    return dict(schema='scene_lab_middle8_binding_v1',profile=copy.deepcopy(m.PROFILE),
        generator_sha256='c'*64,source_commit='d'*40,compression_profile=config,
        SIM_manifest_sha256='e'*64,approved_for_physical_deployment=True)


def full_binding():
    b=binding();b['schema']='scene_lab_full_unet8_binding_v1';b['profile']=copy.deepcopy(m.FULL_PROFILE)
    b['compression_profile'].update(schema='scene_expanded_full_UNet8_v1',UNet_deep_keep=10)
    return b


def rows():
    def row(i,role):
        view=i%(76 if role=='train' else 26)
        digest=hashlib.sha256((role+str(view)).encode()).hexdigest()
        r=dict(sample_id=role+str(i),reference=role+'/'+str(view)+'.png',prompt='One subject. Scene '+str(i)+'.',
            seed=i,reference_id=role+str(view),reference_sha256=digest,category='lamp'if i%2==0 else'pillow',
            condition_id=str(i%20),native_caption='native',reward_caption='reward',prompt_profile=m.PROFILE['prompt_profile'])
        if role=='train':r.update(split='train',source_product_id='product'+str(view%44),target='never_read.png')
        return r
    return [row(i,'train')for i in range(624)],[row(i,'test')for i in range(500)]


def metadata():
    report=dict(schema='scene_expanded_train_only_pairs_v1',complete=True,eligible_for_training=True,
        pairs_sha256=m.PAIRS,request_sha256=m.TEST,validation_set=False,TEST_used_for_model_selection=True,
        TEST_target_images=0,TRAIN_pairs=624,TEST_requests=500)
    selection=dict(gradient_roles=['train'],validation_set=False,TEST_used_for_model_selection=True,
        test_count=500,test_request_sha256=m.TEST,complete_cohort_required=True)
    return report,selection


def state():
    b=binding();descriptor=dict(schema='scene_compact_recovered_v1',profile=b['compression_profile'],sampling_steps=8,
        original_UNet_tensor_names_restored=True,active_graph_requires_explicit_depth_loader=True)
    s=dict(schema='qwen3_scene_paired_v1',variant='qwen',Qwen_input_side=256,resolution=512,
        SDXL_conditioning_size=1024,joint_image_context=False,pair_sha256=m.PAIRS,no_validation_test_PT_selection=True,
        optical_frontend=dict(checkpoint_sha256=m.F6),depth_compression=descriptor,
        sampling_contract=dict(kind='progressive_guided_Euler_CFGpreserved_v1',steps=8,teacher_steps=30,
            CFG=7.5,timesteps=list(m.PROFILE['timesteps']),IP_scale=.25))
    r=dict(complete=True,best_sha256=b['generator_sha256'],compression_training_performed=True,depth_compression=descriptor)
    return s,r,b


def test_import_and_data_help_need_no_torch_or_sdk():
    code="import sys; import LightGenV2.tasks.t12_text_to_image.scene_lab_middle8_runtime; assert 'torch'not in sys.modules; assert not any('tucam'in s.lower()for s in sys.modules)"
    subprocess.run([sys.executable,'-c',code],check=True)


def test_exact_profile_accepts_metadata_but_draft_never_deploys():
    b=binding();assert m.checked_binding(b)==b
    b['approved_for_physical_deployment']=False
    assert m.checked_binding(b,require_approval=False)==b
    with pytest.raises(ValueError):m.checked_binding(b)


def test_capture_side_pure_compression_validator_matches_source_contract():
    from LightGenV2.tasks.t12_text_to_image.scene_expanded_middle import checked_profile
    config=binding()['compression_profile']
    assert m.checked_compression_profile(config)==checked_profile(config)
    for key,value in [('student_steps',12),('UNet_deep_keep',10),('vision_keep',4),('schema','old_R2')]:
        changed=dict(config,**{key:value})
        with pytest.raises(ValueError):m.checked_compression_profile(changed)
        with pytest.raises(ValueError):checked_profile(changed)


def test_full70_has_its_own_schema_and_cannot_masquerade_as64():
    from LightGenV2.tasks.t12_text_to_image.scene_expanded_middle import checked_profile
    b=full_binding();m.checked_binding(b)
    assert m.checked_compression_profile(b['compression_profile'])==checked_profile(b['compression_profile'])
    for changed in [dict(b,schema='scene_lab_middle8_binding_v1'),dict(b,profile=binding()['profile']),
                    dict(b,compression_profile=binding()['compression_profile'])]:
        with pytest.raises(ValueError):m.checked_binding(changed)
    s,r,_=state();s['depth_compression']['profile']=b['compression_profile']
    assert m.checked_state(s,r,b)==s['depth_compression']


@pytest.mark.parametrize('key,value',[('joint_image_context',True),('steps',12),('UNet_blocks',70),('CFG',1.),('resolution',1024)])
def test_graph_or_sampling_changes_are_rejected(key,value):
    b=binding();b['profile'][key]=value
    with pytest.raises(ValueError):m.checked_binding(b)


def test_old_R2_manifest_and_placeholder_SHA_rejected():
    for key,value in [('TRAIN_contexts',132),('pairs_sha256','a'*64)]:
        b=binding();b['compression_profile'][key]=value
        with pytest.raises(ValueError):m.checked_binding(b)
    b=binding();b['generator_sha256']=None
    with pytest.raises(ValueError):m.checked_binding(b)


def test_checkpoint_metadata_requires_exact_depth_and_joint_flag():
    s,r,b=state();assert m.checked_state(s,r,b)==s['depth_compression']
    for change in [dict(joint_image_context=True),dict(SDXL_conditioning_size=512),dict(pair_sha256='a'*64)]:
        with pytest.raises(ValueError):m.checked_state(dict(s,**change),r,b)
    s['sampling_contract']['timesteps'][1]=892
    with pytest.raises(ValueError):m.checked_state(s,r,b)


def test_wrong_PT_fails_before_any_torch_load(tmp_path):
    from LightGenV2.tasks.t12_text_to_image.scene_lab_middle8_runtime import load_head
    p=tmp_path/'NOT_A_MODEL.pt';p.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError,match='SHA'):load_head(p,{},binding())


def test_full_dataset_rows_are_target_free_and_role_explicit():
    a,b=rows();report,selection=metadata();result=m.checked_rows(a,b,report,selection)
    assert len(result['train'])==624 and len(result['test'])==500
    assert set(result['test'][0])=={'sample_id','reference','reference_sha256','prompt','seed','category','split'}
    assert result['test'][0]['split']=='test'
    assert all('target'not in r for role in result.values()for r in role)


def test_TEST_target_leak_duplicate_source_overlap_and_unsafe_paths_rejected():
    a,b=rows();r,s=metadata()
    bad=copy.deepcopy(b);bad[0]['target']='secret.png'
    with pytest.raises(ValueError):m.checked_rows(a,bad,r,s)
    bad=copy.deepcopy(b);bad[-1]=bad[0]
    with pytest.raises(ValueError):m.checked_rows(a,bad,r,s)
    bad=copy.deepcopy(b);bad[0]['reference_sha256']=a[0]['reference_sha256']
    with pytest.raises(ValueError):m.checked_rows(a,bad,r,s)
    for path in ['../secret.png','C:/secret.png','/absolute.png','..\\secret.png']:
        bad=copy.deepcopy(b);bad[0]['reference']=path
        with pytest.raises(ValueError):m.checked_rows(a,bad,r,s)
    s['gradient_roles']=['train','test']
    with pytest.raises(ValueError):m.checked_rows(a,b,r,s)


def test_real_byte_identity_is_required_for_check_data(tmp_path):
    (tmp_path/'pairs.json').write_text('[]')
    with pytest.raises(ValueError):m.check_data(tmp_path)


def test_TEST_features_never_supply_head_calibration():
    a,b=rows();requests=m.public_requests(a,'train')
    measured=dict(schema='scene_lab_middle8_measured_features_v1',evaluation_only=False,
        all_six_stages_measured=True,TEST_gradients=False,binding=binding(),records=requests)
    nominal=dict(schema='scene_lab_middle8_nominal_conditions_v1',binding=binding(),pair_sha256=m.PAIRS,TRAIN_only=True,
        records=[dict(r,input_key=r['sample_id'])for r in requests])
    assert len(matched_training_inputs(measured,nominal))==624
    measured['records']=m.public_requests(b,'test');measured['evaluation_only']=True
    with pytest.raises(ValueError):matched_training_inputs(measured,nominal)


def test_test_reference_products_cannot_overlap_train_products():
    a,b=rows();r,s=metadata();requests=m.checked_rows(a,b,r,s);references={}
    for role,group in requests.items():
        for row in group:
            view=int(row['reference'].split('/')[-1].split('.')[0])
            references[row['reference_sha256']]=dict(reference=row['reference'],reference_sha256=row['reference_sha256'],
                reference_id=role+str(view),split=role,
                source_product_id=('product'+str(view%44))if role=='train'else('testproduct'+str(view%18)))
    m.check_reference_roles(list(references.values()),a,requests)
    references[b[0]['reference_sha256']]['source_product_id']='product0'
    with pytest.raises(ValueError):m.check_reference_roles(list(references.values()),a,requests)


def test_deployment_environment_rejects_A100_before_loading_torch(monkeypatch):
    from LightGenV2.tasks.t12_text_to_image.scene_lab_middle8_runtime import deployment_environment
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55')
    monkeypatch.setenv('T12_SOURCE_COMMIT','a'*40)
    with pytest.raises(ValueError):deployment_environment()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d')
    deployment_environment()
    monkeypatch.delenv('T12_SOURCE_COMMIT')
    with pytest.raises(ValueError):deployment_environment()


def test_render_cli_cannot_inherit_simulation_GPU_or_sampling_overrides():
    from LightGenV2.tasks.t12_text_to_image.scene_lab_middle8_runtime import render_arguments
    args=[]
    for k in ('data','checkpoint','qwen','cache','model-path','adapter-path','output','lg-frontend','lg-features','prompt-revision'):
        args += ['--'+k,'fixture/'+k]
    args+=['--fuse-task-lora']
    a=render_arguments(args)
    assert a.steps==8 and a.ip_scale==.25 and a.unet_precision=='bf16' and not a.measure_software_latency
    for change in [['--steps','12'],['--unet-precision','fp16'],['--probe-conditioning-side','512']]:
        with pytest.raises(SystemExit):render_arguments(args+change)


def test_graph_and_actual_CCD_observers_are_composed_in_order():
    log=[]
    class Graph:
        graph={'backend':{'actual_blocks':64}}
        def prepare(self,q,p,h,pipe):log.append('depth');return p+'-depth'
        def begin(self,row):log.append('depth_begin')
        def end(self,row):log.append('depth_end')
    class Actual:
        def prepare(self,q,p,h,pipe):log.append('actual');return p+'-actual'
        def begin(self,row):log.append('actual_begin')
        def end(self,row):log.append('actual_end')
        def close(self):log.append('actual_close')
    combined=ComposedObserver(Graph(),Actual())
    assert combined.prepare(None,'processor',None,None)=='processor-depth-actual'
    combined.begin({});combined.end({});combined.close()
    assert log==['depth','actual','depth_begin','actual_begin','actual_end','depth_end','actual_close']
    combined.depth.graph={'backend':{'actual_blocks':70}}
    with pytest.raises(ValueError):combined.prepare(None,'p',None,None)


def test_new_ledger_explicit_TEST_and_old_F9_rejection(tmp_path):
    from LightGenV2.tasks.t12_text_to_image.tests.test_scene_lab_protocol import fixture
    from LightGenV2.tasks.t12_text_to_image.scene_lab_protocol import Ledger,open_ledger
    old,_,_=fixture(tmp_path/'old_fixture');a,b=rows();requests=m.public_requests(b,'test');bind=binding()
    session=dict(old.session,cohort_role='test',request_manifest_sha256=m.TEST,TEST_gradients=False,
                 evaluation_only=True,TEST_used_for_model_selection=True)
    models=dict(F6=m.F6,generator=bind['generator_sha256'],source_commit=bind['source_commit'])
    with pytest.raises(ValueError):Ledger(tmp_path/'rejected',session,old.models,requests,create=True)
    new=Ledger(tmp_path/'explicit_test_fixture',session,models,requests,create=True,middle8_binding=bind)
    assert new.contract['schema']=='scene_lab_middle8_ledger_v1'
    assert open_ledger(new.root).key==new.key
    assert open_ledger(old.root).key==old.key
    path=new.root/'contract.json';doc=json.loads(path.read_text());doc.pop('middle8_binding');path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):open_ledger(new.root)


def test_full70_ledger_cannot_be_relabelled_as64(tmp_path):
    from LightGenV2.tasks.t12_text_to_image.tests.test_scene_lab_protocol import fixture
    from LightGenV2.tasks.t12_text_to_image.scene_lab_protocol import Ledger,open_ledger
    old,_,_=fixture(tmp_path/'old_fixture');_,b=rows();bind=full_binding()
    session=dict(old.session,cohort_role='test',request_manifest_sha256=m.TEST,TEST_gradients=False,
                 evaluation_only=True,TEST_used_for_model_selection=True)
    models=dict(F6=m.F6,generator=bind['generator_sha256'],source_commit=bind['source_commit'])
    new=Ledger(tmp_path/'full70_fixture',session,models,m.public_requests(b,'test'),create=True,middle8_binding=bind)
    assert new.contract['schema']=='scene_lab_full_unet8_ledger_v1' and open_ledger(new.root).key==new.key
    p=new.root/'contract.json';doc=json.loads(p.read_text());doc['schema']='scene_lab_middle8_ledger_v1';p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):open_ledger(new.root)


def test_actual_CPU_module_counts_for64_and70_not_production_weights():
    if sys.platform=='win32':pytest.skip('Run actual Torch CPU module-count test in the designated compute runtime')
    import torch
    from torch import nn
    from LightGenV2.tasks.t12_text_to_image.scene_depth_probe import prune_unet
    class Group(nn.Module):
        def __init__(self,count):
            super().__init__();self.transformer_blocks=nn.ModuleList([nn.Linear(1,1)for _ in range(count)])
    class Fixture(nn.Module):
        def __init__(self):
            super().__init__();self.groups=nn.ModuleList([Group(n)for n in [2]*5+[10]*6])
    for depth,expected in [(9,64),(10,70)]:
        fixture=Fixture();before={n:p.clone()for n,p in fixture.named_parameters()}
        graph=prune_unet(fixture,depth)
        assert graph['actual_blocks']==expected
        if depth==10:
            assert graph['removed_parameters']==0
            assert all(torch.equal(v,before[n])for n,v in fixture.named_parameters())


def test_native_adapter_requires_new_explicit_binding_not_false_joint_default():
    from LightGenV2.tasks.t12_text_to_image.scene_lab_native import NativeRunner
    class Frozen:
        joint_image_context=False
        model=SimpleNamespace(hybrid={'vision':SimpleNamespace(pairs=1),'language':SimpleNamespace(pairs=1)})
        def parameters(self):return iter(())
    b=binding();module=Frozen()
    with pytest.raises(ValueError):NativeRunner(module,None,module,'.',frontend_sha256=m.F6,head_sha256=b['generator_sha256'])
    r=NativeRunner(module,None,module,'.',frontend_sha256=m.F6,head_sha256=b['generator_sha256'],middle8_binding=b)
    assert r.native_forward_attempts==0
