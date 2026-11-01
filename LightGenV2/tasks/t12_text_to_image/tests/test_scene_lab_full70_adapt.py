import copy,os,subprocess,sys
import pytest
from LightGenV2.tasks.t12_text_to_image import scene_lab_full70_adapt as a
from LightGenV2.tasks.t12_text_to_image.tests.test_scene_lab_middle8 import full_binding


def admitted_binding():
    b=full_binding();b['generator_sha256']=a.PARENT;return b


def test_import_is_sdk_and_torch_free():
    subprocess.run([sys.executable,'-c',"import sys;import LightGenV2.tasks.t12_text_to_image.scene_lab_full70_adapt;assert 'torch'not in sys.modules"],check=True)


def test_only_the_approved_full70_parent_and_finite_recipes():
    b=admitted_binding();assert a.full_binding(b)==b
    with pytest.raises(ValueError):a.full_binding(full_binding())
    a.fit_protocol(1200,900,3e-5,4,0.);a.fit_protocol(1200,900,3e-5,4,.1)
    a.fit_protocol(1200,900,3e-5,4,1.) # One additional manually approved stronger nominal constraint.
    for args in [(1201,900,3e-5,4,0.),(1200,901,3e-5,4,0.),(1200,900,float('nan'),4,0.),(1200,900,3e-5,5,0.),(1200,900,3e-5,4,.5),(1200,900,3e-5,4,float('nan'))]:
        with pytest.raises(ValueError):a.fit_protocol(*args)


def child_fixture():
    if os.name=='nt':pytest.skip('Torch CPU tests required on server, local Torch DLL unavailable')
    import torch
    parent={'head':{'w':torch.ones(2,3)},'source_commit':'a'*40,'depth_compression':{'profile':'fixture'},
            'unet_lora':{'l':torch.ones(3)},'sampling_contract':{'steps':8},'optical_frontend':{'checkpoint_sha256':a.F6}}
    child=copy.deepcopy(parent);child['head']['w']*=2;child['source_commit']='b'*40
    child['lab_adaptation']=dict(schema=a.SCHEMA,parent_generator_SHA256=a.PARENT,frontend_F6_SHA256=a.F6,
        TRAIN=624,TEST_gradients=False,added_deployment_parameters=0,frozen_body_exact=True,trainable_parameters=7791616,
        measured_ledger_SHA256='c'*64,measured_features_SHA256='d'*64,nominal_conditions_SHA256='e'*64)
    report=dict(complete=True,lab_adaptation_complete=True,parent_generator_SHA256=a.PARENT,
                compression_training_performed=True,depth_compression=parent['depth_compression'],source_commit='b'*40)
    return parent,child,report,admitted_binding()


def test_child_may_only_change_original_head_not_optical_or_sampling_body():
    parent,child,report,binding=child_fixture();a.checked_child(child,parent,report,binding)
    for key,value in [('sampling_contract',{'steps':12}),('optical_frontend',{'checkpoint_sha256':'f'*64}),('unet_lora',{'l':child['unet_lora']['l']+1})]:
        bad=copy.deepcopy(child);bad[key]=value
        with pytest.raises(ValueError):a.checked_child(bad,parent,report,binding)
    bad=copy.deepcopy(child);bad['new_readout']={}
    with pytest.raises(ValueError):a.checked_child(bad,parent,report,binding)
    bad=copy.deepcopy(child);bad['lab_adaptation']['TEST_gradients']=True
    with pytest.raises(ValueError):a.checked_child(bad,parent,report,binding)


def test_child_shape_or_unproven_report_rejected():
    import numpy as np
    parent,child,report,binding=child_fixture()
    bad=copy.deepcopy(child);bad['head']['w']=bad['head']['w'][:1]
    with pytest.raises(ValueError):a.checked_child(bad,parent,report,binding)
    with pytest.raises(ValueError):a.checked_child(child,parent,dict(report,lab_adaptation_complete=False),binding)
