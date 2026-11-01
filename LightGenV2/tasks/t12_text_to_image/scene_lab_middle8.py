"""SDK/Torch-free admission for the explicit expanded F6/64-block/eight-step bench profile.

check-data needs no checkpoint. Actual loading/acquisition needs a separately
accepted exact checkpoint binding; this module does not grant that acceptance.
"""
import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
from .scene_lab_protocol import sha

F6 = '6f34f2d720060c9193cda0e19ffb8fb4d157bc03e3fd5e5c1433c60ad884471e'
PAIRS = 'c2e30cbdbd871094df8827c78f4a02bc23305a0e39898ab53c6acf4178835fd0'
TEST = 'f132238c3a59efb1308649909c96ce3669e02c7420d598bb29183536720f9f47'
SAFE = ('sample_id','reference','prompt','seed','reference_id','reference_sha256',
        'category','condition_id','native_caption','reward_caption','prompt_profile')
PROFILE = json.loads(Path(__file__).with_name('scene_lab_middle8_profile.json').read_text(encoding='utf-8'))
FULL_PROFILE = json.loads(Path(__file__).with_name('scene_lab_full_unet8_profile.json').read_text(encoding='utf-8'))
BINDINGS={'scene_lab_middle8_binding_v1':PROFILE,'scene_lab_full_unet8_binding_v1':FULL_PROFILE}


def hex_sha(value, length=64):
    return isinstance(value,str) and len(value)==length and all(c in '0123456789abcdef' for c in value)


def checked_binding(binding, *, require_approval=True):
    keys={'schema','profile','generator_sha256','source_commit','compression_profile',
          'SIM_manifest_sha256','approved_for_physical_deployment'}
    if (not isinstance(binding,dict) or set(binding)!=keys or binding['schema']not in BINDINGS
        or binding['profile']!=BINDINGS[binding['schema']] or not hex_sha(binding['generator_sha256'])
        or not hex_sha(binding['source_commit'],40) or not hex_sha(binding['SIM_manifest_sha256'])
        or type(binding['approved_for_physical_deployment'])is not bool):
        raise ValueError('Exact middle8 graph, source, PT and matching SIM binding required')
    # Mirror the pure source-side2328 profile contract here, so the capture PC
    # never needs to import training/render dependencies just to read a ledger.
    config=checked_compression_profile(binding['compression_profile'])
    expected_schema='scene_expanded_middle_v1' if binding['schema']=='scene_lab_middle8_binding_v1'else 'scene_expanded_full_UNet8_v1'
    if (config['pairs_sha256']!=PAIRS or config['TRAIN_contexts']!=624
        or config['schema']!=expected_schema or config['UNet_deep_keep']!=binding['profile']['UNet_deep_keep']):
        raise ValueError('Old R2 TRAIN132/DEV48 or another dataset is not this profile')
    if require_approval and not binding['approved_for_physical_deployment']:
        raise ValueError('Preparation is not physical-deployment acceptance')
    return binding


def checked_compression_profile(config):
    keys={'schema','prompt_profile','vision_keep','language_keep','UNet_deep_keep','student_steps',
          'parent_generator_sha256','parent_frontend_sha256','electronic_teacher_sha256','pairs_sha256','TRAIN_contexts'}
    depths={'scene_expanded_middle_v1':9,'scene_expanded_full_UNet8_v1':10}
    if (not isinstance(config,dict)or set(config)!=keys or config['schema']not in depths
        or config['prompt_profile']!=PROFILE['prompt_profile']
        or (config['vision_keep'],config['language_keep'],config['UNet_deep_keep'],config['student_steps'])!=(6,4,depths[config['schema']],8)
        or config['parent_frontend_sha256']!=F6 or type(config['TRAIN_contexts'])is not int
        or not 1<=config['TRAIN_contexts']<=1520
        or any(not hex_sha(config[k])for k in ('parent_generator_sha256','electronic_teacher_sha256','pairs_sha256'))):
        raise ValueError('Exact source expanded-middle compression contract required')
    return dict(config)


def checked_state(state, report, binding):
    checked_binding(binding)
    descriptor=state.get('depth_compression',{})
    if (state.get('schema')!='qwen3_scene_paired_v1' or state.get('variant')!='qwen'
        or state.get('Qwen_input_side')!=256 or state.get('resolution')!=512
        or state.get('SDXL_conditioning_size')!=1024 or state.get('joint_image_context')is not False
        or state.get('pair_sha256')!=PAIRS or state.get('no_validation_test_PT_selection')is not True
        or (state.get('optical_frontend')or{}).get('checkpoint_sha256')!=F6):
        raise ValueError('Wrong native input/head/data/F6 identity')
    expected=dict(kind='progressive_guided_Euler_CFGpreserved_v1', steps=8,teacher_steps=30,
                  CFG=7.5,timesteps=PROFILE['timesteps'],IP_scale=.25)
    if state.get('sampling_contract')!=expected:
        raise ValueError('Keep the exact trained guided Euler8 contract')
    if (descriptor.get('schema')!='scene_compact_recovered_v1'
        or descriptor.get('profile')!=binding['compression_profile']
        or descriptor.get('sampling_steps')!=8
        or descriptor.get('original_UNet_tensor_names_restored')is not True
        or descriptor.get('active_graph_requires_explicit_depth_loader')is not True
        or report.get('depth_compression')!=descriptor or report.get('complete')is not True
        or report.get('compression_training_performed')is not True
        or report.get('best_sha256')!=binding['generator_sha256']):
        raise ValueError('Completed bound depth loader/checkpoint report required')
    return descriptor


def public_requests(rows, role):
    if role not in ('train','test') or not rows:raise ValueError('Explicit TRAIN or TEST role required')
    requests=[]
    for row in rows:
        if not set(SAFE)<=set(row) or (role=='test' and set(row)!=set(SAFE)):
            raise ValueError('TEST has public inputs only, never targets or extra conditions')
        if role=='train' and row.get('split')!='train':raise ValueError('TRAIN role changed')
        path=PurePosixPath(row['reference'].replace('\\','/'))
        if path.is_absolute()or'..'in path.parts or ':'in str(path) or not hex_sha(row['reference_sha256']):
            raise ValueError('Unsafe reference path or source identity')
        if (type(row['seed'])is not int or row['seed']<0 or not isinstance(row['prompt'],str)
            or not row['prompt'] or row['category']not in ('lamp','pillow')
            or row['prompt_profile']!=PROFILE['prompt_profile']):
            raise ValueError('Public sampling/prompt contract changed')
        requests.append({k:row[k] for k in ('sample_id','reference','reference_sha256','prompt','seed','category')})
        requests[-1]['split']=role
    if len({r['sample_id']for r in requests})!=len(requests) or len({(r['reference_sha256'],r['prompt'],r['seed'])for r in requests})!=len(requests):
        raise ValueError('Duplicate acquisition/inference identity')
    return requests


def checked_rows(train, test, report, selection):
    if (report.get('schema')!='scene_expanded_train_only_pairs_v1' or report.get('complete')is not True
        or report.get('eligible_for_training')is not True or report.get('pairs_sha256')!=PAIRS
        or report.get('request_sha256')!=TEST or report.get('validation_set')is not False
        or report.get('TEST_used_for_model_selection')is not True or report.get('TEST_target_images')!=0
        or report.get('TRAIN_pairs')!=624 or report.get('TEST_requests')!=500):
        raise ValueError('Complete current reviewed TRAIN624 / target-free TEST500 required')
    if (selection.get('gradient_roles')!=['train'] or selection.get('validation_set')is not False
        or selection.get('TEST_used_for_model_selection')is not True or selection.get('test_count')!=500
        or selection.get('test_request_sha256')!=TEST or selection.get('complete_cohort_required')is not True):
        raise ValueError('TEST selection is not permission for TEST gradients')
    a,b=public_requests(train,'train'),public_requests(test,'test')
    if len(a)!=624 or len(b)!=500 or Counter(r['category']for r in b)!={'lamp':250,'pillow':250}:
        raise ValueError('Incomplete full acquisition cohorts')
    if len({r['reference_sha256']for r in a})!=76 or len({r['reference_sha256']for r in b})!=26:
        raise ValueError('Views must not be counted as independent products')
    if {r['reference_sha256']for r in a}&{r['reference_sha256']for r in b}:
        raise ValueError('TRAIN/TEST source pixels overlap')
    if len({r['source_product_id']for r in train})!=44:
        raise ValueError('Current TRAIN44 product identity changed')
    return {'train':a,'test':b}


def check_data(folder, reference_root=None):
    folder=Path(folder)
    for filename,digest in [('pairs.json',PAIRS),('inference_inputs.json',TEST)]:
        if sha(folder/filename)!=digest:raise ValueError('Changed data manifest '+filename)
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    result=checked_rows(read('pairs.json'),read('inference_inputs.json'),read('report.json'),read('selection_protocol.json'))
    check_reference_roles(read('references.json'),read('pairs.json'),result)
    if reference_root is not None:
        root=Path(reference_root).resolve();seen={}
        for rows in result.values():
            for row in rows:
                path=(root/row['reference']).resolve()
                if not path.is_relative_to(root):raise ValueError('Reference escaped dataset')
                if path not in seen:seen[path]=sha(path)
                if seen[path]!=row['reference_sha256']:raise ValueError('Reference pixels changed')
    return result


def check_reference_roles(references, train_rows, requests):
    by_pixels={r['reference_sha256']:r for r in references}
    if len(by_pixels)!=len(references)or len(by_pixels)!=102:raise ValueError('Exactly102 distinct reference views required')
    products={role:set()for role in ('train','test')}
    for role,rows in requests.items():
        for row in rows:
            info=by_pixels.get(row['reference_sha256'],{})
            if (info.get('split')!=role or info.get('reference')!=row['reference']
                or info.get('reference_id')is None or not info.get('source_product_id')):
                raise ValueError('Reference view identity/role changed')
            products[role].add(info['source_product_id'])
    for row in train_rows:
        if row['source_product_id']!=by_pixels[row['reference_sha256']]['source_product_id']:
            raise ValueError('TRAIN product/reference identity changed')
    if len(products['train'])!=44 or len(products['test'])!=18 or products['train']&products['test']:
        raise ValueError('TRAIN44 and TEST18 products must remain disjoint')


def require_training(requests):
    if len(requests)!=624 or any(r.get('split')!='train' for r in requests):
        raise ValueError('Only all624 TRAIN inputs may supply calibration/fit gradients')


def checked_ledger(session, models, requests, binding):
    checked_binding(binding)
    if models!={'F6':F6,'generator':binding['generator_sha256'],'source_commit':binding['source_commit']}:
        raise ValueError('No old F9 alias or another PT/source in the new ledger')
    role=session.get('cohort_role')
    if role not in ('train','test') or len(requests)!=(624 if role=='train'else 500):
        raise ValueError('Exact explicit TRAIN624 or TEST500 acquisition role required')
    if (session.get('request_manifest_sha256')!=(PAIRS if role=='train'else TEST)
        or session.get('TEST_gradients')is not False):
        raise ValueError('Manifest and no-TEST-gradient session identity required')
    if role=='test' and (session.get('evaluation_only')is not True or session.get('TEST_used_for_model_selection')is not True):
        raise ValueError('TEST is explicit evaluation/selection, never a DEV relabel')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['check-data','check-binding'],required=True)
    p.add_argument('--data',type=Path);p.add_argument('--references',type=Path);p.add_argument('--binding',type=Path)
    a=p.parse_args()
    if a.mode=='check-data':
        if a.data is None:p.error('--data required')
        rows=check_data(a.data,a.references)
        print(json.dumps(dict(passed=True,TRAIN=len(rows['train']),TEST=len(rows['test']),SDK=False,Torch=False,new_PT_loaded=False,physical_deployment_approved=False)))
    else:
        if a.binding is None:p.error('--binding required')
        b=checked_binding(json.loads(a.binding.read_text()),require_approval=False)
        print(json.dumps(dict(profile_passed=True,actual_PT_loaded=False,physical_deployment_approved=b['approved_for_physical_deployment'])))


if __name__=='__main__':main()
