"""Finite source-restored16 versus128-image gradient-averaging diagnostic.

The larger arm has eight times the gradient-image budget per optimizer step.
This tests gradient averaging, not an equal-compute performance comparison.
No full TEST score, validation forward, checkpoint export or automatic training.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .expert_tail_schedule import save, sha
from .checkpoint_test_selection import model_hash
from .view_conflict_probe import log_margins, summarize


def select_banks(labels, seed=17):
    import numpy as np
    y = np.asarray(labels)
    assert y.shape == (4248,)
    rng = np.random.default_rng(seed + 1200810019)
    positions = np.arange(len(y)); held = []; banks = [[], []]; controls = [[], []]
    for category in range(8):
        for origin in [positions < 3496, positions >= 3496]:
            choices = rng.permutation(np.flatnonzero((y == category) & origin))
            assert len(choices) in [437, 94]
            held.extend(choices[:16].tolist())
            for number in range(2):
                cell = choices[16 + number * 8:24 + number * 8]
                banks[number].extend(cell.tolist()); controls[number].append(int(cell[0]))
    def shuffled(values):
        values = np.asarray(values, dtype=np.int64); rng.shuffle(values); return values
    held = shuffled(held); banks = [shuffled(v) for v in banks]; controls = [shuffled(v) for v in controls]
    assert len(set(held.tolist() + banks[0].tolist() + banks[1].tolist())) == 512
    assert all(len(v) == 128 for v in banks) and all(len(v) == 16 for v in controls)
    assert all(set(c).issubset(b) for c, b in zip(controls, banks))
    return controls, banks, held


def admission(cases, margin_error, nll_error):
    eps = max(1e-5, 10 * margin_error); neps = max(1e-5, 10 * nll_error)
    insufficient = []; rejected = []
    if len(cases) != 2 or not all(math.isfinite(v) and v >= 0 for v in [margin_error, nll_error]):
        return dict(admitted=False, state='inconclusive', rejected=[], insufficient=['Two finite completed trials required'])
    for number, case in enumerate(cases, 1):
        control, candidate = case['small'], case['large']
        if case['actual_update_relative_change'] <= 10 * case['repeat_update_relative_error'] + 1e-7:
            rejected.append(f'batch{number}/no_actual_update_direction_change')
        # Both origins are explicitly observed training/development data.
        for origin in ['train', 'test']:
            row, ref = candidate['clean'][origin], control['clean'][origin]
            label = f'batch{number}/clean/{origin}'
            if row['repair_samples'] < 8:
                insufficient.append(label + '/repair_count'); continue
            if origin == 'test' and (row['repair_mean_delta'] <= eps or row['repair_mean_delta'] <= ref['repair_mean_delta'] + eps):
                rejected.append(label + '/repair_not_positive_and_superior')
            if origin == 'train' and row['repair_mean_delta'] < ref['repair_mean_delta'] - eps:
                rejected.append(label + '/repair_worse_than_control')
            if row['all_mean_margin_delta'] < ref['all_mean_margin_delta'] - eps:
                rejected.append(label + '/mean_margin_worse_than_control')
        for view in ['clean', 'd4']:
            if sum(candidate[view][o]['boundary_guard_samples'] for o in ['train', 'test']) < 8:
                insufficient.append(f'batch{number}/{view}/boundary_count')
            for origin in ['train', 'test']:
                row, ref = candidate[view][origin], control[view][origin]; label = f'batch{number}/{view}/{origin}'
                if row['source_correct_guard'] == 0 or row['boundary_guard_samples'] == 0:
                    insufficient.append(label + '/empty_guard'); continue
                if row['guard_new_wrong'] or row['boundary_guard_new_wrong']:
                    rejected.append(label + '/new_guard_error')
                if row['guard_delta_q10'] < ref['guard_delta_q10'] - eps or row['boundary_guard_delta_q10'] < ref['boundary_guard_delta_q10'] - eps:
                    rejected.append(label + '/guard_tail_worse_than_control')
                if row['nll_delta'] > neps:
                    rejected.append(label + '/NLL_increase')
    return dict(admitted=not rejected and not insufficient,
        state='inconclusive' if insufficient else 'rejected' if rejected else 'admitted_for_bounded_comparison',
        rejected=rejected, insufficient=insufficient, margin_threshold=eps, NLL_threshold=neps,
        requirement='Both independent trials: TEST clean repair positive and superior to16; TRAIN repair and overall margins no worse than16; no new source-guard errors, guard tails not worse, NLL not increased.')


def validate(c):
    assert c['seed'] == 17 and c['model_experts'] in [25, 49]
    assert c['expected_parameter_count'] == {25: 5634176, 49: 11078432}[c['model_experts']]
    assert c['arms'] == ['small', 'small_repeat', 'large']
    assert c['small_batch'] == 16 and c['large_batch'] == 128 and c['microbatch'] == 2
    assert c['fit_batches'] == 2 and c['heldout_per_class_per_origin'] == 16
    assert c['learning_rate'] == 3e-5 and c['fixed_lambda_kd'] == .3203496288685833
    assert c['temperature'] == 2 and c['parent_weight'] == .5
    assert c['explicit_test_training_authorization'] and c['test_images_enter_student_gradient']
    assert c['gpu_index'] in [1, 3] and c['minimum_gpu_free_mib'] >= 6000
    assert c['gpu_uuid'] == {1: 'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d', 3: 'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd'}[c['gpu_index']]
    assert c['training_auto_started'] is False


def run(c, out, profile):
    import numpy as np
    import torch
    from . import reheat_refine as base
    from .test_exposed_distill import Auxiliary, load_pool
    from .diagnose_topk import classification_margin_loss
    from .sharpness_refine import optimization_step
    from .train import objective

    validate(c)
    commit = os.environ['T10_BATCH_PROBE_SOURCE_COMMIT']
    os.environ['T10_REHEAT_SOURCE_COMMIT'] = os.environ['T10_DISTILL_SOURCE_COMMIT'] = commit
    assert os.environ['CUDA_VISIBLE_DEVICES'] == c['gpu_uuid']
    gpu = subprocess.check_output(['nvidia-smi', '-i', str(c['gpu_index']), '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert gpu[0].strip() == c['gpu_uuid'] and int(gpu[1]) >= c['minimum_gpu_free_mib']
    commit, hashes = base.sources(profile, c)
    assert sha(Path(c['target_cache'])/'teacher_manifest.json') == c['locked_teacher_manifest_sha256']
    assert sha(Path(c['parent_cache'])/'receipt.json') == c['locked_parent_receipt_sha256']
    assert sha(Path(c['parent_cache'])/'parent_scores.npz') == c['locked_parent_scores_sha256']
    auxiliary = Auxiliary(c, calibrated=False); auxiliary.weight = c['fixed_lambda_kd']
    assert auxiliary.identity['parent_checkpoint_sha256'] == c['parent_checkpoint_sha256']
    model, metadata, protocol, splits, _, _ = base.prepare(c, 'student')
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    identity = model_hash(original); assert identity == c['source_tensor_sha256']
    tc = protocol['training_proposal']
    assert {k: tc[k] for k in ['label_smoothing','capture_loss_weight','phase_tv_weight','router_balance_weight']} == dict(label_smoothing=.02,capture_loss_weight=.005,phase_tv_weight=.02,router_balance_weight=.01)
    xx, y, ids, val = load_pool(c); x = torch.from_numpy(xx.transpose(0,3,1,2).copy()); del xx
    assert np.array_equal(y, auxiliary.y) and np.array_equal(ids, auxiliary.ids)
    assert set(ids).isdisjoint(splits['val'][2]) and set(ids).isdisjoint(val[2]); del splits, val
    origins = np.asarray(['train']*3496 + ['test']*752)
    controls, banks, held = select_banks(y, c['seed'])
    fit_codes = [np.random.default_rng(c['seed']+i*1000033).integers(0,8,len(y)) for i in [1,2]]
    held_codes = dict(clean=np.zeros(len(held),np.int64), d4=np.random.default_rng(c['seed']+1200810023).integers(1,8,len(held)))
    out.mkdir(parents=True, exist_ok=False); started = time.time()
    counters = dict(source_heldout_forward_inputs=0, post_update_heldout_forward_inputs=0, gradient_image_inputs=0, actual_Adam_steps=0)
    save(out/'metadata.json', dict(pid=os.getpid(),command=sys.argv,source_commit=commit,source_files=hashes,
        host_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),configuration=c,
        source_metadata=metadata,source_model_tensor_sha256=identity,auxiliary_identity=auxiliary.identity,
        graph_unchanged=True,no_checkpoint_written=True,no_validation_forward=True,training_auto_started=False,
        role='TEST-exposed direction diagnostic; larger batch uses8x gradient inputs per step, not an equal-compute or full752 accuracy result.'))
    save(out/'sampling.json',dict(control_indices=[v.tolist() for v in controls],bank_indices=[v.tolist() for v in banks],
        heldout_indices=held.tolist(),heldout_ids=ids[held].tolist(),heldout_origins=origins[held].tolist(),
        heldout_codes={k:v.tolist() for k,v in held_codes.items()},bank_ids=[ids[v].tolist() for v in banks],
        bank_origins=[origins[v].tolist() for v in banks],fit_d4_codes=[v[ix].tolist() for v,ix in zip(fit_codes,banks)],
        selection='Uniform class/origin RNG before any student or teacher predictions;512 underlyingIDs, bank128 contains paired16 control.'))

    def predict(codes, phase, counter):
        model.eval(); values=[]
        with torch.no_grad():
            for begin in range(0,len(held),4):
                rows=held[begin:begin+4]
                values.append(model(auxiliary.images_with_codes(x,torch.from_numpy(rows),codes[begin:begin+4]))['probabilities'].cpu().numpy())
                counters[counter]+=len(rows)
                if begin%64==0:save(out/'status.json',dict(state=phase,pid=os.getpid(),completed=begin+len(rows),total=len(held),timestamp=time.time(),cost=counters))
        result=np.concatenate(values);assert np.isfinite(result).all() and np.max(abs(result.sum(1)-1))<3e-6
        return result

    baseline={v:predict(codes,'source_'+v,'source_heldout_forward_inputs') for v,codes in held_codes.items()}
    repeats={v:predict(codes,'repeat_'+v,'source_heldout_forward_inputs') for v,codes in held_codes.items()}
    margin_error=max(float(abs(log_margins(baseline[v],y[held])-log_margins(repeats[v],y[held])).max()) for v in held_codes)
    nll_error=max(float(abs(np.log(baseline[v][np.arange(len(held)),y[held]].clip(1e-12))-np.log(repeats[v][np.arange(len(held)),y[held]].clip(1e-12))).max()) for v in held_codes)
    probability_error=max(float(abs(baseline[v]-repeats[v]).max()) for v in held_codes)
    for view,codes in held_codes.items():
        np.savez_compressed(out/f'source_{view}.npz',ids=ids[held],labels=y[held],split_origin=origins[held],view_codes=codes,
            scores=baseline[view],repeat_scores=repeats[view],teacher_scores=auxiliary.teacher[codes,held])
    cases=[];weights=torch.ones(8,device='cuda')
    for no,(control,bank,codes) in enumerate(zip(controls,banks,fit_codes),1):
        np.savez_compressed(out/f'fit_bank{no}.npz',ids=ids[bank],labels=y[bank],split_origin=origins[bank],indices=bank,
            control_indices=control,view_codes=codes[bank],teacher_logits=auxiliary.logits[codes[bank],bank],
            teacher_scores=auxiliary.teacher[codes[bank],bank],parent_scores=auxiliary.parent[codes[bank],bank])
        case=dict(batch=no,small={},small_repeat={},large={},steps={});states={};arm_scores={}
        for arm in c['arms']:
            model.load_state_dict(original);assert model_hash(model.state_dict())==identity
            torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
            optimizer=torch.optim.AdamW([dict(params=list(model.expert_phases),lr=3e-5),
                dict(params=list(model.global_phases),lr=1.5e-5),dict(params=[model.router_phase],lr=3e-6)],weight_decay=0.)
            indices=bank if arm=='large' else control
            auxiliary.codes=codes;auxiliary.start_batch(torch.from_numpy(indices),0);model.train()
            def closure():
                total=0.
                for begin in range(0,len(indices),2):
                    rows=torch.from_numpy(indices[begin:begin+2]);labels=torch.from_numpy(y[rows.numpy()]).cuda()
                    output=model(auxiliary.images(x,rows,0));p=output['probabilities']
                    loss=(objective(model,output,labels,weights,tc)+.2*classification_margin_loss(p,labels))*len(rows)/len(indices)
                    loss=loss+auxiliary.loss(p,rows,0)
                    assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
                return total
            info=optimization_step(model,optimizer,closure)
            assert len(info['gradients'])==5 and all(v['finite'] and v['norm']>0 for v in info['gradients'].values())
            assert all(float(s['step'])==1 for s in optimizer.state.values())
            counters['gradient_image_inputs']+=len(indices);counters['actual_Adam_steps']+=1
            states[arm]={name:p.detach().cpu()-original[name] for name,p in model.named_parameters()}
            case['steps'][arm]=dict(optimization=info,gradient_inputs=len(indices),teacher_eligible=auxiliary.denominator,
                parent_protected=auxiliary.parent_denominator,model_tensor_sha256=model_hash(model.state_dict()),
                optimizer_initial_step=0,optimizer_final_step=1,source_restored_before_step=True)
            arm_scores[arm]={}
            for view,vc in held_codes.items():
                scores=predict(vc,f'batch{no}_{arm}_{view}','post_update_heldout_forward_inputs');arm_scores[arm][view]=scores
                case[arm][view]=summarize(baseline[view],scores,auxiliary.teacher[vc,held],y[held],origins[held],.2)
                np.savez_compressed(out/f'batch{no}_{arm}_{view}.npz',ids=ids[held],labels=y[held],split_origin=origins[held],view_codes=vc,
                    scores=scores,source_scores=baseline[view],teacher_scores=auxiliary.teacher[vc,held])
            del optimizer
        denom=sum(float(v.square().sum()) for v in states['small'].values())
        case['actual_update_relative_change']=math.sqrt(sum(float((states['large'][n]-v).square().sum()) for n,v in states['small'].items())/max(denom,1e-30))
        case['repeat_update_relative_error']=math.sqrt(sum(float((states['small_repeat'][n]-v).square().sum()) for n,v in states['small'].items())/max(denom,1e-30))
        for view in held_codes:
            a,b=arm_scores['small'][view],arm_scores['small_repeat'][view]
            probability_error=max(probability_error,float(abs(a-b).max()))
            margin_error=max(margin_error,float(abs(log_margins(a,y[held])-log_margins(b,y[held])).max()))
            nll_error=max(nll_error,float(abs(np.log(a[np.arange(len(held)),y[held]].clip(1e-12))-np.log(b[np.arange(len(held)),y[held]].clip(1e-12))).max()))
        model.load_state_dict(original);assert model_hash(model.state_dict())==identity
        cases.append(case);save(out/'cases.json',cases)
    assert counters==dict(source_heldout_forward_inputs=1024,post_update_heldout_forward_inputs=3072,gradient_image_inputs=320,actual_Adam_steps=6)
    counters['total_model_image_inputs']=4416
    assert sha(Path(c['incumbent_run'])/'best_checkpoint.pt')==c['incumbent_sha256']
    decision=admission(cases,margin_error,nll_error)
    save(out/'receipt.json',dict(pid=os.getpid(),source_commit=commit,source_files=hashes,profile_sha256=sha(profile),
        source_checkpoint_sha256=c['incumbent_sha256'],source_model_tensor_sha256=identity,parent_checkpoint_sha256=c['parent_checkpoint_sha256'],
        auxiliary_identity=auxiliary.identity,cases=cases,admission=decision,cost=counters,source_restored_exactly=True,
        source_best_disk_SHA_unchanged=True,no_checkpoint_written=True,training_auto_started=False,no_validation_forward=True,
        repeated_probability_max_error=probability_error,repeated_logmargin_max_error=margin_error,repeated_NLL_max_error=nll_error,
        elapsed_seconds=time.time()-started,metric_role='Teacher/student TEST-exposed diagnostic only; no new full752 accuracy.'))
    save(out/'status.json',dict(state='complete',pid=os.getpid(),timestamp=time.time(),admission_state=decision['state'],cost=counters))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();c=json.loads(args.profile.read_text());c['_profile']=str(args.profile)
    try:run(c,args.out,args.profile)
    except BaseException as error:
        if args.out.exists():save(args.out/'status.json',dict(state='failed',pid=os.getpid(),error=repr(error),timestamp=time.time()))
        raise


if __name__=='__main__':main()
