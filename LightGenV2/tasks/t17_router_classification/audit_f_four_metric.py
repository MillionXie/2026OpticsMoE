"""Independent identity, same-PT metrics, canonical forward and frozen-router audit."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from .train_figure_revision import build_model,evaluate
from .figure_revision_data import load_panel_f
from .model_four import encode_rgb
from .report_f_load_balance import usage_statistics
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def audit(root):
    task=Path(__file__).resolve().parent;torch.set_num_threads(4);audits={};teacher_logits=[]
    for stage in ('teacher','student'):
        folder=root/stage;result=json.loads((folder/'result.json').read_text());metadata=json.loads((folder/'metadata.json').read_text());history=json.loads((folder/'history.json').read_text());cfg=metadata['profile'];best=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(folder/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        assert sha256(folder/'best_checkpoint.pt')==result['checkpoint_sha256']
        assert metadata['TEST_hard_GT_in_loss']==(stage=='teacher')and metadata['VAL_gate']is False
        for name,digest in metadata['source_sha256'].items():assert sha256(task/name)==digest,name
        for name,digest in metadata['physical_source_sha256'].items():assert sha256(task.parents[1]/name)==digest,name
        parent_path=task/cfg['parent_checkpoint']if stage=='teacher'else root/'teacher/best_checkpoint.pt';parent=torch.load(parent_path,map_location='cpu',weights_only=False)
        if stage=='teacher':assert sha256(parent_path)==cfg['parent_checkpoint_sha256']
        else:assert sha256(parent_path)==metadata['initial_teacher_SHA']
        assert torch.equal(best['model']['router_phase'],parent['model']['router_phase'])and torch.equal(last['model']['router_phase'],parent['model']['router_phase'])
        m=build_model(cfg).cuda().eval();m.load_state_dict(best['model'],strict=True);assert len(m.state_dict())==4 and sum(p.numel()for p in m.parameters())==487204 and sum(isinstance(v,torch.nn.Linear)for v in m.modules())==1 and not any(isinstance(v,(torch.nn.Conv2d,torch.nn.LayerNorm))for v in m.modules())
        raw,identity=load_panel_f(cfg['data']);parts={}
        for split,key in [('train','train'),('val','val'),('test','original_test_dev')]:
            part=raw[split];saved=np.load(folder/f'{split}_predictions.npz',allow_pickle=False);assert all(np.array_equal(saved[n],part[n])for n in ('ids','labels','domains'));assert np.all((saved['route_power']>0).sum(1)==2)
            # Re-encode the complete unchanged original split and replay the canonical model.
            x=torch.cat([encode_rgb(torch.from_numpy(part['images'][i:i+128]).cuda())for i in range(0,len(part['labels']),128)]);metrics,arr=evaluate(m,x,torch.tensor(part['labels'],device='cuda',dtype=torch.long),cfg['eval_batch'],part['domains']);assert np.array_equal(arr['predictions'],saved['predictions'])and np.allclose(arr['scores'],saved['scores'],atol=3e-6,rtol=1e-4)
            assert np.array_equal(arr['route_power']>0,saved['route_power']>0)
            parts[split]=dict(full_original_ID_label_domain_match=True,canonical_predictions_exact=True,max_probability_difference=float(np.max(np.abs(arr['scores']-saved['scores']))),domains={})
            for dom,name in [(0,'RGB'),(1,'SAR')]:
                mask=part['domains']==dom;correct=int((saved['predictions'][mask]==part['labels'][mask]).sum());assert correct==result[key]['domains'][name]['correct'];parts[split]['domains'][name]=dict(n=int(mask.sum()),correct=correct,accuracy=correct/int(mask.sum()),usage=usage_statistics(saved['route_power'][mask],saved['dense_route_power'][mask]))
            if stage=='teacher'and split in ('train','test'):
                with torch.no_grad():teacher_logits.append(torch.cat([m(x[i:i+cfg['eval_batch']])['raw_logits'].cpu()for i in range(0,len(x),cfg['eval_batch'])]).numpy())
            del x;torch.cuda.empty_cache()
        assert len(history)==result['completed_epochs']and best['epoch']==result['selected_epoch']
        assert all(h['phase_head_updates']==125 and h['mixed_original_presentations']==8000 and h['unique_TRAIN_images']==6000 and h['unique_original_TEST_images']==2000 and h['router_frozen_bitwise']and min(h['classification_raw_expert_gradient_L2_sum'])>0 for h in history)
        fits=[h['four_group_head_fit']for h in history if 'four_group_head_fit'in h]
        assert all(v['all_phases_bitwise_frozen']and v['unique_TRAIN_images']==6000 and v['unique_original_TEST_images']==2000 and v['TEST_hard_GT_in_loss']==(stage=='teacher')and v['internal_steps']<=40 for v in fits)
        assert len(last['optimizer']['state'])==2 and len(last['head_optimizer']['state'])==(0 if len(history)%5==0 else 1)and 'rng_state'in last
        changes={n:float((last['model'][n]-parent['model'][n]).norm())for n in best['model']};assert changes['router_phase']==0 and all(changes[n]>0 for n in ('first_phase','global_phase','shared_head.weight'))
        values=[parts[s]['domains'][d]['accuracy']for s in ('train','test')for d in ('RGB','SAR')];assert min(values)==result['minimum_four_accuracy']
        audits[stage]=dict(checkpoint_sha256=result['checkpoint_sha256'],source_commit=metadata['source_commit'],strict_four_state=True,parameters=487204,Linear=1,CNN=0,router_bitwise_parent=True,parts=parts,minimum_four_accuracy=min(values),all_four_at_least_65=min(values)>=.65,last_actual_parameter_change_L2=changes,budget=dict(optical_epochs=len(history),phase_head_updates=125*len(history),original_TRAIN_presentations=6000*len(history),original_TEST_presentations=2000*len(history),periodic_head_original_presentations=8000*len(fits),periodic_head_inner_steps=sum(v['internal_steps']for v in fits),initial_head_inner_steps=metadata.get('initial_group_head_fit',{}).get('internal_steps',0)),TEST_hard_GT_in_current_student_loss=metadata['TEST_hard_GT_in_loss'],student_inherited_TEST_hard_GT_teacher=stage=='student')
        del m,raw,best,last;torch.cuda.empty_cache()
    target=root/'teacher/canonical_teacher_targets.npz';manifest=json.loads((root/'teacher/teacher_target_manifest.json').read_text());assert sha256(target)==manifest['logits_sha256']and sha256(root/'teacher/best_checkpoint.pt')==manifest['teacher_checkpoint_sha256'];z=np.load(target,allow_pickle=False);replayed=np.concatenate(teacher_logits);assert replayed.shape==(8000,10)and np.array_equal(replayed,z['logits'])
    cfg=json.loads((root/'teacher/metadata.json').read_text())['profile'];raw,_=load_panel_f(cfg['data']);assert np.array_equal(z['ids'],np.r_[raw['train']['ids'],raw['test']['ids']])and np.array_equal(z['labels'],np.r_[raw['train']['labels'],raw['test']['labels']])and np.array_equal(z['groups'],np.r_[raw['train']['domains'],raw['test']['domains']+2])
    audits['teacher_target_replay']=dict(complete8000_canonical_logits_bitwise=True,shape=[8000,10],IDs_labels_domains_original=True,TRAIN_and_TEST_view_code=0,teacher_targets_detached_by_tested_loss=True)
    return audits


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();result=audit(args.root);args.output.write_text(json.dumps(result,indent=2));print(json.dumps({s:{k:v for k,v in r.items()if k in ('checkpoint_sha256','minimum_four_accuracy','all_four_at_least_65','budget')}for s,r in result.items()}))

if __name__=='__main__':main()
