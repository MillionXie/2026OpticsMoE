"""TRAIN CCD -> frozen prefix cache -> final decoder adaptation -> fixed TEST.

No camera/SLM access. TEST cannot select an epoch or enter gradients.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

SHA='dae370fff51a0d175a5c73d7b7a19d455c3f27f56232af800b85ecfe026a8a25'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf8');tmp.replace(path)

def protected(model,prefix):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if not name.startswith(prefix+'.'):
            h.update(name.encode());h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=50);p.add_argument('--wait-capture',action='store_true');a=p.parse_args()
    root=a.project.resolve();out=a.output.resolve();train_run=root/'runs/train1000_e2000';test_run=root/'runs/full1000_e2000'
    # Waiting allocates no GPU. Complete report is written only after SDK exit.
    wait_started=time.time()
    while not (train_run/'report.json').exists():
        if not a.wait_capture:raise RuntimeError('TRAIN capture not complete')
        if time.time()-wait_started>3*3600:raise RuntimeError('TRAIN capture wait exceeded three hours; inspect capture task')
        print('Waiting for complete TRAIN report',flush=True);time.sleep(30)
    for folder in (train_run,test_run):
        report=json.loads((folder/'report.json').read_text());assert report['status']=='complete'
        contract=report['contract'];assert contract['checkpoint_sha256']==SHA
        assert contract['count']==1000 and contract['exposure_us']==2000 and contract['gain']=='Gain_X4'
        assert contract['wait_ms']==240
    assert not (out/'report.json').exists(),'Completed tuning must not be restarted'
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(927)
    sys.path.insert(0,str(root/'source'))
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
    from LightGenV2.tasks.t04_semantic_interaction.modeling import build_model
    from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import OpticalBoundary,STAGES
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import OpenMojiEditingDataset,collate_samples,load_prompt_cache
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    phase_hashes={stage:digest(test_run/'phase'/(stage+'.bmp')) for stage in STAGES}
    assert all(digest(train_run/'phase'/(stage+'.bmp'))==phase_hashes[stage] for stage in STAGES),'TRAIN/TEST masks differ'
    for folder in (train_run,test_run):
        for stage in STAGES:
            assert len(list((folder/'ccd'/stage).glob('*.png')))==1000
            assert len(list((folder/'ccd'/stage).glob('*.json')))==1000
    ckpt=root/'weights/best_checkpoint.pt';assert digest(ckpt)==SHA
    cfg=load_settings(root/'source/LightGenV2/tasks/t04_semantic_interaction/configs/layered_scene_electronic_exp05.yaml')
    cfg.qwen_checkpoint=root.parent/'OpenMoji_Lab_SHS_8um/frontend';cfg.asset_dir=root.parent/'OpenMoji_Lab_SHS_8um/assets';cfg.svg_asset_dir=cfg.asset_dir/'openmoji-17.0.0-svg'
    payload=torch.load(ckpt,map_location='cpu',weights_only=False);model=build_model(cfg,torch.device('cuda'));model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False)
    for optic in model._optical_paths():optic.set_phase_dropout_active(False)
    if hasattr(model,'decoder'):prefix='decoder';head=model.decoder
    else:prefix='shared_readout.decoder';head=model.shared_readout.decoder
    before=protected(model,prefix);initial=copy.deepcopy(head.state_dict())
    audit=json.loads((root/'data_train_adapt1000/split_audit.json').read_text());fit=set(audit['fit_ids']);val=set(audit['validation_ids']);assert len(fit)==800 and len(val)==200 and not fit&val
    write(out/'execution.json',dict(status='running',source_sha256=SHA,trainable_prefix=prefix,trainable_parameters=sum(t.numel() for t in head.parameters()),epochs=a.epochs,fit_count=800,validation_count=200,test_count=1000,selection='validation Changed-cell Accuracy only; TEST after selection',protected_before=before,split_audit_sha256=digest(root/'data_train_adapt1000/split_audit.json')))
    def cache(folder,data_dir,manifest,train_scope):
        cfg.data_dir=data_dir;data=OpenMojiEditingDataset(manifest,cfg,load_prompt_cache(data_dir/'token_embeddings_v1.pt'))
        records=data.records;assert len(records)==1000
        if train_scope:assert {r['sample_id'] for r in records}==fit|val
        else:assert not {r['sample_id'] for r in records}&(fit|val)
        features=[];rows=[]
        tapped=[]
        hook=head.register_forward_pre_hook(lambda module,args:tapped.append(args[0].detach().cpu().clone()))
        try:
            with torch.no_grad():
                for index in range(1000):
                    sid=records[index]['sample_id'] if train_scope else f'test_{index:05d}'
                    b=collate_samples([data[index]]);b={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()}
                    frames={stage:torch.from_numpy(np.asarray(Image.open(folder/'ccd'/stage/(sid+'.png')),np.float32).copy()[None]/255) for stage in STAGES}
                    receipts=[json.loads((folder/'ccd'/stage/(sid+'.json')).read_text()) for stage in STAGES]
                    for stage,receipt in zip(STAGES,receipts):
                        assert receipt['sample_id']==sid and receipt['stage']==stage
                        assert receipt['phase_sha256']==phase_hashes[stage]
                        assert receipt['exposure']['exposure_us']==2000 and receipt['exposure']['gain']=='Gain_X4'
                        assert receipt['wait_ms']==240 and receipt['canonical_orientation']=='flip_v'
                    assert len(frames)==6 and all(torch.isfinite(t).all() for t in frames.values())
                    with OpticalBoundary(model,frames):result=model(b['source_image'],b['prompt_hidden'])
                    assert len(tapped)==1;features.append(tapped.pop())
                    row={k:v.cpu() if torch.is_tensor(v) else v for k,v in b.items() if k not in ('source_image','prompt_hidden')};row['task_logits']=result['task_logits'].detach().cpu();rows.append(row)
                    if index%100==0:write(out/'progress.json',dict(status='caching',scope='TRAIN' if train_scope else 'TEST',completed=index+1,total=1000))
        finally:hook.remove()
        value=dict(features=torch.cat(features),rows=rows,source_sha256=SHA,decoder_prefix=prefix)
        torch.save(value,out/('train_features.pt' if train_scope else 'test_features.pt'));return value
    train=cache(train_run,root/'data_train_adapt1000',root/'data_train_adapt1000/capture_train.jsonl',True)
    fitidx=[i for i,r in enumerate(train['rows']) if r['sample_id'][0] in fit];validx=[i for i,r in enumerate(train['rows']) if r['sample_id'][0] in val]
    def batch(data,indices):
        rows=[data['rows'][i] for i in indices];b={}
        for key in rows[0]:b[key]=torch.cat([r[key] for r in rows]).cuda() if torch.is_tensor(rows[0][key]) else sum([r[key] for r in rows],[])
        return data['features'][indices].cuda(),b
    def evaluate(data,indices,export=False):
        acc=MetricAccumulator();head.eval();records=[];logits=[]
        with torch.no_grad():
            for start in range(0,len(indices),32):
                x,b=batch(data,indices[start:start+32]);cat,edit=head(x)
                row,prediction,_=acc.update(dict(category_logits=cat,edit_logits=edit,task_logits=b['task_logits']),b)
                if export:
                    records.extend(row);logits.append(dict(category_logits=cat.cpu(),edit_logits=edit.cpu(),prediction=prediction.cpu()))
        if export:
            write(out/'test_samples.json',records);torch.save(logits,out/'test_outputs.pt')
        return acc.compute()
    baseline_fit=evaluate(train,fitidx);baseline_val=evaluate(train,validx);bestscore=baseline_val['overall']['changed_cell_accuracy'];selected=0
    head.requires_grad_(True);optimizer=torch.optim.AdamW(head.parameters(),lr=5e-5,weight_decay=.01);best=copy.deepcopy(initial)
    history=[]
    for epoch in range(1,a.epochs+1):
        head.train();losses=[];order=torch.randperm(len(fitidx)).tolist()
        for start in range(0,len(order),32):
            ids=[fitidx[i] for i in order[start:start+32]];x,b=batch(train,ids);cat,edit=head(x)
            weight=1+11*b['edit_grid']+2*b['target_grid'].gt(0);ce=F.cross_entropy(cat,b['target_grid'].long(),reduction='none');category=(ce*weight).sum()/weight.sum()
            loss=category+1.25*F.binary_cross_entropy_with_logits(edit,b['edit_grid'],pos_weight=edit.new_tensor(8.))
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        fitmetrics=evaluate(train,fitidx);valmetrics=evaluate(train,validx);score=valmetrics['overall']['changed_cell_accuracy']
        record=dict(epoch=epoch,loss=float(np.mean(losses)),fit=fitmetrics,validation=valmetrics);history.append(record)
        print(json.dumps(record),flush=True);write(out/'history.json',history)
        if score>bestscore:bestscore=score;selected=epoch;best=copy.deepcopy(head.state_dict())
    last=copy.deepcopy(head.state_dict());head.load_state_dict(best);head.eval().requires_grad_(False)
    assert protected(model,prefix)==before,'Protected upstream weights changed'
    chosen=copy.deepcopy(payload);chosen['model']=model.state_dict();chosen['physical_adaptation']=dict(selected_epoch=selected,validation_score=bestscore,fit_ids=sorted(fit),validation_ids=sorted(val),prefix=prefix,test_used_for_selection=False)
    torch.save(chosen,out/'best.pt');head.load_state_dict(last);lastpayload=copy.deepcopy(payload);lastpayload['model']=model.state_dict();torch.save(lastpayload,out/'last.pt');head.load_state_dict(best)
    # Cache TEST only after epoch selection is sealed.
    test=cache(test_run,root/'data',root/'data/test.jsonl',False)
    actual=evaluate(test,list(range(1000)),True);head.load_state_dict(initial);baseline=evaluate(test,list(range(1000)));head.load_state_dict(best)
    assert abs(baseline['overall']['changed_cell_accuracy']-.671)<1e-8,'Original TEST CCD baseline must reproduce .671'
    # Verify cache prediction equals real-CCD full forward with selected head.
    cfg.data_dir=root/'data';dataset=OpenMojiEditingDataset(root/'data/test.jsonl',cfg,load_prompt_cache(root/'data/token_embeddings_v1.pt'))
    with torch.no_grad():
        b=collate_samples([dataset[0]]);b={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()};frames={stage:torch.from_numpy(np.asarray(Image.open(test_run/'ccd'/stage/'test_00000.png'),np.float32).copy()[None]/255) for stage in STAGES}
        with OpticalBoundary(model,frames):full=model(b['source_image'],b['prompt_hidden'])
        cat,edit=head(test['features'][:1].cuda());err=max(float((cat-full['category_logits']).abs().max()),float((edit-full['edit_logits']).abs().max()));assert err<1e-5,err
    assert protected(model,prefix)==before
    write(out/'report.json',dict(status='complete',selected_epoch=selected,validation_best=bestscore,baseline_fit=baseline_fit,baseline_validation=baseline_val,baseline_test=baseline,physical_test=actual,best_sha256=digest(out/'best.pt'),protected_unchanged=True,full_replay_cache_max_error=err,selection='800 original TRAIN fitting / 200 original TRAIN validation; TEST after selection only'))
    write(out/'progress.json',dict(status='complete',selected_epoch=selected))

if __name__=='__main__':main()
