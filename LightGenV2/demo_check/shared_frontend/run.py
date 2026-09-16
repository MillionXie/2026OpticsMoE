"""Matched optical training after a single previously pretrained frozen CNN."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import random
import subprocess
import sys
import time
import types
import numpy as np
import torch
from model import SharedFrontend, FrontendOptics, PhaseOnly, encode_features, TASK, cnn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('shared_frontend_pure_runner', TASK/'pure_optical/run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
save, sha, tensors_sha, batches, objective = base.save, base.sha, base.tensors_sha, base.batches, base.objective


def write_predictions(path, arrays, probabilities):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'domain', 'label', 'prediction']+[f'p{i}' for i in range(10)])
        writer.writerows([str(i), int(d), int(y), int(p.argmax()), *p.tolist()]
                        for i,d,y,p in zip(arrays['validation_ids'],arrays['validation_domains'],arrays['validation_labels'],probabilities))


def load_frontend(run, cfg, data_sha):
    metadata = json.loads((run/'metadata.json').read_text())
    assert metadata['data_sha256'] == data_sha
    path = run/'electronic/best_checkpoint.pt'
    assert sha(path) == cfg['electronic_checkpoint_sha256']
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint['stage'] == 'electronic' and checkpoint['epoch'] == 19
    frontend = SharedFrontend(checkpoint['model']).cuda()
    expected = {k:v for k,v in checkpoint['model'].items() if not k.startswith('head.')}
    assert tensors_sha(frontend.state_dict()) == tensors_sha(expected)
    return frontend, dict(source_checkpoint_sha256=sha(path), source_run=str(run),
                          source_training_commit=metadata['git_commit'], selected_epoch=checkpoint['epoch'],
                          frozen_parameters=sum(p.numel() for p in frontend.parameters()),
                          frozen_tensors_sha256=tensors_sha(frontend.state_dict()))


def smoke(frontend, cfg, optical_cfg, out):
    images = torch.randint(1,255,(8,56,56,3),dtype=torch.uint8,device='cuda')
    labels = torch.arange(8,device='cuda'); frozen = tensors_sha(frontend.state_dict())
    results=[]; reference_features=None
    # Compare the refactored backend to the exact historical image-input implementation.
    old = types.ModuleType('historical_phase_only')
    old.__file__ = str(TASK/'pure_optical/models.py')
    historical_source = subprocess.check_output(['git','show','81fb4e79:LightGenV2/demo_check/pure_optical/models.py'],cwd=HERE,text=True)
    exec(compile(historical_source, old.__file__, 'exec'), old.__dict__)
    for architecture in cfg['architectures']:
        before = old.PhaseOnly(architecture,optical_cfg).cuda()
        after = PhaseOnly(architecture,optical_cfg).cuda()
        a,b = before(images),after(images)
        assert all((a[k] is None and b[k] is None) or torch.equal(a[k],b[k]) for k in a)
        objective(a,labels).backward();objective(b,labels).backward()
        assert all(torch.equal(p.grad,dict(after.named_parameters())[n].grad) for n,p in before.named_parameters())
        del before,after,a,b
        model=FrontendOptics(frontend,architecture,optical_cfg).cuda().train()
        assert all(not m.training for m in frontend.modules())
        assert all(not p.requires_grad for p in frontend.parameters())
        initial=tensors_sha(dict(model.optical.named_parameters()))
        output=model(images);features=output['features'];amplitude=encode_features(features)
        assert tuple(features.shape)==(8,128) and tuple(amplitude.shape)==(8,224,224)
        if reference_features is None:reference_features=features.clone()
        else:assert torch.equal(reference_features,features)
        assert torch.allclose(amplitude.square().sum((-2,-1)),torch.ones(8,device='cuda'),atol=1e-5)
        assert torch.allclose(output['output_power'],output['input_power'],atol=1e-5)
        loss=objective(output,labels);loss.backward()
        gradients={n:float(p.grad.norm()) for n,p in model.optical.named_parameters()}
        assert all(np.isfinite(g) and g>0 for g in gradients.values())
        assert all(p.grad is None for p in frontend.parameters())
        opt=torch.optim.Adam(model.optical.parameters(),lr=cfg['learning_rate']);opt.step()
        assert frozen==tensors_sha(frontend.state_dict())
        assert all(tensors_sha(dict(model.optical.named_parameters()))[n]!=v for n,v in initial.items())
        results.append(dict(architecture=architecture,gradients=gradients,features_identical=True,
                            frozen_state_verified=True,historical_backend_forward_gradient_bitwise_equal=True,
                            feature_shape=list(features.shape),amplitude_shape=list(amplitude.shape),loss=float(loss)))
        del model,output,loss,opt;torch.cuda.empty_cache()
    save(out/'smoke.json',results);print(json.dumps(results),flush=True)


def train(frontend,cfg,optical_cfg,arrays,train_data,val,out):
    frozen=tensors_sha(frontend.state_dict());summaries=[]
    for architecture in cfg['architectures']:
        dest=out/architecture;dest.mkdir()
        model=FrontendOptics(frontend,architecture,optical_cfg).cuda()
        initial=tensors_sha(dict(model.optical.named_parameters()))
        initial_validation,_=base.evaluate(model,val,cfg);save(dest/'initial_validation.json',initial_validation)
        optimizer=torch.optim.Adam(model.optical.parameters(),lr=cfg['learning_rate'],weight_decay=0.)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['epochs'],eta_min=cfg['learning_rate']*cfg['minimum_lr_ratio'])
        best=(-1.,float('-inf'));history=[];started=time.perf_counter()
        for epoch in range(1,cfg['epochs']+1):
            model.train();total=0.;correct=0;n=0;feature_digest=hashlib.sha256();gradients=None
            order=torch.randperm(len(train_data[1]),generator=torch.Generator().manual_seed(cfg['seed']*1000003+epoch))
            aug_seed=cfg['seed']*99991+epoch
            for images,labels,_,_ in batches(*train_data,order,cfg['batch_size'],augment_seed=aug_seed):
                optimizer.zero_grad(set_to_none=True)
                output=model(images);loss=objective(output,labels)
                assert bool(torch.isfinite(loss));loss.backward()
                assert all(p.grad is None for p in frontend.parameters())
                if gradients is None:
                    gradients={name:float(p.grad.norm()) for name,p in model.optical.named_parameters()}
                    assert all(np.isfinite(g) and g>0 for g in gradients.values())
                torch.nn.utils.clip_grad_norm_(model.optical.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                feature_digest.update(output['features'].cpu().numpy().tobytes())
                total+=float(loss)*len(labels);correct+=int((output['probabilities'].argmax(1)==labels).sum());n+=len(labels)
            validation,_=base.evaluate(model,val,cfg)
            assert frozen==tensors_sha(frontend.state_dict())
            row=dict(epoch=epoch,train_loss=total/n,train_accuracy=correct/n,validation=validation,
                     learning_rate=optimizer.param_groups[0]['lr'],order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
                     augmentation_seed=aug_seed,input_feature_sha256=feature_digest.hexdigest(),
                     first_batch_gradients=gradients,frontend_frozen_verified=True,seconds=time.perf_counter()-started)
            history.append(row);key=(validation['accuracy'],-validation['loss'])
            payload=dict(model=model.optical.state_dict(),epoch=epoch,config=cfg,optical_config=optical_cfg,
                         architecture=architecture,validation=validation,frontend_tensors_sha256=frozen)
            if key>best:best=key;torch.save(payload,dest/'best_checkpoint.pt')
            scheduler.step()
            torch.save(dict(**payload,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),dest/'last_checkpoint.pt')
            save(dest/'history.json',history);save(out/'status.json',dict(state='training',architecture=architecture,epoch=epoch))
            print(json.dumps(dict(architecture=architecture,epoch=epoch,train_accuracy=correct/n,
                                 accuracy=validation['accuracy'],domain_accuracy=validation['domain_accuracy'],seconds=row['seconds'])),flush=True)
        checkpoint=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        model.optical.load_state_dict(checkpoint['model'])
        validation,prob=base.evaluate(model,val,cfg);train_metrics,_=base.evaluate(model,train_data,cfg)
        write_predictions(dest/'validation_predictions.csv',arrays,prob)
        end=tensors_sha(dict(model.optical.named_parameters()))
        assert all(end[k]!=v for k,v in initial.items()) and frozen==tensors_sha(frontend.state_dict())
        summary=dict(architecture=architecture,selected_epoch=checkpoint['epoch'],validation=validation,
                     train_unaugmented=train_metrics,initial_parameters_sha256=initial,final_parameters_sha256=end,
                     checkpoint_sha256=sha(dest/'best_checkpoint.pt'),frontend_tensors_sha256=frozen,
                     trainable_parameters=sum(p.numel() for p in model.optical.parameters()),seconds=time.perf_counter()-started)
        save(dest/'summary.json',summary);summaries.append(summary);save(out/'results.json',summaries)
        del model,optimizer,scheduler,checkpoint,output,loss;torch.cuda.empty_cache()
    histories=[json.loads((out/a/'history.json').read_text()) for a in cfg['architectures']]
    assert len({tuple((r['order_sha256'],r['augmentation_seed'],r['input_feature_sha256']) for r in h) for h in histories})==1
    save(out/'fairness.json',dict(frontend_states_unchanged=True,all_epoch_input_features_bitwise_identical=True,
                                identical_order_and_augmentation=True,no_electronic_classifier=True,test_set_used=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','train','evaluate'],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--electronic-run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());optical_cfg=json.loads((TASK/'pure_optical/config.json').read_text())
    assert all(cfg[k]==optical_cfg[k] for k in ['seed','epochs','batch_size','learning_rate','minimum_lr_ratio'])
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    manifest=json.loads(args.data.with_name('manifest.json').read_text());data_sha=sha(args.data)
    assert data_sha==manifest['data_sha256']
    for key in ['protocol','train_pairs_per_class','validation_pairs_per_class']:
        assert manifest['protocol'][key]==optical_cfg[key]
    frontend,provenance=load_frontend(args.electronic_run,cfg,data_sha)
    files=[*HERE.glob('*.py'),args.config,TASK/'pure_optical/models.py',TASK/'pure_optical/run.py',
           TASK/'pure_optical/config.json',TASK/'frozen_electronic/model.py',base.ARCHIVE/'optical_reference/optics.py']
    metadata=dict(config=cfg,optical_config=optical_cfg,command=sys.argv,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
                  source_sha256={str(p.relative_to(TASK)):sha(p) for p in files},python=sys.version,torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  environment=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
                  data_sha256=data_sha,data_manifest_sha256=sha(args.data.with_name('manifest.json')),
                  split_sha256=manifest['original_split_sha256'],frontend=provenance,scope='validation-only matched 6000/2000 pilot')
    save(args.out/'metadata.json',metadata)
    if args.phase=='smoke':smoke(frontend,cfg,optical_cfg,args.out)
    else:
        with np.load(args.data,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        assert not any(k.startswith('test') for k in arrays)
        train_data=tuple(torch.from_numpy(arrays['train_'+k]) for k in ['images','labels','domains'])
        val=tuple(torch.from_numpy(arrays['validation_'+k]) for k in ['images','labels','domains'])
        assert not set(arrays['train_ids'])&set(arrays['validation_ids'])
        for split,data in [('train',train_data),('validation',val)]:
            assert tuple(data[0].shape[1:])==(56,56,3)
            for domain in (0,1):assert torch.bincount(data[1][data[2]==domain],minlength=10).tolist()==[optical_cfg[split+'_pairs_per_class']]*10
        if args.phase=='train':
            (args.out/'frontend').mkdir()
            torch.save(dict(model=frontend.state_dict(),provenance=provenance),args.out/'frontend/best_checkpoint.pt')
            save(args.out/'frontend/summary.json',dict(**provenance,checkpoint_sha256=sha(args.out/'frontend/best_checkpoint.pt')))
            train(frontend,cfg,optical_cfg,arrays,train_data,val,args.out)
        else:
            assert args.run is not None
            original=json.loads((args.run/'metadata.json').read_text())
            assert original['data_sha256']==data_sha and original['config']==cfg and original['optical_config']==optical_cfg
            saved=torch.load(args.run/'frontend/best_checkpoint.pt',map_location='cpu',weights_only=False)
            assert tensors_sha(saved['model'])==tensors_sha(frontend.state_dict())
            audits=[]
            for architecture in cfg['architectures']:
                checkpoint_path=args.run/architecture/'best_checkpoint.pt';checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
                summary=json.loads((args.run/architecture/'summary.json').read_text())
                assert sha(checkpoint_path)==summary['checkpoint_sha256']
                assert checkpoint['frontend_tensors_sha256']==tensors_sha(frontend.state_dict())
                model=FrontendOptics(frontend,architecture,optical_cfg).cuda();model.optical.load_state_dict(checkpoint['model'])
                metrics,prob=base.evaluate(model,val,cfg)
                assert metrics==summary['validation']
                with (args.run/architecture/'validation_predictions.csv').open() as f:rows=list(csv.DictReader(f))
                assert [r['sample_id'] for r in rows]==arrays['validation_ids'].tolist()
                expected=np.array([[float(r[f'p{i}']) for i in range(10)] for r in rows],dtype=np.float32)
                assert np.array_equal(expected,prob)
                audits.append(dict(architecture=architecture,predictions_bitwise_identical=True,validation=metrics))
                del model;torch.cuda.empty_cache()
            save(args.out/'audit.json',dict(passed=True,samples_per_model=len(val[1]),results=audits))
            print(json.dumps(dict(passed=True,samples_per_model=len(val[1]))),flush=True)
    save(args.out/'status.json',dict(state='complete',phase=args.phase))


if __name__=='__main__':main()
