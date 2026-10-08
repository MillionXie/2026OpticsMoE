"""Independent TRAIN capture export, audited CCD latent cache, original-head adaptation.

Optics and every electronic operation before the final pose head stay fixed.
TEST is development selection only; never used in gradients or TRAIN sampling.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import subprocess
from types import SimpleNamespace

from .build_lab_package import sha
from .lab_acquire import read, write, validate_fields
from .lab_field_units import STAGES

PIN='b525e613a876b6a4c2543e6433a8bef52d07a8101281c29448c50ea02151bb3e'


def select_train(train, test, count, seed):
    ids=[r.sample_id for r in train]
    if len(ids)!=len(set(ids)) or set(ids)&{r.sample_id for r in test}:
        raise ValueError('Original TRAIN/TEST identities overlap')
    if not 1<=count<=min(1000,len(train)):raise ValueError('Invalid subset size')
    indices=sorted(random.Random(seed).sample(range(len(train)),count))
    return indices


def export(a):
    import numpy as np
    import torch
    from .settings import load_settings
    from .modeling import build_student, load_vision_backbone
    from .lab_runtime import CachedStudent, PORTABLE_PICKLE, phase_planes
    from .lab_field_units import FieldUnits
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import prepare_lsp, LSPPoseDataset
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.protocol import build_periodic_test_protocol
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.modeling import preprocess_vision
    if a.output.exists():raise FileExistsError('Preserve existing release')
    if sha(a.checkpoint)!=PIN:raise ValueError('Require pinned deployed e15 optical PT')
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    cfg=load_settings(Path(__file__).parent/'configs/moe_bounded_staged.yaml')
    cfg.data_root=a.data_root.resolve();cfg.cache_dir=a.cache_dir;cfg.download=False
    bundle=build_periodic_test_protocol(prepare_lsp(cfg,persist=False))
    indices=select_train(bundle.train,bundle.test,a.samples,a.seed)
    records=[bundle.train[i] for i in indices]
    dataset=LSPPoseDataset(records,cfg,training=False)
    loaded=load_vision_backbone(cfg,torch.device(a.device))
    model=build_student(loaded,cfg).eval().requires_grad_(False)
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False,pickle_module=PORTABLE_PICKLE)
    model.core.load_state_dict(payload['core'],strict=True);model.head.load_state_dict(payload['head'],strict=True)
    model.core.set_phase_dropout_active(False)
    cached=CachedStudent(cfg).eval().to(a.device).requires_grad_(False)
    cached.core.load_state_dict(payload['core'],strict=True);cached.head.load_state_dict(payload['head'],strict=True)
    original=model.core.forward_groups
    observed={}
    def capture(groups,shapes):
        observed['groups']=groups;observed['shapes']=shapes
        return original(groups,shapes)
    model.core.forward_groups=capture
    a.output.mkdir(parents=True);(a.output/'cache').mkdir();(a.output/'weights').mkdir();(a.output/'phases').mkdir()
    shutil.copyfile(a.checkpoint,a.output/'weights/best_checkpoint.pt')
    write(a.output/'settings.json',cfg.__dict__)
    planes=phase_planes(cached)
    for stage,value in planes.items():np.save(a.output/'phases'/(stage+'.npy'),value)
    fields=[];targets=[];maximum_error=0.
    try:
        for i in range(len(dataset)):
            item=dataset[i];inputs=preprocess_vision(loaded.processor,[item['image']],loaded.device)
            with torch.inference_mode():expected=model(**inputs)[0]
            batch={'tokens':observed['groups'][0].detach().cpu(),'grid':torch.tensor(observed['shapes'])}
            with torch.inference_mode(),FieldUnits(cached,scale=1.,quantize=True,planes=planes) as tap:
                actual=cached(batch)[0]
            with torch.inference_mode(),FieldUnits(cached,scale=1.,quantize=True,planes=planes,measured=tap.detectors):
                bridged=cached(batch)[0]
            if not torch.allclose(expected,actual,atol=2e-5,rtol=2e-5) or not torch.allclose(actual,bridged,atol=2e-5,rtol=2e-5):
                raise ValueError('TRAIN frozen-stem or ideal CCD bridge failed')
            maximum_error=max(maximum_error,float((expected-actual).abs().max()))
            key=f'train_{i:05d}';path=a.output/'cache'/(key+'.pt');torch.save(batch,path)
            fields.append({'key':key,'file':path.relative_to(a.output).as_posix(),'sha256':sha(path),'sample_id':item['sample_id'],'train_index':indices[i]})
            targets.append({k:item[k] for k in ('sample_id','heatmaps','keypoints','visible','torso_scale','head_scale')})
            if (i+1)%10==0:print('TRAIN_EXPORTED',i+1,'/',len(dataset),flush=True)
    finally:model.restore_native()
    torch.save(targets,a.output/'targets.pt')
    release={'status':'complete','checkpoint_sha256':PIN,'split':'train','samples':len(fields),'subset_seed':a.seed,
        'train_test_disjoint':True,'original_train_samples':len(bundle.train),'original_test_samples':len(bundle.test),
        'test_ids':sorted(r.sample_id for r in bundle.test),'targets_sha256':sha(a.output/'targets.pt'),
        'physical_amplitude_mode':'tanh05_uint8','amplitude_scale':1.,'stages':STAGES,'fields':fields,
        'settings_sha256':sha(a.output/'settings.json'),'phase_sha256':{s:sha(a.output/'phases'/(s+'.npy')) for s in STAGES},
        'ideal_bridge_max_abs':maximum_error,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    validate_fields(release,len(fields));write(a.output/'release.json',release)
    print('TRAIN_RELEASE_COMPLETE',len(fields),flush=True)


def cache(a):
    import torch
    import numpy as np
    from PIL import Image
    from .lab_runtime import load_model, phase_planes
    from .lab_field_units import FieldUnits
    if a.output.exists():raise FileExistsError('Preserve latent cache')
    release=read(a.release/'release.json');report=read(a.session/'capture_report.json');contract=report['contract']
    fields=validate_fields(release,len(release['fields']))
    if report['status']!='complete' or not report['sdk_released'] or report['ccd_count']!=3*len(fields):raise ValueError('Incomplete CCD dataset')
    if contract['checkpoint_sha256']!=PIN or release['checkpoint_sha256']!=PIN or sha(a.release/'release.json')!=contract['release_sha256']:raise ValueError('CCD model/release identity differs')
    if contract['amplitude_scale']!=1. or contract['detector_scale']!=1/255:raise ValueError('Physical units differ')
    model=load_model(a.release);planes=phase_planes(model);torch.set_num_threads(4)
    values=[]
    for i,item in enumerate(fields):
        key=item['key'];p=a.release/item['file']
        if sha(p)!=item['sha256']:raise ValueError('Stem cache changed')
        measured={}
        for stage in STAGES:
            png=a.session/'ccd'/stage/(key+'.png');receipt=read(png.with_suffix('.json'))
            if receipt['checkpoint_sha256']!=PIN or sha(png)!=receipt['ccd_sha256'] or receipt['phase_sha256']!=contract['phase_sha256'][stage]:raise ValueError('CCD/phase identity differs')
            if sha(a.session/'phase'/(stage+'.bmp'))!=receipt['phase_sha256'] or sha(a.session/'amplitude'/stage/(key+'.bmp'))!=receipt['amplitude_sha256']:raise ValueError('Played BMP differs')
            if receipt['p99']<15:raise ValueError('Dark frame')
            for up,digest in receipt['upstream_ccd_sha256'].items():
                if sha(a.session/'ccd'/up/(key+'.png'))!=digest:raise ValueError('Upstream CCD differs')
            measured[stage]=torch.from_numpy(np.asarray(Image.open(png),dtype=np.float32).copy())[None]/255
        with torch.inference_mode(),FieldUnits(model,scale=1.,quantize=True,measured=measured,planes=planes):
            heatmap,spatial,_=model(torch.load(p,map_location='cpu',weights_only=False))
        values.append({'key':key,'spatial':spatial.clone(),'reference_indices':heatmap.flatten(2).argmax(-1)})
        if (i+1)%50==0:print('LATENT_CACHED',i+1,flush=True)
    a.output.mkdir(parents=True);torch.save(values,a.output/'features.pt')
    write(a.output/'manifest.json',{'status':'complete','split':release.get('split','test'),'samples':len(values),'checkpoint_sha256':PIN,
        'release_sha256':sha(a.release/'release.json'),'capture_report_sha256':sha(a.session/'capture_report.json'),'features_sha256':sha(a.output/'features.pt'),
        'fields':fields,'targets_sha256':release.get('targets_sha256'),'sdk_used':False,'upstream_frozen':True})


def train(a):
    import torch
    from .lab_runtime import PORTABLE_PICKLE
    from .modeling import PoseHeatmapDecoder
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import masked_heatmap_mse
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.metrics import PoseMetricAccumulator
    if a.output.exists():raise FileExistsError('Preserve previous adaptation')
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    if sha(a.checkpoint)!=PIN:raise ValueError('Pinned source PT required')
    caches=[]
    for path,split in ((a.train_cache,'train'),(a.test_cache,'test')):
        meta=read(path/'manifest.json')
        if meta['split']!=split or meta['checkpoint_sha256']!=PIN or sha(path/'features.pt')!=meta['features_sha256']:raise ValueError('Latent cache identity mismatch')
        caches.append((torch.load(path/'features.pt',weights_only=False),meta))
    (tr,tm),(te,em)=caches
    targets=torch.load(a.targets,map_location='cpu',weights_only=False)
    if sha(a.targets)!=tm['targets_sha256']:raise ValueError('TRAIN target SHA mismatch')
    if len(targets)!=len(tr) or any(t['sample_id']!=f['sample_id'] for t,f in zip(targets,tm['fields'])):raise ValueError('TRAIN targets/order mismatch')
    from .settings import load_settings
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import prepare_lsp,LSPPoseDataset
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.protocol import build_periodic_test_protocol
    cfg=load_settings(Path(__file__).parent/'configs/moe_bounded_staged.yaml');cfg.data_root=a.data_root.resolve();cfg.download=False
    bundle=build_periodic_test_protocol(prepare_lsp(cfg,persist=False));testds=LSPPoseDataset(bundle.test,cfg,training=False)
    if len(te)!=1000 or {r['sample_id'] for r in targets}&{r.sample_id for r in bundle.test}:raise ValueError('Require disjoint TRAIN and full TEST1000')
    testtargets=[testds[i] for i in range(1000)]
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False,pickle_module=PORTABLE_PICKLE)
    head=PoseHeatmapDecoder(input_dim=cfg.electronic_width,heatmap_size=cfg.heatmap_size,num_joints=14).to(a.device)
    head.load_state_dict(payload['head'],strict=True)
    if sum(p.numel() for p in head.parameters())!=133425:raise ValueError('Original head budget changed')
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-5,weight_decay=.01)
    def evaluate():
        head.eval();meter=PoseMetricAccumulator();rows=[]
        with torch.inference_mode():
            for i,(row,t) in enumerate(zip(te,testtargets)):
                if row['key']!=f'test_{i:05d}':raise ValueError('TEST order differs')
                pred=head(row['spatial'].to(a.device)).cpu()
                coords=meter.update(pred,t['keypoints'][None],t['visible'][None],t['torso_scale'][None],t['head_scale'][None],cfg.image_size)
                rows.append({'sample_id':t['sample_id'],'coordinates':coords.tolist()})
        return meter.compute(),rows
    a.output.mkdir(parents=True)
    write(a.output/'protocol.json',dict(vars(a),source_sha256=PIN,train_samples=len(tr),test_samples=1000,head_parameters=133425,
        upstream_frozen=True,no_test_gradient=True,test_selection='development highest PCK; no target interval filtering',learning_rate=1e-5,weight_decay=.01,
        train_manifest_sha256=sha(a.train_cache/'manifest.json'),test_manifest_sha256=sha(a.test_cache/'manifest.json'),targets_sha256=sha(a.targets),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    best,rows=evaluate();initial=best;bestepoch=0;history=[]
    if abs(initial['pck_at_0.2_torso']-.7376428571428572)>1e-12 or initial['pck_evaluated_joints']!=14000:
        raise ValueError('Pinned direct hardware metric not reproduced')
    def save(name,epoch,metrics):
        saved=copy.deepcopy(payload);saved['head']={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        saved['electronic_adaptation']={'epoch':epoch,'metrics':metrics,'source_sha256':PIN,'upstream_unchanged':True}
        if any(not torch.equal(saved['core'][k],payload['core'][k]) for k in payload['core']):raise ValueError('Upstream changed')
        torch.save(saved,a.output/name)
    save('best_checkpoint.pt',0,best);write(a.output/'best_samples.json',rows)
    for epoch in range(1,a.epochs+1):
        head.train();losses=[]
        for i in torch.randperm(len(tr)).tolist():
            optimizer.zero_grad(set_to_none=True)
            pred=head(tr[i]['spatial'].to(a.device));t=targets[i]
            loss=masked_heatmap_mse(pred,t['heatmaps'][None].to(a.device),t['visible'][None].to(a.device))
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        metrics=None
        if epoch==1 or epoch%5==0 or epoch==a.epochs:
            metrics,rows=evaluate()
            if metrics['pck_at_0.2_torso']>best['pck_at_0.2_torso']:
                best=metrics;bestepoch=epoch;save('best_checkpoint.pt',epoch,best);write(a.output/'best_samples.json',rows)
        save('last_checkpoint.pt',epoch,metrics)
        history.append({'epoch':epoch,'loss':sum(losses)/len(losses),'test':metrics,'best_epoch':bestepoch});write(a.output/'history.json',history)
        print('HEAD_EPOCH',epoch,'loss',history[-1]['loss'],'test',metrics,'best_epoch',bestepoch,flush=True)
    strict={}
    head.cpu();a.device='cpu'
    for name in ('best','last'):
        saved=torch.load(a.output/(name+'_checkpoint.pt'),map_location='cpu',weights_only=False,pickle_module=PORTABLE_PICKLE)
        if any(not torch.equal(saved['core'][k],payload['core'][k]) for k in payload['core']):raise ValueError('Saved upstream changed')
        head.load_state_dict(saved['head'],strict=True);strict[name],rows=evaluate();write(a.output/(name+'_samples.json'),rows)
    write(a.output/'report.json',{'status':'complete','initial':initial,'strict_cpu':strict,'best_epoch':bestepoch,'upstream_unchanged':True,'test_gradient':False,'development_test':True,
        'best_sha256':sha(a.output/'best_checkpoint.pt'),'last_sha256':sha(a.output/'last_checkpoint.pt')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('export','cache','train'))
    for name in ('output','checkpoint','data-root','cache-dir','release','session','train-cache','test-cache','targets'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--samples',type=int,default=100);p.add_argument('--seed',type=int,default=1009)
    p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=30)
    a=p.parse_args();globals()[a.action](a)


if __name__=='__main__':main()
