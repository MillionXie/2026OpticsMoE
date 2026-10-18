"""Reference-only CPU/native-BMP admission. This never releases physical capture."""
import argparse,hashlib,importlib,json,math,os,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT)]
from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images,sha,DIRECT,BOUNDED
from lab_caltech10_20261015 import native,phase_native


@torch.inference_mode()
def check(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')=='' and a.depth==2
    assert sha(a.cache)==a.cache_sha and sha(a.selection)==a.selection_sha
    selection=json.loads(a.selection.read_text());assert selection['state']=='preparation_only_not_capture_release'
    assert selection['cache_sha256']==a.cache_sha and selection['selected_train_samples']==1000
    with np.load(a.cache,allow_pickle=False) as z:
        labels=z['train_labels'];ids=z['train_ids'];indices=np.asarray(selection['train_indices'])
        assert len(indices)==1000 and len(np.unique(indices))==1000
        assert np.bincount(labels[indices],minlength=20).tolist()==[50]*20
        assert ids[indices].tolist()==selection['train_ids'] and labels[indices].tolist()==selection['train_labels']
        balanced=np.array([next(int(j) for j in indices if labels[j]==k) for k in range(20)])
        images=torch.from_numpy(z['train_images'][balanced].transpose(0,3,1,2).copy())
    assert not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(4)
    model,saved=load_checkpoint(a.checkpoint,a.checkpoint_sha,a.depth,a.rho,alpha_schema=a.alpha_schema)
    module=importlib.import_module(a.reference_module);reference=getattr(module,a.reference_class)(a.depth,a.rho).cpu().eval()
    reference.load_state_dict(saved['model'],strict=True);reference.requires_grad_(False)
    encoding=importlib.import_module('resisc20_input_encoder').encode
    pmax=cmax=fmax=fieldmax=route_max=encoder_max=0.;native_max=0.;proof=[]
    for offset in range(0,20,4):
        x=encode_images(images[offset:offset+4]);original=encoding(images[offset:offset+4],None,100)
        encoder_max=max(encoder_max,float((x-original).abs().max()));assert torch.equal(x,original)
        actual,c=model(x);expected,ce=reference(original)
        pmax=max(pmax,float((actual-expected).abs().max()));cmax=max(cmax,float((c-ce).abs().max()))
        fmax=max(fmax,float((model.readout['features']-reference.readout['features']).abs().max()))
        route_max=max(route_max,float((model.routing_weights-reference.routing_weights).abs().max()))
        stages=['router','0','1'];assert list(model.last)==list(reference.last)==stages
        for stage in stages:
            for key in ('amplitude','phase_gray','intensity'):
                fieldmax=max(fieldmax,float((model.last[stage][key].float()-reference.last[stage][key].float()).abs().max()))
        calls=[]
        def callback(stage,amplitude,phase,sample_ids):
            gray=np.rint(amplitude.clip(0,1)*255).astype(np.uint8)
            assert np.max(np.abs(gray/255-amplitude))<=1e-7
            display=native(gray[0],'amplitude')
            back=np.asarray(Image.fromarray(display[32:1048,452:1468]).resize((478,478),Image.Resampling.NEAREST));assert np.array_equal(back,gray[0])
            ph=phase_native(phase);back=np.flipud(np.fliplr(255-ph[92:1108,452:1468]))
            back=np.asarray(Image.fromarray(back).resize((478,478),Image.Resampling.NEAREST));assert np.array_equal(back,phase)
            field=torch.from_numpy(gray.astype(np.float32)/255)*torch.exp(2j*math.pi*torch.from_numpy(phase.astype(np.float32)/256))[None]
            calls.append(stage)
            return model.propagator(torch.nn.functional.pad(field,(20,20,20,20))).abs().square()[:,20:498,20:498].numpy()
        replay,_=model(x,callback=callback,ids=ids[balanced[offset:offset+4]])
        native_max=max(native_max,float((replay-actual).abs().max()));assert calls==stages
        proof.append(dict(TRAIN_samples=len(x),stages=calls))
    assert max(pmax,cmax,fmax,fieldmax,route_max,native_max)<=3e-6
    f=model.readout['features'].numpy();w=model.linear.weight.numpy();b=model.linear.bias.numpy()
    logits=f@w.T+b;logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
    head_error=float(np.max(np.abs(p-replay.numpy())));assert head_error<=3e-6
    result=dict(passed=True,scope='reference_PT_preparation_only',release_for_capture=False,checkpoint_sha256=a.checkpoint_sha,state_key='model',reference_module=a.reference_module,reference_class=a.reference_class,contract=model.contract(),alphas=model.alphas(),native_BMP_probability_max_error=native_max,reference_probability_max_error=pmax,reference_capture_max_error=cmax,reference_features_max_error=fmax,reference_fields_max_error=fieldmax,reference_router_max_error=route_max,encoder_max_error=encoder_max,NumPy_existing_linear_max_error=head_error,head_parameters=5140,TRAIN_images=20,TEST_images_read=0,VALID_images_read=0,optimizer_steps=0,SDK_opened=False,proof=proof,sources={p.name:sha(p) for p in (HERE/'lab_caltech20_rich_model_20261018.py',Path(__file__))})
    (a.out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('checkpoint','cache','selection','out'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('checkpoint-sha','cache-sha','selection-sha'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--depth',type=int,default=2);parser.add_argument('--rho',type=float,choices=(0.,.3),required=True)
    parser.add_argument('--alpha-schema',choices=(DIRECT,BOUNDED),required=True)
    parser.add_argument('--reference-module',required=True);parser.add_argument('--reference-class',required=True);args=parser.parse_args()
    try:check(args)
    except BaseException as error:
        if args.out.exists():(args.out/'failure.json').write_text(json.dumps(dict(error=repr(error),automatic_retry=False,SDK_opened=False)),encoding='utf-8')
        raise
