"""Four-query, six-capture physical smoke flow for pinned ABO-I2I 0.83125.

The run jointly screens phase encoding/orientation and canonical CCD orientation
against the fixed simulation for four predeclared queries, then executes the
real router->expert->global chain for vision and language.  CCD values remain
linear uint8; only one geometric warp is applied and no per-frame normalization
is used before model injection.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import AutoProcessor


HERE=Path(__file__).resolve()
PROJECT=HERE.parents[1]
ROOT=HERE.parents[2]
OLD=ROOT/'ABO_Lab_SHS_8um';ADAPTER=ROOT/'ABO_I2I_DVP_adapter_20260922'
sys.path[:0]=[str(PROJECT),str(OLD),str(ADAPTER)]
from lab_dvp import Camera  # noqa: E402
from phase_hdmi import PhaseHDMI  # noqa: E402
from vendor_driver import HoloeyeSLM  # noqa: E402
from standalone.io import inputs,picture  # noqa: E402
from standalone.model import OpticalRetrieval,fuse  # noqa: E402

BEST='c9926cbaaa1ef066d9657a8028dffc130a33915aa9f392551573dc79894192d0'
CAMERA_DLL=ROOT.parent/'小相机/SDK二次开发包/DVP2  SDK 中性版本/DVP2 SDK/library/Visual C++/bin/x64/DVPCamera64.dll'
PHASE_SDK=Path(r'C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\SDK')
PHASE_LUT=Path(r'C:\Program Files\Meadowlark Optics\Blink 1920 HDMI\LUT Files\19x12_8bit_linearVoltage.lut')
AMP_SDK=OLD/'vendor/holoeye_python';AMP_BIN=Path(r'C:\Program Files\HOLOEYE SLM SDK SlideshowPlayer 2.0')
BASE_CORNERS=np.float32([[900,142],[4310,142],[4297,3551],[874,3544]]) # screen TL,TR,BR,BL
ACTIVE=478;PHYSICAL=1016


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def pcc(a,b):
    x=np.asarray(a,dtype=np.float64).ravel();y=np.asarray(b,dtype=np.float64).ravel()
    x-=x.mean();y-=y.mean();d=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x.dot(y)/d) if d else 0.0


def camera_variants(a):
    return {'identity':a,'flip_h':np.fliplr(a),'flip_v':np.flipud(a),'rot180':np.rot90(a,2),
            'transpose':a.T,'rot90':np.rot90(a,1),'rot270':np.rot90(a,3),'anti_transpose':np.rot90(a.T,2)}


def warp(frame):
    dst=np.float32([[0,0],[477,0],[477,477],[0,477]])
    matrix=cv2.getPerspectiveTransform(BASE_CORNERS,dst)
    return cv2.warpPerspective(frame,matrix,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def orient(a,name):
    return camera_variants(a)[name].copy()


def active_to_native(active,kind='amplitude'):
    resized=cv2.resize(np.asarray(active), (PHYSICAL,PHYSICAL), interpolation=cv2.INTER_NEAREST)
    h=1080 if kind=='amplitude' else 1200;full=np.zeros((h,1920),np.uint8)
    x=(1920-PHYSICAL)//2;y=(h-PHYSICAL)//2;full[y:y+PHYSICAL,x:x+PHYSICAL]=resized
    return full


def phase_gray(radians,spatial,inverted):
    a=np.mod(np.asarray(radians,dtype=np.float32),2*math.pi)
    g=np.floor(a/(2*math.pi)*256).clip(0,255).astype(np.uint8)
    if 'h' in spatial:g=np.fliplr(g)
    if 'v' in spatial:g=np.flipud(g)
    full=active_to_native(g,'phase')
    return 255-full if inverted else full


def phase_planes(model):
    result={}
    for mode in ('vision','language'):
        optics=getattr(model,mode).optics
        router=np.zeros((478,478),np.float32)
        router[127:351,127:351]=(2*math.pi*torch.sigmoid(optics.router.raw_router_phase)).cpu().numpy()
        expert=np.zeros((478,478),np.float32)
        for raw,(y,x) in zip(optics.experts,((0,0),(0,254),(254,0),(254,254))):
            expert[y:y+224,x:x+224]=(2*math.pi*torch.sigmoid(raw)).cpu().numpy()
        global_phase=(2*math.pi*torch.sigmoid(optics.global_phase)).cpu().numpy()
        result.update({f'{mode}_router':router,f'{mode}_expert':expert,f'{mode}_global':global_phase})
    return result


def save_amplitudes(active,out,stage,ids):
    active=np.asarray(active,dtype=np.float32);scale=float(active.max())
    if not np.isfinite(active).all() or scale<=0:raise ValueError('Invalid amplitude')
    folder=out/'amplitude'/stage;folder.mkdir(parents=True,exist_ok=True);paths=[]
    for sample_id,a in zip(ids,active):
        gray=np.rint(np.clip(a/scale,0,1)*255).astype(np.uint8)
        path=folder/(sample_id+'.bmp');Image.fromarray(active_to_native(gray)).save(path);paths.append(path)
    return paths,scale


def stage_active(amplitude,weights=None):
    if weights is None:
        result=np.zeros((len(amplitude),478,478),np.float32);result[:,127:351,127:351]=amplitude.detach().float().cpu().numpy();return result
    field=amplitude.new_zeros(len(amplitude),518,518)
    for i,(y,x) in enumerate(((20,20),(20,274),(274,20),(274,274))):
        field[:,y:y+224,x:x+224]=amplitude.float()*weights[:,i,None,None].float().clamp_min(0)
    return field[:,20:498,20:498].cpu().numpy()


class Bench:
    def __init__(self,out,exposure,wait_ms,phase_paths):
        self.out=out;self.exposure=exposure;self.wait=wait_ms/1000;self.phase_paths=phase_paths
        self.amp=HoloeyeSLM(AMP_SDK,AMP_BIN,(1920,1080),None,True,True,5)
        self.phase=PhaseHDMI(PHASE_SDK,PHASE_LUT,settle_s=.8,pixel_format='rgba')
        self.camera=Camera(CAMERA_DLL);self.rows=[]
    def __enter__(self):
        self.phase.__enter__();self.amp.__enter__();self.camera.__enter__()
        self.settings=self.camera.settings(exposure=self.exposure,gain=1.0);return self
    def __exit__(self,*args):
        errors=[]
        for device in (self.camera,self.amp,self.phase):
            try:device.__exit__(*args)
            except Exception as e:errors.append(repr(e))
        if errors and args[0] is None:raise RuntimeError('; '.join(errors))
    def capture(self,stage,phase_path,amplitude_paths,ids,camera_orientation,save=True):
        receipt=self.phase.show(phase_path);self.amp.preload_files(amplitude_paths);values=[]
        folder=self.out/'ccd'/stage
        if save:folder.mkdir(parents=True,exist_ok=True)
        for sample_id,path in zip(ids,amplitude_paths):
            self.amp.display_file(path);time.sleep(self.wait)
            frames=[];metas=[]
            for _ in range(6):
                image,meta=self.camera.capture();frames.append(image);metas.append(meta)
            image=orient(warp(frames[-1]),camera_orientation);values.append(image)
            row={'stage':stage,'sample_id':sample_id,'phase_sha256':sha(phase_path),'amplitude_sha256':sha(path),
                 'exposure':self.settings,'wait_ms':self.wait*1000,'frame_ids':[m['frame_id'] for m in metas],
                 'mean':float(image.mean()),'p99':float(np.percentile(image,99)),'maximum':int(image.max()),
                 'saturation_fraction':float(np.mean(image==255)),'canonical_orientation':camera_orientation,
                 'no_photometric_normalization':True}
            self.rows.append(row);print(json.dumps(row),flush=True)
            if row['saturation_fraction'] > .01:
                raise RuntimeError(f"Capture saturated at {stage}/{sample_id}: {row['saturation_fraction']:.4%}")
            if save:
                Image.fromarray(image).save(folder/(sample_id+'.png'));write(folder/(sample_id+'.json'),row)
        return np.stack(values),receipt


def snapshot_simulation(model,batch):
    vector=model(batch).float().cpu();result={'descriptor':vector}
    for mode in ('vision','language'):
        branch=getattr(model,mode).optics
        result[f'{mode}_router']=branch.router.last['intensity'].float().cpu()
        result[f'{mode}_router_probabilities']=branch.router.last['probabilities'].float().cpu()
        result[f'{mode}_expert']=branch.last_ccd['expert'].float().cpu()
        result[f'{mode}_global']=branch.last_ccd['global'].float().cpu()
    return result


def calibrate_stage(bench,out,stage,candidates,amplitude_paths,ids,target,minimum_pcc=None):
    calibration=[];captured={};receipts={}
    for name,path in candidates.items():
        raw,receipt=bench.capture('cal_'+stage+'_'+name,path,amplitude_paths,ids,'identity',save=False)
        captured[name]=raw;receipts[name]=receipt
        for camera_name in camera_variants(raw[0]):
            values=np.stack([orient(x,camera_name) for x in raw])
            scores=[pcc(x,y.numpy()) for x,y in zip(values,target)]
            calibration.append({'phase_candidate':name,'camera_orientation':camera_name,
                                'mean_pcc':float(np.mean(scores)),'per_sample_pcc':scores})
    calibration.sort(key=lambda r:r['mean_pcc'],reverse=True);best=calibration[0]
    status='passed' if minimum_pcc is None or best['mean_pcc']>=minimum_pcc else 'failed'
    write(out/(stage+'_calibration.json'),{'schema':1,'status':status,'minimum_mean_pcc':minimum_pcc,
          'best':best,'top8':calibration[:8]})
    if minimum_pcc is not None and best['mean_pcc']<minimum_pcc:
        raise RuntimeError(f"{stage} calibration rejected: best mean PCC {best['mean_pcc']:.4f} < {minimum_pcc:.4f}")
    phase_name=best['phase_candidate'];camera_name=best['camera_orientation']
    chosen=np.stack([orient(x,camera_name) for x in captured[phase_name]])
    folder=out/'ccd'/stage;folder.mkdir(parents=True,exist_ok=True)
    for sample_id,image in zip(ids,chosen):Image.fromarray(image).save(folder/(sample_id+'.png'))
    selected=out/'phase'/(stage+'.bmp');Image.open(candidates[phase_name]).save(selected)
    print(json.dumps({'calibrated_stage':stage,'phase':phase_name,'camera':camera_name,
                      'mean_pcc':best['mean_pcc']}),flush=True)
    return chosen,receipts[phase_name],best,selected


def main():
    import argparse
    global BASE_CORNERS
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--exposure-us',type=float,required=True)
    p.add_argument('--wait-ms',type=float,default=240);p.add_argument('--simulation-bank',type=Path,required=True)
    p.add_argument('--minimum-calibration-pcc',type=float,default=.30)
    p.add_argument('--corners-tltrbrbl', type=float, nargs=8, metavar=('TL_X','TL_Y','TR_X','TR_Y','BR_X','BR_Y','BL_X','BL_Y'))
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if a.corners_tltrbrbl is not None:
        BASE_CORNERS=np.asarray(a.corners_tltrbrbl,dtype=np.float32).reshape(4,2)
    if not 100<=a.exposure_us<=20000 or not 150<=a.wait_ms<=500:raise ValueError('Unsafe exposure/wait bounds')
    if sha(PROJECT/'assets/best.pt')!=BEST:raise ValueError('Wrong best.pt')
    device=torch.device('cuda');payload=torch.load(PROJECT/'assets/best.pt',map_location='cpu',weights_only=True)
    model=OpticalRetrieval(payload['metadata']);model.load_state_dict(payload['state_dict'],strict=True)
    model.to(device).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(str(PROJECT/'assets/processor'),local_files_only=True)
    protocol=json.loads((PROJECT/'protocol.json').read_text(encoding='utf-8'))
    rows=[];seen_products=set()
    for row in protocol['rows']:
        if row['split']!='query' or row['product_id'] in seen_products:continue
        rows.append(row);seen_products.add(row['product_id'])
        if len(rows)==4:break
    if len(rows)!=4:raise ValueError('Protocol does not contain four distinct query products')
    ids=[r['sample_id'] for r in rows];images=[picture(PROJECT/'data'/r['image_path'],model.metadata.get('input_preprocessing','contain_white')) for r in rows]
    batch=inputs(processor,images,device)
    with torch.inference_mode():sim=snapshot_simulation(model,batch)
    np.savez_compressed(out/'simulation_reference.npz',**{k:v.numpy() for k,v in sim.items()})
    phases=phase_planes(model);phase_dir=out/'phase';phase_dir.mkdir()
    # Calibrate every trained plane independently. Router-only calibration cannot
    # resolve the four-quadrant expert layout on a mirrored/rotated physical bench.
    phase_candidates={}
    for stage,radians in phases.items():
        phase_candidates[stage]={}
        for spatial in ('none','h','v','hv'):
            for inverted in (False,True):
                name=spatial+('_inverse' if inverted else '_normal')
                path=phase_dir/(f'cal_{stage}_{name}.bmp')
                Image.fromarray(phase_gray(radians,spatial,inverted)).save(path);phase_candidates[stage][name]=path
    v=model.vision;patches=model.frontend.patches(batch['pixel_values'],len(ids));latent=v.input_norm(v.input_adapter(patches.float()))
    e1=v.blocks[0](latent);amplitude=v.optics.encode(latent);router_active=stage_active(amplitude)
    router_paths,router_scale=save_amplitudes(router_active,out,'vision_router',ids)
    with Bench(out,a.exposure_us,a.wait_ms,phase_candidates) as bench:
        chosen,receipt,best,selected=calibrate_stage(bench,out,'vision_router',phase_candidates['vision_router'],router_paths,ids,sim['vision_router'],a.minimum_calibration_pcc)
        stage_calibration={'vision_router':best};selected_phase={'vision_router':selected}
        ccd=torch.from_numpy(chosen).to(device).float();measured={'vision_router':ccd.cpu()};receipts={'vision_router':receipt}
        v.optics.router.measured_ccd=ccd;weights=v.optics.router(amplitude)
        routes={'vision':v.optics.router.last['probabilities'].float().cpu()}
        expert_active=stage_active(amplitude,weights);paths,scale=save_amplitudes(expert_active,out,'vision_expert',ids)
        raw,receipts['vision_expert'],stage_calibration['vision_expert'],selected_phase['vision_expert']=calibrate_stage(bench,out,'vision_expert',phase_candidates['vision_expert'],paths,ids,sim['vision_expert'])
        ccd=torch.from_numpy(raw).to(device).float();measured['vision_expert']=ccd.cpu()
        o1=v.optics.decode(ccd,latent.shape[1],latent.dtype,False);f1=fuse(e1,o1,v.block1_optical_fusion_logit,v.alpha_bounds)
        e2=v.blocks[1](f1);global_amp=v.optics.encode(f1);global_active=stage_active(global_amp,weights);paths,gscale=save_amplitudes(global_active,out,'vision_global',ids)
        raw,receipts['vision_global'],stage_calibration['vision_global'],selected_phase['vision_global']=calibrate_stage(bench,out,'vision_global',phase_candidates['vision_global'],paths,ids,sim['vision_global'])
        ccd=torch.from_numpy(raw).to(device).float();measured['vision_global']=ccd.cpu()
        o2=v.optics.decode(ccd,latent.shape[1],latent.dtype,True);f2=fuse(e2,o2,v.block2_optical_fusion_logit,v.alpha_bounds);vo=v.output_norm(f2)
        vision=(patches.float()+torch.sigmoid(v.residual_logit)*v.output_adapter(vo)).to(patches.dtype)
        image_features=model.frontend.merge(vision);emb=model.frontend.embed(batch['input_ids'])
        mask=batch['input_ids'].eq(model.metadata['image_token_id']).unsqueeze(-1).expand_as(emb);emb=emb.masked_scatter(mask,image_features.to(emb.dtype))
        l=model.language;latent=l.input_norm(l.input_adapter(emb.float()));e1=l.blocks[0](latent);amplitude=l.optics.encode(latent)
        paths,lrouter_scale=save_amplitudes(stage_active(amplitude),out,'language_router',ids)
        raw,receipts['language_router'],stage_calibration['language_router'],selected_phase['language_router']=calibrate_stage(bench,out,'language_router',phase_candidates['language_router'],paths,ids,sim['language_router'])
        ccd=torch.from_numpy(raw).to(device).float();measured['language_router']=ccd.cpu()
        l.optics.router.measured_ccd=ccd;weights=l.optics.router(amplitude);routes['language']=l.optics.router.last['probabilities'].float().cpu()
        paths,lescale=save_amplitudes(stage_active(amplitude,weights),out,'language_expert',ids)
        raw,receipts['language_expert'],stage_calibration['language_expert'],selected_phase['language_expert']=calibrate_stage(bench,out,'language_expert',phase_candidates['language_expert'],paths,ids,sim['language_expert'])
        ccd=torch.from_numpy(raw).to(device).float();measured['language_expert']=ccd.cpu()
        o1=l.optics.decode(ccd,latent.shape[1],latent.dtype,False);f1=fuse(e1,o1,l.block1_optical_fusion_logit,l.alpha_bounds);e2=l.blocks[1](f1)
        global_amp=l.optics.encode(f1);paths,lgscale=save_amplitudes(stage_active(global_amp,weights),out,'language_global',ids)
        raw,receipts['language_global'],stage_calibration['language_global'],selected_phase['language_global']=calibrate_stage(bench,out,'language_global',phase_candidates['language_global'],paths,ids,sim['language_global'])
        ccd=torch.from_numpy(raw).to(device).float();measured['language_global']=ccd.cpu()
        o2=l.optics.decode(ccd,latent.shape[1],latent.dtype,True);f2=fuse(e2,o2,l.block2_optical_fusion_logit,l.alpha_bounds);lo=l.output_norm(f2)
        descriptor=model.readout(lo,batch['input_ids'].eq(model.metadata['image_token_id'])).float().cpu()
        capture_rows=bench.rows
    bank=torch.load(a.simulation_bank,map_location='cpu',weights_only=True);id_to_row={r['sample_id']:r for r in protocol['rows']}
    # The manifest stores one copy of enrolled TRAIN rows; retrieval_screen
    # presents those same rows as the1600-item gallery without duplicating IDs.
    gallery_idx=[i for i,sid in enumerate(bank['ids']) if id_to_row[sid]['split']=='train'];gallery=F.normalize(bank['vectors'][gallery_idx].float(),dim=-1)
    gallery_ids=[bank['ids'][i] for i in gallery_idx];similarity=F.normalize(descriptor,dim=-1)@gallery.T
    top=similarity.argmax(1).tolist();predictions=[]
    for row,index in zip(rows,top):
        target=id_to_row[gallery_ids[index]];predictions.append({'sample_id':row['sample_id'],'product_id':row['product_id'],
            'top1_sample_id':target['sample_id'],'top1_product_id':target['product_id'],'hit_at_1':int(target['product_id']==row['product_id'])})
    stage_pcc={}
    for name,value in measured.items():stage_pcc[name]=[pcc(x,y.numpy()) for x,y in zip(value,sim[name])]
    router_best=stage_calibration['vision_router'];phase_name=router_best['phase_candidate'];camera_name=router_best['camera_orientation']
    spatial=phase_name.split('_')[0];inverted=phase_name.endswith('_inverse')
    report={'schema':1,'status':'complete','checkpoint_sha256':BEST,'samples':rows,'exposure_requested_us':a.exposure_us,
      'camera_actual':capture_rows[0]['exposure'],'settle_delay_ms':a.wait_ms,'base_corners_screen_TL_TR_BR_BL':BASE_CORNERS.tolist(),
      'phase_candidate':phase_name,'phase_spatial_transform':spatial,'phase_gray_inverted':inverted,
      'camera_canonical_orientation':camera_name,'stage_calibration':stage_calibration,
      'amplitude_common_scales':{'vision_router':router_scale,'vision_expert':scale,'vision_global':gscale,'language_router':lrouter_scale,'language_expert':lescale,'language_global':lgscale},
      'stage_pcc_to_simulation':stage_pcc,'descriptor_cosine_to_simulation':[float(x) for x in F.cosine_similarity(descriptor,sim['descriptor'])],
      'router_probabilities':{k:v.tolist() for k,v in routes.items()},'predictions_against_simulated_gallery':predictions,
      'diagnostic_hit_at_1':sum(x['hit_at_1'] for x in predictions)/len(predictions),'capture_rows':capture_rows,
      'phase_receipts':receipts,'no_per_image_photometric_normalization':True,
      'scope':'Four-query physical flow smoke. Simulated gallery is domain-mixed; not the full hardware R@1.'}
    torch.save({'ids':ids,'descriptors':descriptor,'measured_ccd':measured},out/'measured_features.pt');write(out/'report.json',report)
    print(json.dumps({'status':'complete','phase':phase_name,'camera_orientation':camera_name,'stage_pcc':stage_pcc,
                      'descriptor_cosine':report['descriptor_cosine_to_simulation'],'diagnostic_hit_at_1':report['diagnostic_hit_at_1']},indent=2),flush=True)


if __name__=='__main__':main()
