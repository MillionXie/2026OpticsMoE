"""Matched bounded-amplitude simulation stress; never actual CCD tuning."""
import argparse,copy,torch
from pathlib import Path
from transformers import AutoProcessor
from .model import OpticalRetrieval
from .io import write_json,sha256
from .retrieval_screen import load_screen,rank_instances
from .retrieval_adapt import encode_rows
from .robust_training import prepare,attach

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    runs=a.root/'LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation'
    assets=a.root/'.codex_tmp/t07_robust_assets_20260926'
    _,groups=load_screen(runs/'abo200_enrolled_protocol_20260913/protocol.json',a.root/'data/abo_similarity10_data')
    rows=groups['gallery']+groups['query'];processor=AutoProcessor.from_pretrained(str(assets/'processor'),local_files_only=True)
    cases={'initial_with_bounded_mapping':assets/'best.pt','bounded3':runs/'physical_bounded30_20260927/best.pt','bounded5':runs/'physical_bounded30_conv5_20260927/best.pt'}
    initial=prepare(torch.load(assets/'best.pt',map_location='cpu',weights_only=True),dict(robust_alpha_min=.30,bounded_amplitude=True,pixel_shift=1))
    common=initial['metadata']['optical_training_noise'];results={}
    for name,path in cases.items():
        payload=initial if name=='initial_with_bounded_mapping' else torch.load(path,map_location='cpu',weights_only=True)
        for pixels in (0,1):
            md=copy.deepcopy(payload['metadata']);md['optical_training_noise']=common
            model=OpticalRetrieval(md).cuda();model.load_state_dict(payload['state_dict'],strict=True);model.eval()
            attach(model,dict(pixel_shift=pixels));original=model.forward
            def noisy_forward(batch):
                for modality in (model.vision,model.language):
                    modality.optics.training=True;modality.optics.router.training=True;modality.optics.set_training_noise(True)
                return original(batch)
            model.forward=noisy_forward;torch.manual_seed(1042)
            vectors,routing=encode_rows(model,processor,rows,a.root/'data/abo_similarity10_data',torch.device('cuda'),4,True)
            metrics,_=rank_instances(vectors,rows);results[f'{name}_pixels{pixels}']=dict(metrics=metrics,router=routing,checkpoint_sha256=sha256(path))
            del model;torch.cuda.empty_cache()
            write_json(runs/'physical_bounded30_matched_stress_20260927.json',dict(status='running',scope='Same bounded input mapping, DC/CCD proxy noise and seed. Initial is original weights AFTER bounded mapping, not original legacy 0.830 clean baseline. No real CCD.',seed=1042,noise=common,results=results))
    write_json(runs/'physical_bounded30_matched_stress_20260927.json',dict(status='complete',scope='Matched bounded-amplitude stochastic simulation, NOT physical capture. 800 TEST queries against1600 TRAIN gallery; TEST-selected checkpoints, not independent validation.',seed=1042,noise=common,results=results))

if __name__=='__main__':main()
