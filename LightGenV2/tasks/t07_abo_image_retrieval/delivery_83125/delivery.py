"""Pinned ABO-200 handoff. Run from anywhere; paths resolve beside this file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
BEST='c9926cbaaa1ef066d9657a8028dffc130a33915aa9f392551573dc79894192d0'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    for name,identity in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path)!=identity['sha256']:
            raise RuntimeError('Missing/changed release file: '+name)
    if sha(ROOT/'assets/best.pt')!=BEST:raise RuntimeError('Wrong checkpoint')
    print(json.dumps({'verified_files':len(manifest['files']),'source_commit':manifest['source_commit'],'best_sha256':BEST}),flush=True)


def phase_export(output):
    import math
    import torch
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    payload=torch.load(ROOT/'assets/best.pt',map_location='cpu',weights_only=True)
    from standalone.model import OpticalRetrieval
    model=OpticalRetrieval(payload['metadata']);model.load_state_dict(payload['state_dict'],strict=True)
    phases={};planes={}
    for mode in ('vision','language'):
        branch=getattr(model,mode).optics
        raw={'router':branch.router.raw_router_phase,'global':branch.global_phase}
        raw.update({f'expert_{i}':p for i,p in enumerate(branch.experts)})
        for name,p in raw.items():phases[f'{mode}.{name}']=(2*math.pi*p.detach().float().sigmoid()).cpu()
        router=torch.zeros(478,478);router[127:351,127:351]=phases[f'{mode}.router']
        experts=torch.zeros(478,478)
        for i,(y,x) in enumerate(((0,0),(0,254),(254,0),(254,254))):
            experts[y:y+224,x:x+224]=phases[f'{mode}.expert_{i}']
        planes.update({f'{mode}.router':router,f'{mode}.expert':experts,f'{mode}.global':phases[f'{mode}.global']})
    output.mkdir(parents=True,exist_ok=False)
    torch.save({'checkpoint_sha256':BEST,'units':'radians','logical_pixel_pitch_um':17.,
                'phase_encoding':'2*pi*sigmoid(raw), NOT raw itself',
                'phases':phases,'active_planes':planes},output/'phases_radians.pt')
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for ax,(name,plane) in zip(axes.flat,planes.items()):
        im=ax.imshow(plane,vmin=0,vmax=2*math.pi,cmap='twilight');ax.set_title(name);ax.set_axis_off()
    fig.colorbar(im,ax=axes.ravel().tolist(),label='phase [rad]');fig.savefig(output/'phase_overview.png',dpi=160);plt.close(fig)
    (output/'README.txt').write_text('Logical 478x478 planes at17um. NOT physical SLM BMPs.\n'
        'Expert index0/1/2/3 = logical TL/TR/BL/BR. Outside support phase0; amplitude must define aperture.\n'
        'For8um SLM preserve physical dimensions; do not display these logical arrays1:1.\n'
        'Use calibrated phase LUT, panel center and optical homography; see AI_HANDOFF.md.\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['verify','evaluate','finetune','export-phase'])
    p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--steps',type=int,default=100)
    args=p.parse_args();verify()
    if args.command=='verify':return
    if args.output is None:p.error('--output is required and must be a NEW directory')
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    if args.command=='export-phase':phase_export(output);return
    module='retrieval_screen' if args.command=='evaluate' else 'retrieval_adapt'
    cmd=[sys.executable,'-m','standalone.'+module]
    if args.command=='evaluate':cmd+=['optical']
    cmd+=['--assets',str(ROOT/'assets'),'--data',str(ROOT/'data'),'--manifest',str(ROOT/'protocol.json'),
          '--checkpoint',str(ROOT/'assets/best.pt'),'--expected-checkpoint-sha256',BEST,
          '--batch-size','4','--device',args.device,'--output',str(output)]
    if args.command=='finetune':
        cmd+=['--multi-view','--refine-profile','sku_phase_head_top1','--epochs',str(args.epochs),
              '--steps',str(args.steps),'--eval-every','5','--classes-per-batch','4','--lr-scale','0.25']
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
