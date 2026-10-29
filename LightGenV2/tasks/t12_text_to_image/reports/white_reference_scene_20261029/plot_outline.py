"""Matched complete outline comparison; fixed samples and seeds, no quality pick."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
from PIL import Image,ImageDraw,ImageFont


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--sweep-run',default='white_reference_outline_sweep_20261029')
    p.add_argument('--variant-names',nargs=2,default=['scale035','scale065'])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('New review run required')
    assert json.loads((a.metrics/'report.json').read_text())['complete']
    variants=[('No outline',a.runs/'white_reference_sdxl_free_lighting_20261029'),
              ('Control '+a.variant_names[0],a.runs/a.sweep_run/a.variant_names[0]),
              ('Control '+a.variant_names[1],a.runs/a.sweep_run/a.variant_names[1])]
    scores=json.loads((a.metrics/'per_image.json').read_text());groups={}
    for label,root in variants:
        rows=[r for r in scores if r['run_id']==root.name];assert len(rows)==16
        for row in rows:assert sha(root/row['generated_PNG'])==row['generated_sha256']
        groups[label]=(root,{(r['reference_id'],r['condition_id'],r['seed']):r for r in rows})
    keys=sorted(groups['No outline'][1]);assert all(sorted(v[1])==keys for v in groups.values())
    for key in keys:
        reference=groups['No outline'][1][key]
        for _,rows in groups.values():assert (rows[key]['prompt'],rows[key]['input_sha256'])==(reference['prompt'],reference['input_sha256'])
        assert groups[variants[1][0]][1][key]['initial_latent_sha256']==groups[variants[2][0]][1][key]['initial_latent_sha256']
    font=ImageFont.load_default()
    for name in ['arial.ttf','DejaVuSans.ttf']:
        try:font=ImageFont.truetype(name,14);break
        except OSError:pass
    a.output.mkdir(parents=True)
    def panel(name,selected):
        canvas=Image.new('RGB',(1024,40+len(selected)*410),'white');d=ImageDraw.Draw(canvas)
        d.text((4,3),'AI-generated | native1024 outputs previewed256 | fixed full prompts; no best-of-seed selection',font=font,fill='black')
        for i,key in enumerate(selected):
            y=40+i*410;root,rows=groups['No outline'];row=rows[key]
            d.multiline_text((4,y),'\n'.join(textwrap.wrap(row['prompt'],137)),font=font,fill='black',spacing=3)
            d.text((4,y+65),f'{key[0]} / condition{key[1]} / seed{key[2]} | white ABO input',font=font,fill='black')
            with Image.open(root/(key[0]+'.png')) as im:canvas.paste(im,(0,y+145))
            for j,(label,(source,rs)) in enumerate(groups.items(),1):
                r=rs[key];x=j*256
                d.text((x+3,y+89),label,font=font,fill='black')
                d.text((x+3,y+107),f'CLIP-T {r["CLIP_T"]:.3f} / DINOv2 {r["DINOv2_I"]:.3f}',font=font,fill='black')
                d.text((x+3,y+125),f'Reward {r["ImageReward_compact"]:.3f}',font=font,fill='black')
                with Image.open(source/r['generated_PNG']) as im:canvas.paste(im.resize((256,256),Image.Resampling.LANCZOS),(x,y+145))
        canvas.save(a.output/name)
    for category in ['lamp','pillow']:
        for seed in [1042,1043]:
            selected=[k for k in keys if k[2]==seed and groups['No outline'][1][k]['category']==category]
            assert len(selected)==4;panel(f'{category}_all_requests_seed{seed}.png',selected)
    first={c:min(k[0] for k in keys if groups['No outline'][1][k]['category']==c) for c in ['lamp','pillow']}
    panel('fixed_four_comparison.png',[(first[c],condition,1042) for c in ['lamp','pillow'] for condition in ['0','1']])
    report={'complete':True,'source_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
            'source_sha256':sha(Path(__file__)),'metric_report_sha256':sha(a.metrics/'report.json'),
            'rows':scores,'files':{p.name:sha(p) for p in a.output.glob('*.png')},
            'scope':'All48 images retained in category/seed panels. Summary uses first source/category, both prompts,seed1042. Not independent TEST or attribute accuracy.',
            'noise_identity':'Explicit initial latent verified across two outline variants. Old no-outline run matched by integer seed without recorded initial latent hash.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(a.output.resolve())


if __name__=='__main__':main()
