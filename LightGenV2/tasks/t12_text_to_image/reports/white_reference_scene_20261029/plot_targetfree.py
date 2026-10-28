"""Fixed complete free-lighting previews with target-free diagnostic scores."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from PIL import Image, ImageDraw


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('New analysis run required')
    report=json.loads((a.run/'report.json').read_text(encoding='utf-8'))
    metric_report=json.loads((a.metrics/'report.json').read_text(encoding='utf-8'))
    rows=json.loads((a.metrics/'per_image.json').read_text(encoding='utf-8'))
    rows=[r for r in rows if r['run_id']==a.run.name]
    assert report['complete'] and metric_report['complete'] and len(rows)==16
    assert report['prompt_profile']=='free_lighting_v2'
    for row in rows:
        assert sha(a.run/row['generated_PNG'])==row['generated_sha256']
        assert 'light_side' not in row['requested_attributes']
    a.output.mkdir(parents=True)
    for category in ['lamp','pillow']:
        chosen=sorted({r['reference_id'] for r in rows if r['category']==category})
        sheet=Image.new('RGB',(768,35+4*370),'white');draw=ImageDraw.Draw(sheet)
        draw.text((4,4),'AI-generated | no unique GT | all references, both conditions, BOTH seeds | previews256 from native1024',fill='black')
        for i,(ref,condition) in enumerate((r,c) for r in chosen for c in ['0','1']):
            y=35+i*370;subset=sorted([r for r in rows if r['reference_id']==ref and r['condition_id']==condition],key=lambda r:r['seed'])
            assert len(subset)==2
            draw.multiline_text((4,y),'\n'.join(textwrap.wrap(subset[0]['prompt'],123)),fill='black')
            draw.text((4,y+70),ref+' | white ABO input',fill='black')
            with Image.open(a.run/(ref+'.png')) as im:sheet.paste(im,(0,y+110))
            for j,row in enumerate(subset):
                x=256*(j+1)
                draw.text((x+4,y+70),f'seed{row["seed"]} | CLIP-T {row["CLIP_T"]:.3f} | DINOv2 {row["DINOv2_I"]:.3f}',fill='black')
                draw.text((x+4,y+85),f'ImageReward compact {row["ImageReward_compact"]:.3f} (not accuracy)',fill='black')
                with Image.open(a.run/row['generated_PNG']) as im:sheet.paste(im.resize((256,256),Image.Resampling.LANCZOS),(x,y+110))
        sheet.save(a.output/(category+'_all8_full_prompts.png'))
    sheet=Image.new('RGB',(1024,1030),'white');draw=ImageDraw.Draw(sheet)
    draw.text((4,4),'AI-GENERATED | fixed first reference/category, BOTH prompts, seed1042 | no quality selection | native1024 outputs',fill='black')
    for col,category in enumerate(['lamp','pillow']):
        ref=min(r['reference_id'] for r in rows if r['category']==category)
        for condition in ['0','1']:
            row=next(r for r in rows if r['reference_id']==ref and r['condition_id']==condition and r['seed']==1042)
            x=col*512;y=26+int(condition)*500
            draw.multiline_text((x+4,y),'\n'.join(textwrap.wrap(row['prompt'],81)),fill='black')
            draw.text((x+4,y+73),f'CLIP-T {row["CLIP_T"]:.3f} | DINOv2 {row["DINOv2_I"]:.3f} | Reward {row["ImageReward_compact"]:.3f}',fill='black')
            with Image.open(a.run/(ref+'.png')) as im:sheet.paste(im.resize((128,128),Image.Resampling.LANCZOS),(x,y+235))
            with Image.open(a.run/row['generated_PNG']) as im:sheet.paste(im.resize((384,384),Image.Resampling.LANCZOS),(x+128,y+100))
    sheet.save(a.output/'fixed_four_requests.png')
    value={'source_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'argv':sys.argv,
           'source_sha256':sha(Path(__file__)),'source_report_sha256':sha(a.run/'report.json'),
           'metric_report_sha256':sha(a.metrics/'report.json'),'selection':'Full category panels contain all16; summary first source/category, both prompts,1042; no best-of-seed.',
           'rows':rows,'files':{p.name:sha(p) for p in a.output.glob('*.png')},'complete':True}
    (a.output/'report.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(a.output.resolve())


if __name__=='__main__':main()
