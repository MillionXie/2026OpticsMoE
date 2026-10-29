"""All matched DEV outputs with complete prompts and hash-bound scores."""
import argparse
import json
import math
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw
from .plot_condition_pilot import font
from ...white_reference_scene_probe import digest, write

MODELS = [('qwen_single_forward','Qwen single forward'),
          ('native_sdxl_ip','Native SDXL + IP'),
          ('native_sdxl_ip_canny','Native SDXL + IP + Canny')]


def panel(generation, rows, scores, summary, path):
    image = Image.new('RGB',(1280,170+len(rows)*468),'white');draw=ImageDraw.Draw(image)
    resolution=rows[0]['resolution']
    draw.text((8,4),f'Matched DEV | same RGB / initial noise / native{resolution} / 30 denoising steps | no image feedback',font=font(17),fill='black')
    draw.text((8,30),'8 held-out DEV products, 16 views, 32 requests/model. DEV selection data; independent TEST500 remains sealed.',font=font(13),fill='black')
    draw.text((8,51),'All scores higher is better; proxy similarities/preferences, NOT instruction accuracy. Qwen has not yet had full paired training.',font=font(12),fill='black')
    for i,(name,label) in enumerate(MODELS):
        s=summary[name+'/all']['scores']
        line=f"n=32 {label}: CLIP-T {s['CLIP_T']['mean']:.3f}, DINO-I {s['DINOv2_I']['mean']:.3f}, CLIP-I {s['CLIP_I']['mean']:.3f}, ImageReward {s['ImageReward_compact']['mean']:.3f}"
        draw.text((8,74+i*21),line,font=font(13),fill='black')
    for i,row in enumerate(rows):
        y=170+i*468
        draw.multiline_text((8,y),'\n'.join(textwrap.wrap(row['prompt'],172)),font=font(12),fill='black',spacing=1)
        draw.text((8,y+48),row['reference_id']+' | scene '+row['condition_id']+' | seed '+str(row['seed']),font=font(11),fill='black')
        draw.text((8,y+75),'White ABO input',font=font(13),fill='black')
        with Image.open(generation/MODELS[0][0]/(row['reference_id']+'.png')) as im:
            image.paste(im.convert('RGB').resize((236,236),Image.Resampling.LANCZOS),(8,y+130))
        for j,(name,label) in enumerate(MODELS):
            x=264+j*338;s=scores[(name,row['sample_id'])]
            draw.text((x,y+69),label,font=font(14),fill='black')
            draw.text((x,y+90),f"CLIP-T {s['CLIP_T']:.3f} | DINO-I {s['DINOv2_I']:.3f} | CLIP-I {s['CLIP_I']:.3f}",font=font(12),fill='black')
            draw.text((x,y+108),f"ImageReward {s['ImageReward_compact']:.3f}",font=font(13),fill='black')
            with Image.open(generation/name/s['generated_PNG']) as im:
                image.paste(im.convert('RGB').resize((328,328),Image.Resampling.LANCZOS),(x,y+130))
    image.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['generation','metrics','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('New plot directory required')
    assert json.loads((a.generation/'report.json').read_text())['generated_count']==96
    rows=json.loads((a.generation/MODELS[0][0]/'images.json').read_text())
    measured=json.loads((a.metrics/'per_image.json').read_text());assert len(measured)==96
    summary=json.loads((a.metrics/'summary.json').read_text())
    scores={(r['run_id'],r['sample_id']):r for r in measured};assert len(scores)==96
    for name,_ in MODELS:
        other=json.loads((a.generation/name/'images.json').read_text());assert len(other)==len(rows)==32
        for row,item in zip(rows,other):
            assert all(row[k]==item[k] for k in ['sample_id','prompt','seed','input_sha256','initial_latent_sha256'])
            s=scores[(name,row['sample_id'])]
            assert s['generated_sha256']==item['generated_sha256']==digest(a.generation/name/item['generated_PNG'])
            assert s['input_sha256']==digest(a.generation/name/(item['reference_id']+'.png'))
            assert all(math.isfinite(s[k]) for k in ['CLIP_T','DINOv2_I','CLIP_I','ImageReward_compact'])
    a.output.mkdir(parents=True);panels=[]
    for category in ['lamp','pillow']:
        selected=[r for r in rows if r['category']==category];assert len(selected)==16
        for page,start in enumerate(range(0,len(selected),4)):
            name=f'{category}_all_requests_page{page+1}.png'
            panel(a.generation,selected[start:start+4],scores,summary,a.output/name);panels.append(name)
    write(a.output/'plot_data.json',measured);write(a.output/'summary.json',summary)
    write(a.output/'report.json',{'complete':True,'source_sha256':digest(Path(__file__)),
        'n_outputs':96,'all_requests_shown':panels,'same_initial_latents_verified':True,
        'native_generation_resolution':rows[0]['resolution'],'images_only_resized_for_preview':True,'quality_selected_outputs':False,
        'DEV_not_independent_TEST':True,'score_source_sha256':digest(a.metrics/'report.json')})


if __name__=='__main__':main()
