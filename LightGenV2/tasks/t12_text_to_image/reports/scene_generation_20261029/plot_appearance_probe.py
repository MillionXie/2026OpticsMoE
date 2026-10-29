"""All no-recolor requests with hash-bound per-image scores, no best-of selection."""
import argparse
import json
import math
from pathlib import Path
import textwrap
from PIL import Image,ImageDraw
from .plot_condition_pilot import font
from ...white_reference_scene_probe import digest,write
from ...appearance_scene_protocol import PROFILE


def score_text(row):
    return f"CLIP-T {row['CLIP_T']:.3f} | DINOv2-I {row['DINOv2_I']:.3f}",f"ImageReward {row['ImageReward_compact']:.3f}"


def panel(data,generation,rows,scores,summary,path,group='all'):
    out=Image.new('RGB',(960,138+len(rows)*468+28),'white');d=ImageDraw.Draw(out)
    d.text((8,4),'No color-change requests | Full Qwen3-VL -> SDXL | TRAIN development pilot',font=font(15),fill='black')
    d.text((8,28),'CLIP-T: text match | DINOv2-I: reference appearance | ImageReward: compact-caption preference',font=font(12),fill='black')
    d.text((8,47),'Higher is better; these scores are NOT instruction accuracy. Full request and all outputs retained.',font=font(12),fill='black')
    for j,name in enumerate(['existing','focused']):
        value=summary[name+'/'+group];s=value['scores']
        text=(f"Complete {group} group mean n={value['n']} | {name}: CLIP-T {s['CLIP_T']['mean']:.3f}, "
              f"DINOv2-I {s['DINOv2_I']['mean']:.3f}, ImageReward {s['ImageReward_compact']['mean']:.3f}")
        d.text((8,69+j*20),text,font=font(12),fill='black')
    for i,row in enumerate(rows):
        y=138+i*468
        d.multiline_text((8,y),'\n'.join(textwrap.wrap(row['prompt'],138)),font=font(12),fill='black',spacing=2)
        d.text((8,y+47),row['reference_id']+' | room '+row['condition_id']+' | seed '+str(row['seed']),font=font(11),fill='black')
        d.text((8,y+69),'Original white ABO reference',font=font(12),fill='black')
        with Image.open(data/row['reference']) as im:out.paste(im.convert('RGB').resize((208,208),Image.Resampling.LANCZOS),(8,y+160))
        for x,name,label in [(232,'existing','Existing Qwen | no recolor'),(584,'focused','Text-adapted Qwen | no recolor')]:
            score=scores[(name,row['sample_id'])]
            d.text((x,y+66),label,font=font(12),fill='black')
            for k,text in enumerate(score_text(score)):d.text((x,y+82+k*18),text,font=font(14),fill='black')
            with Image.open(generation/name/score['generated_PNG']) as im:out.paste(im.convert('RGB').resize((336,336),Image.Resampling.LANCZOS),(x,y+120))
    d.text((8,out.height-21),'No unique GT | Same initial noise | Native1024 RGB previewed; no pixel enhancement or quality-based selection',font=font(11),fill='black')
    out.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','generation','metrics','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('New scored-review directory required')
    report=json.loads((a.generation/'report.json').read_text());assert report['complete'] and report['prompt_profile']==PROFILE
    assert json.loads((a.metrics/'report.json').read_text())['complete']
    measured=json.loads((a.metrics/'per_image.json').read_text());summary=json.loads((a.metrics/'summary.json').read_text())
    assert len(measured)==32
    scores={(r['run_id'],r['sample_id']):r for r in measured};assert len(scores)==32
    left=json.loads((a.generation/'existing/images.json').read_text());right=json.loads((a.generation/'focused/images.json').read_text())
    assert len(left)==len(right)==16
    for l,r in zip(left,right):
        assert all(l[n]==r[n] for n in ['sample_id','prompt','seed','input_sha256','initial_latent_sha256'])
        for name,row in [('existing',l),('focused',r)]:
            s=scores[(name,row['sample_id'])]
            assert s['generated_sha256']==row['generated_sha256']==digest(a.generation/name/row['generated_PNG'])
            assert s['prompt']==row['prompt'] and s['prompt_profile']==PROFILE
            assert s['input_sha256']==digest(a.data/row['reference'])
            assert all(math.isfinite(s[n]) for n in ['CLIP_T','DINOv2_I','ImageReward_compact'])
    a.output.mkdir(parents=True);chosen=[];panels=[]
    for category in ['lamp','pillow']:
        rows=[r for r in left if r['category']==category];assert len(rows)==8
        for page in range(2):
            name=category+'_all_requests_page'+str(page+1)+'.png';panel(a.data,a.generation,rows[page*4:page*4+4],scores,summary,a.output/name,category);panels.append(name)
        refs=list(dict.fromkeys(r['reference_id'] for r in rows))[:2]
        chosen.extend(next(r for r in rows if r['reference_id']==ref) for ref in refs)
    panel(a.data,a.generation,chosen,scores,summary,a.output/'fixed_four_scored.png')
    write(a.output/'plot_data.json',measured);write(a.output/'summary.json',summary)
    write(a.output/'report.json',{'complete':True,'source_sha256':digest(Path(__file__)),'prompt_profile':PROFILE,
      'n_all_outputs':32,'all_sixteen_requests_shown_in_pages':panels,'fixed_four_selection':'first two references/category, first scene; not quality selected',
      'scores_bound_to_RGB_and_full_prompt':True,'metrics_per_image_and_group_means_on_figures':True,
      'metric_report_sha256':digest(a.metrics/'report.json'),'same_initial_noise_verified':True,'unique_GT_used':False,
      'limitations':'TRAIN pilot. New no-recolor task does not demonstrate color editing, independent generalization, paired diffusion quality or LG acceleration.'})


if __name__=='__main__':main()
