"""Fixed/all TRAIN request panels; no quality-based choice or enhanced PNGs."""
import argparse
import json
from pathlib import Path
import textwrap
from PIL import Image,ImageDraw,ImageFont
from ...white_reference_scene_probe import digest,write


def font(size):
    for path in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def draw_panel(data,teacher,student,rows,path):
    width=1056;height=80+len(rows)*358;out=Image.new('RGB',(width,height),'white');d=ImageDraw.Draw(out)
    d.text((8,4),'TRAIN pilot | conditions aligned only; not paired diffusion fine-tuned',fill='black',font=font(16))
    for i,row in enumerate(rows):
        y=64+i*358
        d.multiline_text((8,y),'\n'.join(textwrap.wrap(row['prompt'],135)),fill='black',font=font(13),spacing=2)
        d.text((8,y+58),row['reference_id']+' | source-derived seed '+str(row['seed']),fill='black',font=font(12))
        for x,label,image in [(8,'White ABO reference',data/row['reference']),
              (272,'Non-Qwen SDXL + Canny candidate',teacher/row['teacher_PNG']),
              (536,'Complete Qwen3-VL -> SDXL + IP',student/row['generated_PNG'])]:
            d.text((x,y+78),label,fill='black',font=font(11))
            with Image.open(image) as im:out.paste(im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS),(x,y+96))
        d.multiline_text((808,y+104),'No unique GT\nNo CLIP at Qwen inference\nNo reference VAE branch\nBoth raw outputs\nAll selected prompts retained',fill='black',font=font(12),spacing=4)
    out.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','teacher','student','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('New review directory required')
    t={r['sample_id']:r for r in json.loads((a.teacher/'images.json').read_text())};rows=json.loads((a.student/'images.json').read_text())
    assert len(rows)==16 and json.loads((a.student/'report.json').read_text())['complete']
    for row in rows:
        peer=t[row['sample_id']]
        assert all(row[n]==peer[n] for n in ['prompt','reference','seed','reference_sha256'])
        assert row['initial_latent_sha256']==peer['initial_noise_sha256']
        assert digest(a.student/row['generated_PNG'])==row['generated_sha256']
        assert digest(a.teacher/peer['generated_PNG'])==peer['generated_sha256']
        row['teacher_PNG']=peer['generated_PNG']
    a.output.mkdir(parents=True)
    chosen=[]
    for category in ['lamp','pillow']:
        available=[r for r in rows if r['category']==category]
        draw_panel(a.data,a.teacher,a.student,available,a.output/(category+'_all_eight.png'))
        refs=list(dict.fromkeys(r['reference_id'] for r in available))[:2]
        chosen.extend(next(r for r in available if r['reference_id']==ref) for ref in refs)
    draw_panel(a.data,a.teacher,a.student,chosen,a.output/'fixed_four_comparison.png')
    write(a.output/'plot_data.json',rows)
    write(a.output/'report.json',{'complete':True,'n_all_shown':len(rows),'n_fixed_shown':4,'selection':'first two references/category, first condition; no quality selection',
      'same_initial_noise_verified':True,'source_sha256':digest(Path(__file__)),'raw_image_pixels_not_enhanced':True,
      'limitations':'TRAIN condition warm-up, not independent scores, paired fine-tuned baseline, or isolated one-factor architecture ablation.'})


if __name__=='__main__':main()
