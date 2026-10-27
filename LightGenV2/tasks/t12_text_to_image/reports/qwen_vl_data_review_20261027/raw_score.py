"""Offline scoring only: GT files are never passed to the inference module."""
import argparse,csv,hashlib,json,math
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rgb(p):return np.asarray(Image.open(p).convert('RGB')).astype(np.float64)/255
def psnr(a,b):return -10*math.log10(max(float(np.square(a-b).mean()),1e-12))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('raw','run','inputs','mapping','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();report=read(args.raw/'report.json');inputs=read(args.inputs);mapping=read(args.mapping)
    assert report['complete'] and not report['target_or_mask_input'] and not report['reference_vae_encoder']
    assert [r['sample_id'] for r in inputs]==[r['sample_id'] for r in mapping]
    lookup={(r['sample_id'],r['variant']):r for r in report['records']};rows=[];differences=[]
    for i,(x,m) in enumerate(zip(inputs,mapping)):
        target=rgb(m['target']);stored=rgb(args.run/'images/val_best/target'/f"sample_{m['val_index']:04d}.png")
        if not np.array_equal(stored,target):raise ValueError('Raw evaluation GT identity differs')
        cached=rgb(args.run/'images/val_best/generated'/f"sample_{m['val_index']:04d}.png")
        for variant in ('normal','gray_vision','shuffled_text'):
            r=lookup[x['sample_id'],variant];path=Path(r['output'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=r['output_sha256']:raise ValueError('Raw PNG SHA differs')
            value=rgb(path);rows.append({'sample_id':x['sample_id'],'category':x['category'],'mode':x['mode'],
                'variant':variant,'psnr_png_db':psnr(value,target),'text_changed':r['text_changed']})
            if variant=='normal':differences.append({'sample_id':x['sample_id'],
                'mean_abs_8bit':float(np.abs(value-cached).mean()*255),'max_abs_8bit':float(np.abs(value-cached).max()*255)})
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'raw_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    categories=list(dict.fromkeys(x['category'] for x in inputs))
    selected=[next(i for i,x in enumerate(inputs) if (x['category'],x['mode'])==(cat,mode))
        for cat in categories for mode in ('background','object','joint')]
    reconstruction=all((args.raw/'qwen_reference_reconstruction'/f'sample_{i:04d}.png').exists() for i in selected)
    width=1024 if reconstruction else 1280
    sheet=Image.new('RGB',(width,310*len(selected)),'white');draw=ImageDraw.Draw(sheet)
    for j,i in enumerate(selected):
        x=inputs[i];y=j*310;draw.text((3,y+3),x['category']+'/'+x['mode']+' | GT is offline scoring only; ablations are separate forward passes',fill='black')
        if reconstruction:
            titles=('Reference RGB','Qwen reference reconstruction','Target (offline scoring only)','Raw RGB + text generation')
            paths=[Path(x['image']),args.raw/'qwen_reference_reconstruction'/f'sample_{i:04d}.png',Path(mapping[i]['target']),args.raw/'normal'/f'sample_{i:04d}.png']
        else:
            titles=('Reference RGB','Target (offline scoring only)','Raw RGB + text generation','Gray reference input','Wrong instruction')
            paths=[Path(x['image']),Path(mapping[i]['target']),args.raw/'normal'/f'sample_{i:04d}.png',args.raw/'gray_vision'/f'sample_{i:04d}.png',args.raw/'shuffled_text'/f'sample_{i:04d}.png']
        for col,title in enumerate(titles):draw.text((col*256+3,y+25),title,fill='black')
        for col,path in enumerate(paths):
            im=Image.open(path).convert('RGB')
            if im.size!=(256,256):raise ValueError('Unexpected image size')
            sheet.paste(im,(col*256,y+54))
    sheet.save(args.output/'raw_nine_mode_diagnostic.png')
    for j,category in enumerate(categories):sheet.crop((0,930*j,width,930*(j+1))).save(args.output/f'raw_{category}_diagnostic.png')
    full_n=read(args.run/'report.json')['best_val']['count']
    summary={'scope':f'{len(inputs)} development VAL cases; PNG-domain metrics, not the full {full_n} float-domain mean',
        'psnr_png_by_variant':{v:float(np.mean([r['psnr_png_db'] for r in rows if r['variant']==v])) for v in ('normal','gray_vision','shuffled_text')},
        'cached_vs_fresh_raw_png':differences,'inference_target_or_mask_input':False,
        'caution':'Gray/shuffled comparisons are interventions, not separately trained baselines. Cached-versus-fresh equality is measured, not assumed.',
        'checkpoint_sha256':report['checkpoint_sha256']}
    (args.output/'raw_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
