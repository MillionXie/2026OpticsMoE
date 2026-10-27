"""Compare two completed experiments on the exact same reviewed development VAL."""
import argparse,csv,json
from pathlib import Path

from PIL import Image,ImageDraw


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('fused','native','output'):parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();a=read(args.fused/'report.json');b=read(args.native/'report.json')
    for r in (a,b):
        if not r['complete'] or r['test_evaluated']:raise ValueError('Need completed development VAL, not TEST')
    ea=read(args.fused/'execution.json');eb=read(args.native/'execution.json')
    if ea['cache_sha256']!=eb['cache_sha256']:raise ValueError('Original data/semantic cache identities differ')
    groups={'fused_parent_same_cohort':a['val_before_paired_training'],'fused_curated':a['best_val'],
        'native_initial_parent':b['val_before_paired_training'],'native_curated':b['best_val'],
        'native_spatial_off_at_inference':b['val_without_spatial']}
    ids=[r['sample_id'] for r in a['best_val']['images']]
    if any([r['sample_id'] for r in g['images']]!=ids for g in groups.values()):raise ValueError('VAL identities/order differ')
    for i in range(len(ids)):
        name=f'sample_{i:04d}.png'
        for role in ('reference','target'):
            pa=args.fused/'images/val_best'/role/name;pb=args.native/'images/val_best'/role/name
            if Image.open(pa).convert('RGB').tobytes()!=Image.open(pb).convert('RGB').tobytes():
                raise ValueError('Input/GT pixel identity differs: '+name)
    args.output.mkdir(parents=True,exist_ok=True);rows=[]
    for model,g in groups.items():
        for mode in ('all','background','object','joint'):
            images=[r for r in g['images'] if mode=='all' or r['mode']==mode]
            row={'model':model,'mode':mode,'n':len(images)}
            for key in ('psnr','lpips','edited_psnr','retained_psnr'):
                values=[r[key] for r in images if r[key] is not None]
                row[key]=sum(values)/len(values) if values else None
            rows.append(row)
    with (args.output/'mode_metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    chosen=[next(i for i,r in enumerate(b['best_val']['images']) if (r['category'],r['mode'])==(cat,mode))
            for cat in ('lamp','table','pillow') for mode in ('background','object','joint')]
    sheet=Image.new('RGB',(1024,310*9),'white');draw=ImageDraw.Draw(sheet);examples=[]
    for j,i in enumerate(chosen):
        old=a['best_val']['images'][i];new=b['best_val']['images'][i];y=j*310
        draw.text((3,y+3),f"{new['category']}/{new['mode']}  previous {old['psnr']:.2f}dB LPIPS {old['lpips']:.3f} | native {new['psnr']:.2f}dB LPIPS {new['lpips']:.3f}",fill='black')
        for col,title in enumerate(('Reference input','Target (supervision)','Previous fused readout','Native visual readout')):
            draw.text((col*256+3,y+26),title,fill='black')
        name=f'sample_{i:04d}.png'
        paths=[args.native/'images/val_best/reference'/name,args.native/'images/val_best/target'/name,
               args.fused/'images/val_best/generated'/name,args.native/'images/val_best/generated'/name]
        for col,path in enumerate(paths):
            image=Image.open(path).convert('RGB')
            if image.size!=(256,256):raise ValueError('Native resolution must remain 256')
            sheet.paste(image,(col*256,y+54))
        examples.append({'val_index':i,'sample_id':new['sample_id'],'prompt':new['prompt']})
    sheet.save(args.output/'all_nine_modes.png')
    sheet.crop((0,3*310,1024,6*310)).save(args.output/'table_three_modes.png')
    sheet.crop((0,0,1024,3*310)).save(args.output/'lamp_three_modes.png')
    sheet.crop((0,6*310,1024,9*310)).save(args.output/'pillow_three_modes.png')
    result={'scope':'Development VAL only; same 72 identities and exact input/GT pixels. No independent TEST claim.',
      'counts':{'train':a['train_optimization_pairs'],'val':len(ids)},'source_cache_sha256':ea['cache_sha256'],
      'fused_checkpoint_sha256':a['checkpoint_sha256'],'native_checkpoint_sha256':b['checkpoint_sha256'],
      'parameters':b['parameters'],'mode_metrics':rows,'examples':examples,
      'selection':'First fixed example in each category/mode; all 72 predictions retained in source runs',
      'interpretation':'Native variant also recalibrates TRAIN conditions and changes spatial injection LR; not an isolated causal ablation. Spatial OFF is inference intervention only.',
      'figures':'Native pixels, no enhancement, sharpening, retrieval or GT compositing in generated outputs'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for row in rows:
        if row['mode']=='all':print(row)


if __name__=='__main__':main()
