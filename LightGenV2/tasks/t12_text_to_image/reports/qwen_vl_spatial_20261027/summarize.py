"""Same-A100 development comparison; fixed class/mode examples, no enhancement."""
import argparse,csv,hashlib,json
from pathlib import Path

from PIL import Image,ImageDraw


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run
    report=read(run/'report.json')
    if not report['complete'] or report['best_step']<=0 or report['test_evaluated']:
        raise ValueError('Need a completed, VAL-selected development checkpoint')
    groups={'parent_same_A100':report['val_before_paired_training'],
            'spatial':report['best_val'],'without_new_spatial':report['val_without_spatial']}
    ids=[r['sample_id'] for r in groups['spatial']['images']]
    if any([r['sample_id'] for r in group['images']]!=ids for group in groups.values()):
        raise ValueError('Comparison cohorts/order differ')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for model,group in groups.items():
        for mode in ('background','object','joint'):
            records=[r for r in group['images'] if r['mode']==mode]
            rows.append({'model':model,'mode':mode,'n':len(records),
              'psnr_db':group['by_mode'][mode],'lpips':group['lpips_by_mode'][mode],
              'edited_psnr_db':group['edited_psnr_by_mode'][mode],
              'retained_psnr_db':group['retained_psnr_by_mode'][mode]})
    with (args.output/'mode_metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    chosen=[]
    categories=list(dict.fromkeys(r['category'] for r in groups['spatial']['images']))
    category_rows=[]
    for model,group in groups.items():
        for category in categories:
            for mode in ('background','object','joint'):
                records=[r for r in group['images'] if (r['category'],r['mode'])==(category,mode)]
                if not records:continue
                row={'model':model,'category':category,'mode':mode,'n':len(records)}
                for key in ('psnr','lpips','edited_psnr','retained_psnr'):
                    values=[r[key] for r in records if r[key] is not None]
                    row[key]=sum(values)/len(values) if values else None
                category_rows.append(row)
    with (args.output/'category_mode_metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(category_rows[0]));writer.writeheader();writer.writerows(category_rows)
    for category in categories:
        for mode in ('background','object','joint'):
            chosen.append(next(i for i,r in enumerate(groups['spatial']['images']) if (r['category'],r['mode'])==(category,mode)))
    # The parent prediction is extracted from the saved LOSSLESS preview PNG,
    # not from the old RTX4090 export. Crop only: no scaling or sharpening.
    parent_grid=Image.open(run/'val_before_training.png').convert('RGB')
    if parent_grid.size!=(768,310*len(chosen)):raise ValueError('Unexpected saved parent preview geometry')
    panel=Image.new('RGB',(1280,310*len(chosen)),'white');draw=ImageDraw.Draw(panel);examples=[]
    for rownum,index in enumerate(chosen):
        r=groups['spatial']['images'][index];y=rownum*310;name=f'sample_{index:04d}.png'
        old=groups['parent_same_A100']['images'][index];zero=groups['without_new_spatial']['images'][index]
        draw.text((5,y+3),f"VAL {r['category']}/{r['mode']} | parent {old['psnr']:.2f} dB | spatial {r['psnr']:.2f} dB / LPIPS {r['lpips']:.3f} | no spatial {zero['psnr']:.2f} dB",fill='black')
        for col,title in enumerate(('Reference','Target','Parent, same A100','Spatial Qwen','New spatial injection OFF')):
            draw.text((col*256+5,y+25),title,fill='black')
        paths=[run/'images/val_best/reference'/name,run/'images/val_best/target'/name,
               None,run/'images/val_best/generated'/name,run/'images/val_without_spatial/generated'/name]
        provenance=[]
        for col,path in enumerate(paths):
            if path is None:
                image=parent_grid.crop((512,rownum*310+54,768,(rownum+1)*310))
                provenance.append({'file':str((run/'val_before_training.png').resolve()),'sha256':digest(run/'val_before_training.png'),'crop':[512,rownum*310+54,768,(rownum+1)*310]})
            else:
                image=Image.open(path).convert('RGB')
                provenance.append({'file':str(path.resolve()),'sha256':digest(path)})
            if image.size!=(256,256):raise ValueError('Expected native 256 pixels')
            panel.paste(image,(col*256,y+54))
        examples.append({'sample_id':r['sample_id'],'val_index':index,'prompt':r['prompt'],'sources':provenance})
    panel.save(args.output/'fixed_nine_mode_comparison.png')
    small=Image.new('RGB',(1280,930),'white')
    for j,i in enumerate(((m%len(categories))*3+m for m in range(3))):small.paste(panel.crop((0,i*310,1280,(i+1)*310)),(0,j*310))
    small.save(args.output/'fixed_three_mode_preview.png')
    summary={'scope':'Development VAL only; all generator comparisons on the same A100',
      'train_pairs':report['train_optimization_pairs'],'val_pairs':len(ids),'best_step':report['best_step'],
      'checkpoint_sha256':report['checkpoint_sha256'],'parameters':report['parameters'],
      'metrics':{name:{k:v for k,v in group.items() if k!='images'} for name,group in groups.items()},
      'mode_rows':rows,'category_mode_rows':category_rows,'examples':examples,
      'selection':'First sample of each category/mode, fixed independently of image quality',
      'ablation_note':'without_new_spatial leaves full Qwen multimodal semantic conditioning active; it is not removal of the whole vision tower',
      'image_note':'Native PNG tiles, parent crop from a lossless PNG; no enhancement. Float model outputs determine metrics.'}
    write=args.output/'summary.json';write.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for name,group in groups.items():print(name,group['mean_per_image_psnr_db'],group['mean_lpips'])


if __name__=='__main__':main()
