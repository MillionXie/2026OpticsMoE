"""Compare completed same-cohort development runs with fixed native-PNG rows.

No model execution, test selection or image enhancement is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(run):
    report=json.loads((run/'report.json').read_text(encoding='utf-8'))
    if not report.get('complete') or report['best_step']<=0:
        raise ValueError('Selected checkpoint is not ready: '+str(run))
    if report.get('test_evaluated'):
        raise ValueError('This report is restricted to development VAL')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    before,after=read_run(args.before),read_run(args.after)
    old,new=before['best_val']['images'],after['best_val']['images']
    if [r['sample_id'] for r in old]!=[r['sample_id'] for r in new]:
        raise ValueError('VAL cohorts/order differ')
    if before['train_optimization_pairs']!=after['train_optimization_pairs']:
        raise ValueError('Training cohorts differ')
    annotations={r['sample_id']:r for r in json.loads((args.after/'annotated_pairs.json').read_text(encoding='utf-8'))}
    chosen=[]
    for category in ('lamp','table','pillow'):
        for mode in ('background','object','joint'):
            chosen.append(next(i for i,r in enumerate(new) if (r['category'],r['mode'])==(category,mode)))
    args.output.mkdir(parents=True,exist_ok=True)
    panel=Image.new('RGB',(1024,316*len(chosen)),'white')
    draw=ImageDraw.Draw(panel);sources=[]
    for rownum,index in enumerate(chosen):
        name=f'sample_{index:04d}.png';y=rownum*316
        draw.text((5,y+3),f"VAL {new[index]['category']}/{new[index]['mode']} | fixed first example of each class/mode",fill='black')
        draw.text((5,y+19),f"Before {old[index]['psnr']:.2f} dB | After {new[index]['psnr']:.2f} dB, LPIPS {new[index].get('lpips',float('nan')):.3f}",fill='black')
        for col,label in enumerate(('Reference','Target','Before: full TRAIN best','After: semantic/GAN best')):
            draw.text((col*256+5,y+36),label,fill='black')
        record={'sample_id':new[index]['sample_id'],'val_index':index,
                'prompt':annotations[new[index]['sample_id']]['prompt'],'files':{}}
        for col,(run,role) in enumerate(((args.after,'reference'),(args.after,'target'),
                                         (args.before,'generated'),(args.after,'generated'))):
            path=run/'images/val_best'/role/name
            image=Image.open(path).convert('RGB')
            if image.size!=(256,256):raise ValueError('Native output must be 256x256')
            panel.paste(image,(col*256,y+60))
            record['files'][str(col)]={'file':str(path.resolve()),'sha256':digest(path)}
        for role in ('reference','target'):
            first=args.before/'images/val_best'/role/name
            second=args.after/'images/val_best'/role/name
            if digest(first)!=digest(second):raise ValueError('Reference/GT pixels differ')
        sources.append(record)
    panel.save(args.output/'fixed_nine_mode_comparison.png')
    # Three-category joint rows are a smaller, readable preview, with no rescale.
    joint=Image.new('RGB',(1024,316*3),'white')
    for j,i in enumerate((2,5,8)):
        joint.paste(panel.crop((0,316*i,1024,316*(i+1))),(0,316*j))
    joint.save(args.output/'fixed_joint_preview.png')
    records=[]
    for run,report in ((args.before,before),(args.after,after)):
        records.append({'run':str(run.resolve()),'best_step':report['best_step'],
          'val_count':report['best_val']['count'],'train_pairs':report['train_optimization_pairs'],
          'val_psnr_db':report['best_val']['mean_per_image_psnr_db'],
          'val_by_mode':report['best_val']['by_mode'],
          'val_lpips':report['best_val'].get('mean_lpips'),
          'val_lpips_by_mode':report['best_val'].get('lpips_by_mode'),
          'wrong_condition_psnr_db':report['val_wrong_condition']['mean_per_image_psnr_db'],
          'parameters':report['parameters'],'checkpoint_sha256':report['checkpoint_sha256'],
          'report_sha256':digest(run/'report.json')})
    # The refinement measures LPIPS for the unchanged incoming parent as well.
    baseline=after['val_before_paired_training']
    if abs(baseline['mean_per_image_psnr_db']-before['best_val']['mean_per_image_psnr_db'])>.01:
        raise ValueError('Refinement incoming-parent evaluation does not match the saved parent')
    summary={'scope':'TRAIN 3456 / VAL 144 development; no independent TEST claim',
      'runs':records,'incoming_parent_lpips':baseline.get('mean_lpips'),
      'incoming_parent_lpips_by_mode':baseline.get('lpips_by_mode'),
      'examples':sources,'selection':'First row of each category/mode; selected before quality inspection',
      'image_note':'Native 256 PNG tiles, no enhancement. PSNR is per-image mean on floating RGB, not on this PNG grid.',
      'safety_note':'SKU/exact RGBA identity isolation checked; incomplete family metadata still requires near-duplicate review.'}
    (args.output/'full_cohort_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for record in records:print(record['run'],record['val_psnr_db'],record['val_lpips'])


if __name__=='__main__':main()
