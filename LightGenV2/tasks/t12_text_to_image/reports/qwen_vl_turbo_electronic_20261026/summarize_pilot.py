"""Summarize development pilots and fixed example rows, never TEST results."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for tag in ('electronic','aligned','foreground','cross_attention'):
        run=a.runs/f'qwen_vl_turbo_{tag}_20261026'
        data=json.loads((run/'report.json').read_text())
        if not data.get('complete'):raise ValueError('Incomplete run '+str(run))
        history=json.loads((run/'history.json').read_text())
        records.append({'run_id':run.name,'selected_step':data['best_step'],
           'selected_train_psnr_db':data['train']['mean_per_image_psnr_db'],
           'selected_val_psnr_db':data['best_val']['mean_per_image_psnr_db'],
           'selected_val_by_mode':data['best_val']['by_mode'],
           'wrong_condition_val_psnr_db':data['val_wrong_condition']['mean_per_image_psnr_db'],
           'last_step':history[-1]['step'],'last_train_psnr_db':history[-1]['train_psnr'],
           'inference_parameters':data['parameters']['inference_total'],
           'trainable_parameters':data['parameters']['trainable'],
           'checkpoint_sha256':data['checkpoint_sha256'],'report_sha256':digest(run/'report.json'),
           'test_evaluated':data['test_evaluated'],'development_only':True})
    run=a.runs/'qwen_vl_turbo_cross_attention_20261026'
    chosen=(0,7,14)  # fixed before inspecting: lamp/background, table/object, pillow/joint
    panel=Image.new('RGB',(768,(298*3+36)*2),'white');draw=ImageDraw.Draw(panel)
    sources=[]
    for block,(name,title) in enumerate((('train_latest.jpg','TRAIN last step600 (not selected checkpoint)'),
                                        ('val_best.jpg','VAL selected checkpoint step200'))):
        image=Image.open(run/name)
        if image.size!=(768,5364):raise ValueError('Unexpected source grid geometry')
        y=block*(298*3+36)
        draw.text((8,y+8),title+' / reference | target | generated',fill='black')
        for j,i in enumerate(chosen):
            panel.paste(image.crop((0,i*298,768,(i+1)*298)),(0,y+36+j*298))
        sources.append({'file':str(run/name),'sha256':digest(run/name),'rows':list(chosen),'label':title})
    panel.save(a.output/'fixed_three_mode_examples.png')
    summary={'scope':'18 TRAIN + 18 VAL development pairs; not a usable baseline; no TEST evaluation',
             'decision':'Do not scale data or replace blocks with optics yet: held-out grounding/generalization gate failed',
             'runs':records,'examples':sources,
             'image_note':'Panels crop existing JPEG previews only; metrics come from raw floating outputs, not JPEGs',
             'parameter_note':'Deployment includes full Qwen, adapter, UNet including LoRA, and VAE decoder. CLIP and VAE encoder are training only.',
             'data_note':'Source and target roles are split-local; SKU/exact decoded pixel audits passed. Family metadata is incomplete; no claim of final near-duplicate audit.'}
    (a.output/'pilot_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for r in records:print(r['run_id'],r['selected_train_psnr_db'],r['selected_val_psnr_db'])


if __name__=='__main__':main()
