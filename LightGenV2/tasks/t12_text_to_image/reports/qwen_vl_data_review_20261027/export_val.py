"""Export actual VAL-best PNGs and metrics; data-only TAR, not a lab package."""
import argparse,csv,hashlib,io,json,tarfile
from pathlib import Path

from PIL import Image,ImageDraw


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run;report=read(run/'report.json');execution=read(run/'execution.json')
    if not report['complete'] or report['best_step']<=0 or report['test_evaluated']:raise ValueError('Require completed VAL-selected model')
    if digest(run/'best_checkpoint.pt')!=report['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    records=report['best_val']['images'];out=run/'overview';out.mkdir(exist_ok=True)
    with (out/'per_image_metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    files={};sheet=Image.new('RGB',(1536,286*((len(records)+5)//6)),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(records):
        for role in ('reference','target','generated'):
            name=f'{role}/sample_{i:04d}.png';path=run/'images/val_best'/name
            if Image.open(path).size!=(256,256):raise ValueError('Unexpected native resolution')
            files[name]={'sha256':digest(path),'bytes':path.stat().st_size,'sample_id':r['sample_id']}
        x=i%6*256;y=i//6*286
        draw.text((x+3,y+3),f"{i:02d} {r['category']}/{r['mode']} {r['psnr']:.2f}dB",fill='black')
        sheet.paste(Image.open(run/'images/val_best/generated'/f'sample_{i:04d}.png').convert('RGB'),(x,y+30))
    sheet.save(out/'all_val_generated.png')
    summary=out/'fixed_nine_mode_comparison.png'
    if summary.exists():
        panel=Image.open(summary).convert('RGB');categories=list(dict.fromkeys(r['category'] for r in records))
        for j,category in enumerate(categories):panel.crop((0,930*j,1024,930*(j+1))).save(out/f'{category}_three_modes.png')
    metadata={'scope':'Development VAL; all samples retained. Target PNGs are offline supervision/evaluation only, never inference inputs.',
        'count':len(records),'best_step':report['best_step'],'checkpoint_sha256':report['checkpoint_sha256'],
        'checkpoint_path':str((run/'best_checkpoint.pt').resolve()),'training_source_commit':execution['git']['published_source_commit'],
        'data_cache_sha256':execution['cache_sha256'],'export_source_sha256':digest(Path(__file__)),
        'native_resolution':[256,256],'image_note':'Raw model PNGs; no enhancement, sharpening or GT paste in generated outputs',
        'reference_target_source':'ABO CC BY 4.0 foregrounds and procedurally drawn backgrounds; retain original attribution metadata',
        'files':files}
    (out/'image_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    archive=out/'val_images.tar'
    with tarfile.open(archive,'w') as handle:
        handle.add(out/'image_manifest.json',arcname='image_manifest.json',recursive=False)
        for name in files:handle.add(run/'images/val_best'/name,arcname=name,recursive=False)
    (out/'image_archive_sha256.json').write_text(json.dumps({'file':archive.name,'sha256':digest(archive),'bytes':archive.stat().st_size},indent=2))
    print('Exported',len(records),'VAL cases,',len(files),'native images; data TAR bytes',archive.stat().st_size)


if __name__=='__main__':main()
