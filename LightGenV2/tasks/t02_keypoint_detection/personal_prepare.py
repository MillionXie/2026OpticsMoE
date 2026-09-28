"""Non-destructive phone-photo ingest. No labels or splits are inferred as truth."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def prepare(source, output, overrides=None):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a fresh output directory; original photos never overwritten')
    output.mkdir(parents=True);(output/'images').mkdir();(output/'contact').mkdir()
    overrides={} if overrides is None else overrides
    rows=[]
    for index,path in enumerate(sorted(source.glob('*'))):
        if path.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
        with Image.open(path) as raw:
            exif=raw.getexif();size=list(raw.size)
            im=ImageOps.exif_transpose(raw).convert('RGB')
            rotation=int(overrides.get(path.name,0))
            if rotation not in (0,90,180,270):raise ValueError('Rotation is CCW 0/90/180/270')
            if rotation:im=im.rotate(rotation,expand=True)
            upright=list(im.size);im.thumbnail((1600,1600),Image.Resampling.LANCZOS)
            ident=f'photo_{index:03d}';rel=f'images/{ident}.jpg';im.save(output/rel,quality=95,subsampling=0)
            row={'id':ident,'original_name':path.name,'original_sha256':sha256(path),
                 'original_size_wh':size,'exif_orientation':exif.get(274,1),'camera_model':str(exif.get(272,'')),
                 'datetime_original':str(exif.get(36867,exif.get(306,''))),
                 'extra_rotation_ccw':rotation,'upright_size_wh':upright,'size_wh':list(im.size),
                 'image':rel,'image_sha256':sha256(output/rel),
                 'group_id':None,'split':None,'orientation_reviewed':False,'people':[]}
            rows.append(row)
    for start in range(0,len(rows),24):
        page=Image.new('RGB',(1200,6*235),'#eeeeee');draw=ImageDraw.Draw(page)
        for j,row in enumerate(rows[start:start+24]):
            im=Image.open(output/row['image']);im.thumbnail((290,202))
            x=(j%4)*300;y=(j//4)*235;page.paste(im,(x+(300-im.width)//2,y))
            draw.text((x+5,y+204),row['id']+' '+row['original_name'][:26],fill='black')
        page.save(output/'contact'/f'page_{start//24:02d}.jpg',quality=90)
    manifest={'schema_version':1,'source':str(source),'model_input_size_wh':[224,224],
              'policy':'upright full frames max1600; person crops later square224 without stretching; no labels yet',
              'joint_names':['right_ankle','right_knee','right_hip','left_hip','left_knee','left_ankle',
                             'right_wrist','right_elbow','right_shoulder','left_shoulder','left_elbow','left_wrist','neck','head_top'],
              'images':rows}
    (output/'annotations.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'photos':len(rows),'camera_models':dict(Counter(r['camera_model'] for r in rows)),
                      'output':str(output)},ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rotations',type=Path)
    a=p.parse_args();prepare(a.source,a.output,json.loads(a.rotations.read_text()) if a.rotations else None)
