"""Draw user-supplied logical SHS corners on a RAW full-sensor image. No calibration writes."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw
from guarded_workflow import require_geometry

def render(image,config,out):
    c=json.loads(Path(config).read_text(encoding='utf-8-sig'))
    require_geometry({**c,'geometry_confirmed':True})
    im=Image.open(image)
    if im.size!=(1920,1080):raise ValueError('Use raw full-sensor SHS 1920x1080 image, not cropped/478x478/GUI screenshot')
    im=im.convert('RGB');draw=ImageDraw.Draw(im)
    labels=['top_left','top_right','bottom_right','bottom_left']
    points=[tuple(c['logical_corners_full_sensor_xy'][n]) for n in labels]
    draw.line(points+[points[0]],fill='yellow',width=3)
    for n,(x,y) in zip(labels,points):
        draw.ellipse((x-7,y-7,x+7,y+7),fill='red');draw.text((x+10,y+10),f'{n} ({x}, {y})',fill='yellow')
    dest=Path(out)
    if dest.exists():raise FileExistsError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest)
    return dest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(render(a.image,a.config,a.out))
