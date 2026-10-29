"""Reference-aware contact boards for every teacher data candidate, no selection."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from PIL import Image,ImageDraw
from .plot_condition_pilot import font
from ...white_reference_scene_probe import digest,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('New QA directory required')
    report=json.loads((a.data/'report.json').read_text(encoding='utf8'));assert report['complete']
    rows=json.loads((a.data/'candidates.json').read_text(encoding='utf8'));assert len(rows)==report['candidate_count']
    groups=defaultdict(list)
    for row in rows:
        assert digest(a.data/row['reference'])==row['reference_sha256'] and digest(a.data/row['target'])==row['target_sha256']
        groups[row['reference_id']].append(row)
    a.output.mkdir(parents=True);panels=[];references=sorted(groups)
    width=(1+max(map(len,groups.values())))*180
    for page,start in enumerate(range(0,len(references),8)):
        selected=references[start:start+8];canvas=Image.new('RGB',(width,92+len(selected)*214),'white');draw=ImageDraw.Draw(canvas)
        teachers=sorted({r.get('teacher','unspecified teacher') for r in rows})
        draw.text((8,3),'SYNTHETIC TRAIN/DEV DATA CANDIDATES | '+' / '.join(teachers)+' | NOT Qwen output',font=font(13),fill='black')
        draw.text((8,26),'Input: white ABO RGB + Generate text. Coarse shape retained; fine details free. No pasted product pixels.',font=font(12),fill='black')
        draw.text((8,48),'All candidates included. These boards do not certify every prompt clause or constitute test results.',font=font(12),fill='black')
        for i,ref in enumerate(selected):
            records=sorted(groups[ref],key=lambda r:(int(r['condition_id']),r['seed_index']));y=92+i*214
            draw.text((4,y),ref,font=font(10),fill='black')
            with Image.open(a.data/records[0]['reference']) as im:canvas.paste(im.resize((176,176)),(2,y+32))
            for j,row in enumerate(records):
                x=(j+1)*180
                draw.text((x+2,y),row['split']+f" scene{row['condition_id']} draw{row['seed_index']}",font=font(11),fill='black')
                with Image.open(a.data/row['target']) as im:canvas.paste(im.resize((176,176)),(x+2,y+32))
        filename=f'all_reference_candidates_page{page+1}.png';canvas.save(a.output/filename);panels.append(filename)
    write(a.output/'report.json',{'complete':True,'source_sha256':digest(Path(__file__)),'candidate_count':len(rows),
        'every_candidate_shown':True,'panels':panels,'no_quality_selection':True,
        'candidate_manifest_sha256':digest(a.data/'candidates.json'),'teacher_outputs_not_Qwen_or_LG':True})


if __name__=='__main__':main()
