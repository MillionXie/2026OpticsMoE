"""Independent RTMPose proposals; deliberately NOT human ground truth."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .personal_prepare import sha256

COCO_TO_LSP=[16,14,12,11,13,15,10,8,6,5,7,9]


def proposals(keypoints,scores,width,height):
    people=[]
    for idx,(xy,sc) in enumerate(zip(keypoints,scores)):
        good=sc>.4
        if good.sum()<6:continue
        lo=xy[good].min(0);hi=xy[good].max(0)
        # Keep substantial people, not small distant bystanders. Retain excluded proposals for audit.
        foreground=hi[1]-lo[1]>=height*.23
        points=[]
        for k in COCO_TO_LSP:
            x,y=map(float,xy[k]);v=int(sc[k]>.4 and 0<=x<width and 0<=y<height)
            points.append([round(x,3),round(y,3),v])
        neck=(xy[5]+xy[6])*.5
        # COCO has no LSP neck/head-top. Geometric placeholders always require review.
        face=xy[:5][sc[:5]>.4]
        head=face.mean(0) if len(face) else neck+(neck-(xy[11]+xy[12])*.5)*.4
        top=head+.6*(head-neck)
        for p in (neck,top):points.append([round(float(p[0]),3),round(float(p[1]),3),int(0<=p[0]<width and 0<=p[1]<height)])
        people.append({'id':f'p{idx:02d}','include':bool(foreground),'reviewed':False,'keypoints':points,
                       'label_source':'RTMPose independent preannotation, not GT','manual_required':[12,13],
                       'coco17_xy':xy.tolist(),'coco17_score':sc.tolist(),
                       'excluded_reason':None if foreground else 'small background person'})
    return people


def run(root):
    import cv2
    from rtmlib import Body
    root=Path(root);output=root/'preannotations.json'
    if output.exists():raise FileExistsError(output)
    data=json.loads((root/'annotations.json').read_text(encoding='utf-8'))
    model=Body(mode='balanced',backend='onnxruntime',device='cpu',to_openpose=False)
    for row in data['images']:
        im=np.asarray(Image.open(root/row['image']).convert('RGB'))[:,:,::-1].copy()
        xy,sc=model(im)
        row['people']=proposals(xy,sc,im.shape[1],im.shape[0])
        print(row['id'],len(row['people']),flush=True)
        # Interrupted run leaves resumable evidence, not a finalized dataset.
        (root/'preannotations.partial.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    data['preannotation']={'library':'rtmlib','mode':'balanced','models':Body.MODE['balanced'],
                           'independent_of_ours_and_qwen':True,'ground_truth':False,
                           'input_manifest_sha256':sha256(root/'annotations.json')}
    output.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);run(p.parse_args().dataset)
