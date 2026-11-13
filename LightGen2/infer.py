"""Run the selected LightGen-II inference model on ordered videos."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

if __package__:
    from .model import load_lightgen
else:
    from model import load_lightgen


def frames_from_video(path):
    import cv2
    cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count<=0:
        cap.release()
        raise ValueError('Cannot decode video: '+str(path))
    frames=[]
    try:
        for fraction in np.linspace(.1,.9,4):
            cap.set(cv2.CAP_PROP_POS_FRAMES,min(count-1,int(round(fraction*(count-1)))))
            ok,frame=cap.read()
            if not ok:raise ValueError('Frame decoding failed: '+str(path))
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            h,w=rgb.shape[:2];side=max(2,int(round(min(h,w)*.65)));y=(h-side)//2;x=(w-side)//2
            rgb=cv2.resize(rgb[y:y+side,x:x+side],(448,448),interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(rgb.copy()).permute(2,0,1))
    finally:cap.release()
    return torch.stack(frames)


def iter_video_fields(video_paths):
    """Keep original order and cyclically pad from the beginning of the input."""
    if not video_paths:
        raise ValueError('At least one video is required')
    first_frames = []
    for start in range(0, len(video_paths), 16):
        chunk = video_paths[start:start+16]
        frames = [frames_from_video(path) for path in chunk]
        if not first_frames:
            first_frames = frames[:15]
        padding = [first_frames[i % len(first_frames)] for i in range(16-len(frames))]
        yield chunk, torch.stack(frames + padding)[None]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--videos',type=Path,nargs='+',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();model,meta=load_lightgen(a.weights,a.device)
    records=[]
    # New videos retain the same ordered field grouping/padding convention.
    # Optical slot dependence is reported by the separately saved slot audit.
    for chunk, pixels in iter_video_fields(a.videos):
        with torch.inference_mode():
            prediction=model(pixels=pixels.to(a.device))['prediction'][0]
        records.extend({'video':str(v),'slot':i,'MOS':float(prediction[i])} for i,v in enumerate(chunk))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'model':'public_surrogate','predictions':records},indent=2)+'\n','utf-8')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
