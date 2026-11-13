"""Create a synthetic video for the standalone inference example."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def create_video(output):
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),24.,(320,320))
    if not writer.isOpened():
        raise RuntimeError('MP4 video encoder is unavailable')
    y,x=np.mgrid[:320,:320]
    try:
        for frame in range(96):
            image=np.stack(((x+frame)%256,(y+frame//2)%256,(x//2+y//2)%256),axis=-1).astype(np.uint8)
            center=(60+frame*2,160+int(35*np.sin(frame/12)))
            cv2.circle(image,center,28,(40,220,245),-1)
            cv2.rectangle(image,(110,205),(210,260),(220,70,50),-1)
            writer.write(image)
    finally:
        writer.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    create_video(parser.parse_args().output)
