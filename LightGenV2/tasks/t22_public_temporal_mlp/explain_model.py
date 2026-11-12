"""Draw dimensions and separate the two retained global stages explicitly."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax,x,y,width,height,text,color):
    ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0.08',
                               facecolor=color,edgecolor='#526575',linewidth=1))
    ax.text(x+width/2,y+height/2,text,ha='center',va='center',fontsize=11)


def arrow(ax,start,end):
    ax.annotate('',xy=end,xytext=start,
                arrowprops={'arrowstyle':'->','color':'#34495e','lw':1.5})


def draw_network(path):
    fig,ax=plt.subplots(figsize=(13,9),layout='constrained')
    ax.set(xlim=(0,13),ylim=(0,9));ax.axis('off')
    box(ax,.4,7.7,8.2,.8,'One video: four RGB frames -> extracted Qwen patch / position\n49 x 1024 per frame -> LN / linear adapter -> 49 x 192','#e4edf9')
    box(ax,.4,6.15,8.2,1.0,'VISION STAGE   (code: frame_*)\nEach frame: router -> Top2 linear experts -> optical global\nFour frames stay separate: 4 x 49 x 192','#d9e8f8')
    arrow(ax,(4.5,7.7),(4.5,7.2))
    box(ax,.4,4.8,8.2,.8,'Original Frame Merger: each 49 x 192 -> one 192-D token\nOne video: 4 x 192 visual tokens','#e4edf9')
    arrow(ax,(4.5,6.15),(4.5,5.65))
    box(ax,9.2,4.8,3.2,.8,'Fixed prompt\n38 x 192 text tokens','#fff0d6')
    box(ax,.4,3.55,12,.7,'Concatenate four visual tokens and 38 text tokens -> 42 x 192','#f2e5f4')
    arrow(ax,(4.5,4.8),(4.5,4.3));arrow(ax,(10.8,4.8),(10.8,4.3))
    box(ax,.4,1.9,12,1.1,'VISION-LANGUAGE STAGE   (schema 7, code: video_*)\nRouter and linear experts read all 42 image/text tokens\nRouter: LN -> Flatten 8064 -> FC128 -> ReLU -> FC4; optical global -> 42 x 192','#f2e5f4')
    arrow(ax,(6.4,3.55),(6.4,3.05))
    box(ax,.4,.5,12,.8,'Flatten 8064 -> Linear 256 -> ReLU -> Linear 5 -> softmax -> expected MOS','#e5f1e7')
    arrow(ax,(6.4,1.9),(6.4,1.35))
    ax.set_title('Two serial global operations per forward pass\nThe prompt enters after the first global and the Frame Merger',fontsize=16)
    fig.savefig(path,dpi=180);plt.close(fig)


def draw_frame(path):
    fig,ax=plt.subplots(figsize=(13,10),layout='constrained')
    ax.set(xlim=(0,13),ylim=(0,10));ax.axis('off')
    box(ax,3.5,8.7,6,.8,'ONE FRAME: 7 x 7 spatial token positions\n49 tokens, each with 192 channels -> X[49,192]','#e4edf9')
    box(ax,.4,6.7,5.4,1.2,'Router\nLN -> Flatten 9408 -> FC128 -> ReLU -> FC4\nsoftmax over four -> Top2 -> L1 renormalize','#fff0d6')
    box(ax,6.5,6.7,6.1,1.2,'Four independent linear experts\nEach: X[49,192] @ Wk.T[192,192]\nEach output Yk has shape [49,192]','#e5f1e7')
    arrow(ax,(5,8.7),(3.1,7.95));arrow(ax,(8.5,8.7),(9.5,7.95))
    box(ax,2.2,5.2,8.6,.8,'Y = g1 Y1 + g2 Y2 + g3 Y3 + g4 Y4\nTwo nonzero gates, shared by all 49 tokens of this frame -> 49 x 192','#f2e5f4')
    arrow(ax,(3.1,6.7),(4.2,6.05));arrow(ax,(9.5,6.7),(8.7,6.05))
    box(ax,2.2,3.8,8.6,.8,'Channel projection: Linear(192,27) -> ReLU\n49 x 192 -> 49 x 27','#e4edf9')
    arrow(ax,(6.5,5.2),(6.5,4.65))
    box(ax,2.2,2.35,8.6,.9,'Transpose: 49 x 27 -> 27 x 49\nToken-axis projection: Linear(49,27) -> ReLU\nResult: 27 x 27 nonnegative amplitudes','#e4edf9')
    arrow(ax,(6.5,3.8),(6.5,3.3))
    box(ax,2.2,.8,8.6,.9,'RMS normalize -> pack all 64 frame fields on the canvas\nOne full coherent propagation -> read this frame CCD -> 49 x 192','#d9e8f8')
    arrow(ax,(6.5,2.35),(6.5,1.75))
    ax.set_title('Frame token processing and optical encoding\nFour frames use this same set of parameters and make separate routing decisions',fontsize=16)
    fig.savefig(path,dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    draw_network(args.out/'network_overview.png')
    draw_frame(args.out/'frame_dimensions.png')


if __name__=='__main__':main()
