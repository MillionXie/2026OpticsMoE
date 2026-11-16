from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch


def configure_figures():
    font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'figure.titlesize':8,'axes.linewidth':.6})


def export_results(records,metrics,directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'01_predictions.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sample_id','target_MOS','predicted_MOS'])
        writer.writeheader();writer.writerows(records)
    target=np.asarray([row['target_MOS'] for row in records])
    prediction=np.asarray([row['predicted_MOS'] for row in records])
    figure,axis=plt.subplots(figsize=(3.3,3.1),layout='constrained')
    axis.scatter(target,prediction,s=9,alpha=.6,color='#6486a3',linewidths=0)
    lower,upper=min(target.min(),prediction.min()),max(target.max(),prediction.max())
    axis.plot([lower,upper],[lower,upper],color='#888888',linewidth=.7)
    score=f"{metrics['SRCC']:.2f}" if metrics['SRCC'] is not None else 'undefined'
    axis.set(xlabel='Human temporal MOS',ylabel='Predicted temporal MOS',title=f"{len(records)} videos with SRCC {score}")
    axis.spines[['top','right']].set_visible(False)
    figure.savefig(directory/'03_prediction_vs_mos.png',dpi=300,bbox_inches='tight',pad_inches=.05)
    plt.close(figure)


def export_input_frames(images,count,directory):
    frames=((images[0].float()+1)*127.5).round().clamp(0,255).to(torch.uint8)
    frames=frames.permute(0,1,3,4,2).cpu().numpy()
    figure,axes=plt.subplots(4,4,figsize=(6.6,7.0),layout='constrained')
    for index,axis in enumerate(axes.flat):
        samples=frames[index,:,::4,::4]
        mosaic=np.concatenate((np.concatenate(samples[:2],1),np.concatenate(samples[2:],1)),0)
        axis.imshow(mosaic)
        axis.set_title(f'Video {index+1}' if index<count else 'Padding')
        axis.axis('off')
    figure.suptitle('Input video frames')
    figure.savefig(Path(directory)/'04_input_frames.png',dpi=300,bbox_inches='tight',pad_inches=.05)
    plt.close(figure)
