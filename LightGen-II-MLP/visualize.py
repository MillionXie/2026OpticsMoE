from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def configure_figures():
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'axes.titlesize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'figure.titlesize':8,'axes.linewidth':.6})


def export_results(records,metrics,directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'01_predictions.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sample_id','target_MOS','predicted_MOS'])
        writer.writeheader();writer.writerows(records)
    (directory/'03_prediction_vs_mos.png').unlink(missing_ok=True)


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
    figure.savefig(Path(directory)/'03_input_frames.png',dpi=300,bbox_inches='tight',pad_inches=.05)
    plt.close(figure)
