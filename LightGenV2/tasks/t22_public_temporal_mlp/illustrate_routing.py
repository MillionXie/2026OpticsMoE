"""Show one identified video's four frames and its actual routing weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__:
    from .plot_style import save_figure
    from .analyze_routing import COLORS
    from .data import batch,groups,load_inputs,sha256
    from .model import load_lightgen
else:
    from plot_style import save_figure
    from analyze_routing import COLORS
    from data import batch,groups,load_inputs,sha256
    from model import load_lightgen


def routing_bar(ax,probabilities,gates):
    x=np.arange(4)
    ax.bar(x,probabilities*100,width=.65,color=COLORS,
           edgecolor=['#222222' if gate>0 else '#cccccc' for gate in gates])
    for position,value in zip(x,probabilities*100):
        ax.text(position,value+2,f'{value:.1f}',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=['E1','E2','E3','E4'],ylim=(0,105),ylabel='Softmax probability (%)')
    ax.spines[['top','right']].set_visible(False)


def plot_example(identity,inputs,probabilities,gates,joint_probabilities,joint_gates,path):
    fig=plt.figure(figsize=(7.2,4.8),layout='constrained')
    grid=fig.add_gridspec(3,4,height_ratios=[1.2,1,1])
    for index in range(4):
        ax=fig.add_subplot(grid[0,index])
        ax.imshow(plt.imread(inputs/f'frame{index+1}.png'))
        ax.set_title(f'Frame {index+1}, index {identity["frame_indices"][index]}')
        ax.axis('off')
        ax=fig.add_subplot(grid[1,index])
        routing_bar(ax,probabilities[index],gates[index])
        ax.set_title(f'Vision router, frame {index+1}')
    ax=fig.add_subplot(grid[2,:2])
    routing_bar(ax,joint_probabilities,joint_gates)
    ax.set_title('Vision-language router, whole video')
    ax=fig.add_subplot(grid[2,2:]);ax.axis('off')
    ax.text(.02,.9,'Vision: one decision per frame, shared by 49 tokens.\n\n'
            'Merger: 4 visual + 38 text = 42 tokens.\n\n'
            'Vision-language: one decision for all 42 tokens.',
            ha='left',va='top')
    fig.suptitle('Four-expert softmax probabilities with Top2 outlines')
    save_figure(fig,path)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    identity=json.loads((args.inputs/'input_identity.json').read_text('utf-8'))
    model,metadata=load_lightgen(args.weights,'cpu')
    payload=load_inputs(json.loads(args.paths.read_text('utf-8')))
    if metadata['dataset']['manifest']['sha256']!=payload['identity']['manifest']['sha256']:
        raise ValueError('Checkpoint data identity mismatch')
    indices,valid=groups(payload,'test')
    if payload['ids'][indices[0,0]]!=identity['sample_id']:
        raise ValueError('Illustration image and cached feature IDs differ')
    item=batch(payload,indices[:1],valid[:1],'cpu')
    with torch.inference_mode():
        result=model(item['vision'])
    frame=result['routing']['frame']
    joint=result['routing']['video']
    probabilities=frame['probabilities'][0,0].numpy()
    gates=frame['weights'][0,0].numpy()
    joint_probabilities=joint['probabilities'][0,0].numpy()
    joint_gates=joint['weights'][0,0].numpy()
    args.out.mkdir(parents=True)
    plot_example(identity,args.inputs,probabilities,gates,
                 joint_probabilities,joint_gates,args.out/'routing_example.png')
    record={'input_identity':identity,'checkpoint_SHA256':sha256(args.weights),
            'feature_source':'matching original TEST cache; first field, physical slot 0',
            'frame_probabilities':probabilities.tolist(),'frame_gates':gates.tolist(),
            'joint_probabilities':joint_probabilities.tolist(),'joint_gates':joint_gates.tolist(),
            'data_identity':payload['identity'],'script_SHA256':sha256(Path(__file__)),
            'image_SHA256':{f'frame{k+1}.png':sha256(args.inputs/f'frame{k+1}.png') for k in range(4)}}
    (args.out/'routing_example.json').write_text(json.dumps(record,indent=2)+'\n','utf-8')


if __name__=='__main__':main()
