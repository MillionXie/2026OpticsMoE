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
    from .analyze_routing import COLORS
    from .data import batch,groups,load_inputs,sha256
    from .model import load_lightgen
else:
    from analyze_routing import COLORS
    from data import batch,groups,load_inputs,sha256
    from model import load_lightgen


def routing_bar(ax,probabilities,gates):
    x=np.arange(4)
    ax.bar(x-.18,probabilities*100,width=.36,color='#eeeeee',edgecolor='#666666',
           label='All-four softmax')
    ax.bar(x+.18,gates*100,width=.36,color=COLORS,label='Top2 after L1 renormalization')
    for position,value in zip(x+.18,gates*100):
        if value>0:
            ax.text(position,value+2,f'{value:.1f}',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=['E1','E2','E3','E4'],ylim=(0,105),ylabel='Weight (%)')
    ax.spines[['top','right']].set_visible(False)


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
    fig=plt.figure(figsize=(13,10),layout='constrained')
    grid=fig.add_gridspec(3,4,height_ratios=[1.2,1,1])
    for index in range(4):
        ax=fig.add_subplot(grid[0,index])
        ax.imshow(plt.imread(args.inputs/f'frame{index+1}.png'))
        ax.set_title(f'Frame {index+1}: image index {identity["frame_indices"][index]}')
        ax.axis('off')
        ax=fig.add_subplot(grid[1,index])
        routing_bar(ax,probabilities[index],gates[index])
        ax.set_title(f'Vision routing for frame {index+1}')
    ax=fig.add_subplot(grid[2,:2])
    routing_bar(ax,joint_probabilities,joint_gates)
    ax.set_title('Second-stage routing for this whole video')
    ax.legend(frameon=False,fontsize=9)
    ax=fig.add_subplot(grid[2,2:]);ax.axis('off')
    ax.text(.02,.9,'First stage: four separate routing decisions\n'
            'Each decision gates all 49 tokens of one frame.\n\n'
            'Merger -> four visual tokens\n'
            '+ 38 prompt tokens -> 42-token sequence\n\n'
            'Second stage: one routing decision\n'
            'Its two gates apply to all 42 sequence tokens.',
            ha='left',va='top',fontsize=11)
    fig.suptitle('One real input video, four frame decisions, one joint-stage decision\n'
                 'Gray bars sum to 100%; the two nonzero colored bars also sum to 100%',fontsize=15)
    args.out.mkdir(parents=True)
    fig.savefig(args.out/'routing_example.png',dpi=180)
    plt.close(fig)
    record={'input_identity':identity,'checkpoint_SHA256':sha256(args.weights),
            'feature_source':'matching original TEST cache; first field, physical slot 0',
            'frame_probabilities':probabilities.tolist(),'frame_gates':gates.tolist(),
            'joint_probabilities':joint_probabilities.tolist(),'joint_gates':joint_gates.tolist(),
            'data_identity':payload['identity'],'script_SHA256':sha256(Path(__file__)),
            'image_SHA256':{f'frame{k+1}.png':sha256(args.inputs/f'frame{k+1}.png') for k in range(4)}}
    (args.out/'routing_example.json').write_text(json.dumps(record,indent=2)+'\n','utf-8')


if __name__=='__main__':main()
