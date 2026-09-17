"""Display the first three validation images, independent of predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((a.data/'val_questions.json').read_text())
    keys = ['fixed_moe', 'fixed_d2nn', 'learned_moe', 'learned_d2nn']
    pred = {k: np.load(a.data/(k+'.npz'))['val'] for k in keys}
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), gridspec_kw={'width_ratios': [1, 1.8]})
    records = []
    for row, start in enumerate([0, 6, 12]):
        image_id = rows[start]['image_id']
        axes[row, 0].imshow(Image.open(a.data/image_id))
        axes[row, 0].set_title(image_id, fontsize=10)
        axes[row, 0].axis('off')
        ax = axes[row, 1]
        ax.axis('off')
        for j, i in enumerate([start, start+1]):
            item = rows[i]
            y = .98-j*.5
            truth = 'YES' if item['label'] else 'NO'
            ax.text(0, y, item['question'], fontsize=12, weight='bold', va='top')
            ax.text(0, y-.10, 'Ground truth: '+truth, fontsize=11, va='top')
            rec = dict(index=i, **item, predictions={})
            for n, k in enumerate(keys):
                probability = float(pred[k][i, 1])
                answer = int(pred[k][i].argmax())
                label = ('One-hot' if k.startswith('fixed') else 'GRU')+' / '+('MoE' if k.endswith('moe') else 'D2NN')
                ax.text(0 if n < 2 else .51, y-.22-(n%2)*.10,
                        f"{label}: {'YES' if answer else 'NO'} (Pyes={probability:.2f})",
                        fontsize=10, color='#20724d' if answer == item['label'] else '#b33d39', va='top')
                rec['predictions'][k] = dict(answer=answer, p_yes=probability)
            records.append(rec)
    fig.suptitle('First 3 validation images, first positive/negative query each | seed 17\nShared frozen visual CNN; 2 optical layers + per-layer OEO; green=correct, red=incorrect', fontsize=13)
    fig.text(.02, .015, 'Original CLEVR renders shown; model receives resized 64x64 images through the CNN. Source: Johnson et al., CLEVR (CC BY 4.0).', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    for ext in ['png', 'pdf']:
        fig.savefig(a.out/('validation_examples.'+ext), dpi=180)
    (a.out/'validation_examples.json').write_text(json.dumps(records, indent=2))
    plt.close(fig)


if __name__ == '__main__':
    main()
