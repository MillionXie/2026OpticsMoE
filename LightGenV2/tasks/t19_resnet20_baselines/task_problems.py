"""Task contracts for remaining ResNet-20 experiments; no pretrained model."""
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .lgvq_data import LGVQ

ROOT = Path(__file__).resolve().parents[3]
LGVQ_ROOT = ROOT / 'experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts'


class Problem:
    def __init__(self, task):
        self.task = task
        if not task.startswith('lgvq_'):
            raise NotImplementedError(f'Adapter not yet bound: {task}')
        self.data = LGVQ(LGVQ_ROOT)
        target = task.removeprefix('lgvq_')
        self.train = self.data.dataset('train', target)
        self.test = self.data.dataset('test', target)
        self.train_mos = torch.tensor([float(r[target]) for r in self.data.rows if r['split']=='train'])
        self.identity = self.data.identity
        self.batch_size = 8
        self.metric = 'srcc'
        self.selection = 'TEST SRCC development; epoch1/every5/final'

    def configure_model(self, model, device):
        self.boundaries = model.set_train_score_range(self.train_mos).to(device)

    def loss(self, model, batch, device):
        result = model(batch['image'].to(device, non_blocking=True))
        labels = torch.bucketize(batch['score'].to(device), self.boundaries, right=False)
        return F.cross_entropy(result['quality_logits'], labels)

    @torch.inference_mode()
    def evaluate(self, model, loader, device):
        from LightGenV2.tasks.t06_video_quality_assessment.quality_token_train import score_metrics
        model.eval()
        targets, predictions, records = [], [], []
        for batch in loader:
            predicted = model(batch['image'].to(device, non_blocking=True))['quality_score'].cpu()
            targets.append(batch['score'])
            predictions.append(predicted)
            records.extend({'sample_id': sid, 'target': float(y), 'prediction': float(p)}
                           for sid,y,p in zip(batch['sample_id'],batch['score'],predicted))
        metrics = score_metrics(torch.cat(targets), torch.cat(predictions))
        metrics = {k: v if math.isfinite(v) else None for k,v in metrics.items()}
        return metrics, records
