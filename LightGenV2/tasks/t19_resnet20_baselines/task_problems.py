"""Task contracts for remaining ResNet-20 experiments; no pretrained model."""
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .lgvq_data import LGVQ

ROOT = Path(__file__).resolve().parents[3]
LGVQ_ROOT = ROOT / 'experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts'


class Problem:
    def __init__(self, task):
        self.task = task
        self.batch_sampler=None
        self.final_test=None
        self.minimize=False
        if not task.startswith('lgvq_'):
            self._other_task(task)
            return
        self.data = LGVQ(LGVQ_ROOT)
        target = 'temporal' if task == 'lgvq_temporal_resnet18' else task.removeprefix('lgvq_')
        self.train = self.data.dataset('train', target)
        self.test = self.data.dataset('test', target)
        self.train_mos = torch.tensor([float(r[target]) for r in self.data.rows if r['split']=='train'])
        self.identity = self.data.identity
        self.batch_size = 8
        self.metric = 'srcc'
        self.selection = 'TEST SRCC development; epoch1/every5/final'

    def configure_model(self, model, device):
        if self.task.startswith('lgvq_'):
            self.boundaries = model.set_train_score_range(self.train_mos).to(device)

    def loss(self, model, batch, device):
        if not self.task.startswith('lgvq_'):
            return self._other_loss(model,batch,device)
        result = model(batch['image'].to(device, non_blocking=True))
        labels = torch.bucketize(batch['score'].to(device), self.boundaries, right=False)
        return F.cross_entropy(result['quality_logits'], labels)

    @torch.inference_mode()
    def evaluate(self, model, loader, device):
        if not self.task.startswith('lgvq_'):
            return self._other_evaluate(model,loader,device)
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

    def _other_task(self,task):
        from . import other_data as data
        self.selection='TEST metric development; epoch1/every5/final'
        self.batch_size=16
        if task=='abo_image_to_image':
            self.train,self.test,self.identity=data.abo_data()
            self.gallery=self.train
            self.batch_sampler=data.ProductBatches(self.train.rows)
            self.batch_size=32;self.metric='hit_at_1'
            self.identity['training_sampler']='16_SKUs_x2_views; all1600 views once/epoch'
        elif task=='lsp':
            self.train,self.test,self.settings,self.identity=data.pose_data()
            self.metric='train_loss';self.minimize=True
            self.selection='TRAIN loss minimum; original teacher checkpoint policy; TEST observed development'
        elif task=='salicon':
            self.train,self.test,self.settings,self.identity=data.saliency_data()
            self.metric='cc'
        elif task=='openmoji':
            self.train,self.test,self.identity=data.openmoji_data()
            self.settings=SimpleNamespace(changed_cell_weight=12.,foreground_cell_weight=3.,
                category_loss_weight=1.,edit_loss_weight=1.25,preservation_loss_weight=.15,
                task_loss_weight=.1,ccd_loss_weight=0.,router_balance_weight=0.)
            self.metric='changed_cell_accuracy';self.batch_size=32
            self.identity['loss_weights']=vars(self.settings)
        elif task=='abo_conditional_edit':
            datasets,self.identity=data.editor_data()
            self.train,self.test,self.final_test=datasets['train'],datasets['val'],datasets['test']
            self.batch_size=8;self.metric='psnr_db';self.selection='VAL PSNR; fixed model TEST once'
        else:raise ValueError(f'Unknown remaining task: {task}')

    def _other_loss(self,model,batch,device):
        images=batch['image'].to(device,non_blocking=True)
        if self.task=='abo_image_to_image':
            from .retrieval import multi_positive_loss
            z=model(images)
            logits=z@z.T/.07
            diagonal=torch.eye(len(z),device=device,dtype=torch.bool)
            positives=torch.tensor([[a==b for b in batch['sku']] for a in batch['sku']],device=device)&~diagonal
            return multi_positive_loss(logits.masked_fill(diagonal,-1e4),positives)
        if self.task=='lsp':
            from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import masked_heatmap_mse,masked_coordinate_loss
            output=model(images)
            heat=masked_heatmap_mse(output,batch['heatmaps'].to(device),batch['visible'].to(device))
            coords=masked_coordinate_loss(output,batch['keypoints'].to(device),batch['visible'].to(device),224)
            return self.settings.heatmap_loss_weight*heat+self.settings.coordinate_loss_weight*coords
        if self.task=='salicon':
            from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import saliency_loss
            return saliency_loss(model(images),batch['density'].to(device),batch['fixation'].to(device),self.settings)[0]
        tokens=batch['tokens'].to(device)
        if self.task=='openmoji':
            from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.objectives import editing_objective
            output=model(images,tokens)
            output.update(ccd_operating_loss=images.new_zeros(()),router_balance_loss=images.new_zeros(()))
            moved={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            return editing_objective(output,moved,self.settings)['total']
        predicted=model(images,tokens)
        target=batch['target'].to(device)
        loss=F.mse_loss(predicted,target)
        for axis in (-1,-2):loss=loss+.05*F.l1_loss(torch.diff(predicted,dim=axis),torch.diff(target,dim=axis))
        return loss

    @torch.inference_mode()
    def _other_evaluate(self,model,loader,device):
        model.eval();records=[]
        if self.task=='abo_image_to_image':
            from torch.utils.data import DataLoader
            from .retrieval import exact_sku_ranking
            def encode(batches):
                zs,skus,ids=[],[],[]
                for batch in batches:
                    zs.append(model(batch['image'].to(device)).cpu());skus+=list(batch['sku']);ids+=list(batch['sample_id'])
                return torch.cat(zs),skus,ids
            gallery,gskus,gids=encode(DataLoader(self.gallery,batch_size=32,num_workers=0))
            query,qskus,qids=encode(loader)
            result=exact_sku_ranking(query@gallery.T,qskus,gskus)
            records={'query_ids':qids,'candidate_ids':gids,**result}
            return {k:v for k,v in result.items() if isinstance(v,(int,float))},records
        if self.task=='lsp':
            from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.metrics import PoseMetricAccumulator
            accumulator=PoseMetricAccumulator()
        elif self.task=='salicon':
            from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import SaliencyAccumulator
            accumulator=SaliencyAccumulator()
        elif self.task=='openmoji':
            from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
            accumulator=MetricAccumulator()
        for batch in loader:
            images=batch['image'].to(device,non_blocking=True)
            if self.task=='lsp':
                output=model(images)
                predicted=accumulator.update(output,batch['keypoints'],batch['visible'],batch['torso_scale'],batch['head_scale'],224)
                records.extend({'sample_id':sid,'prediction_xy':pred.tolist()} for sid,pred in zip(batch['sample_id'],predicted))
            elif self.task=='salicon':
                from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import correlation_coefficient,density_from_logits
                output=model(images);density=batch['density'].to(device);fixation=batch['fixation'].to(device)
                accumulator.update(output,density,fixation)
                predicted=density_from_logits(output)
                records.extend({'sample_id':sid,'cc':float(correlation_coefficient(p[None],y[None]))}
                    for sid,p,y in zip(batch['sample_id'],predicted,density))
            elif self.task=='openmoji':
                output=model(images,batch['tokens'].to(device))
                moved={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                rows,predicted,_=accumulator.update(output,moved)
                for row,pred in zip(rows,predicted):row['prediction_grid']=pred.cpu().tolist()
                records.extend(rows)
            else:
                from LightGenV2.tasks.t12_text_to_image.lab_shs8um.train_channel_robust import ssim_per_image
                predicted=model(images,batch['tokens'].to(device)).clamp(0,1)
                target=batch['target'].to(device).clamp(0,1)
                mse=(predicted-target).square().mean((1,2,3))
                psnr=-10*torch.log10(mse.clamp_min(1e-12));ssim=ssim_per_image(predicted,target)
                records.extend({'sample_id':sid,'mode':mode,'psnr_db':float(p),'ssim':float(s)}
                    for sid,mode,p,s in zip(batch['sample_id'],batch['mode'],psnr,ssim))
        if self.task=='abo_conditional_edit':
            return {key:sum(r[key] for r in records)/len(records) for key in ('psnr_db','ssim')},records
        metrics=accumulator.compute()
        if self.task=='openmoji':metrics=metrics['overall']
        return metrics,records
