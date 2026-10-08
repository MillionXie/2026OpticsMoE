"""Bind audited replay argmax indices to the original LSP crop/PCK evaluator."""
import argparse
import json
from pathlib import Path
import torch
from .settings import load_settings
from .build_lab_package import sha
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import prepare_lsp, LSPPoseDataset
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.protocol import build_periodic_test_protocol
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.metrics import PoseMetricAccumulator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indices',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,help='Existing original dataset location; no download')
    p.add_argument('--expected-checkpoint-sha256',default='495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518')
    p.add_argument('--expected-reference-pck',type=float,default=.7347857142857143)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve metrics output')
    settings=load_settings(Path(__file__).parent/'configs/moe_optical_router_scale_matched_dc20_no_shift_warmstart.yaml')
    replay=json.loads(a.indices.with_name('report.json').read_text())
    if replay['status']!='complete' or replay['samples']!=1000 or replay['checkpoint_sha256']!=a.expected_checkpoint_sha256:
        raise ValueError('Replay checkpoint identity mismatch')
    if not 0 <= a.expected_reference_pck <= 1:
        raise ValueError('Invalid reference PCK')
    if a.data_root is not None:settings.data_root=a.data_root.resolve()
    settings.download=False
    bundle=build_periodic_test_protocol(prepare_lsp(settings,persist=False))
    dataset=LSPPoseDataset(bundle.test,settings,training=False)
    records=json.loads(a.indices.read_text())
    if len(records)!=1000:raise ValueError('Require original TEST1000')
    meters={k:PoseMetricAccumulator() for k in ('physical','reference')}
    samples=[]
    for i,row in enumerate(records):
        if row['key']!=f'test_{i:05d}':raise ValueError('Replay order mismatch')
        item=dataset[i]
        if row['shape']!=[1,14,settings.heatmap_size,settings.heatmap_size]:raise ValueError('Heatmap shape mismatch')
        entry={'key':row['key'],'sample_id':item['sample_id']}
        for name,meter in meters.items():
            indices=torch.tensor(row[name+'_indices'],dtype=torch.long)
            heatmaps=torch.zeros(row['shape']).flatten(2)
            heatmaps.scatter_(2,indices[:,:,None],1.)
            predicted=meter.update(heatmaps.reshape(row['shape']),item['keypoints'][None],item['visible'][None],
                torch.as_tensor(item['torso_scale'])[None],torch.as_tensor(item['head_scale'])[None],settings.image_size)
            entry[name+'_coordinates']=predicted.tolist()
        samples.append(entry)
    metrics={k:v.compute() for k,v in meters.items()}
    if metrics['reference']['pck_evaluated_joints']!=14000 or abs(metrics['reference']['pck_at_0.2_torso']-a.expected_reference_pck)>1e-12:
        raise ValueError('Original simulation/TEST order/target metric not reproduced')
    if metrics['physical']['pck_evaluated_joints']!=14000:raise ValueError('Physical joint count mismatch')
    a.output.mkdir(parents=True)
    report={'status':'complete','samples':1000,'indices_sha256':sha(a.indices),'metrics':metrics,
            'checkpoint_sha256':a.expected_checkpoint_sha256,'expected_reference_pck':a.expected_reference_pck,
            'metric_implementation':'original PoseMetricAccumulator and hardargmax_coordinates',
            'target_contract':'original LSPPoseDataset(training=False), original periodic TEST order',
            'no_test_gradient':True,'no_tta':True}
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    (a.output/'samples.json').write_text(json.dumps(samples))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
