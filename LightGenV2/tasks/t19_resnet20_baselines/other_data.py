"""Original task data only; trained features and model targets never enter inputs."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image,ImageOps
import torch
from torch.utils.data import Dataset,Sampler

from .lgvq_data import sha
from .models import tokenize_text

ROOT=Path(__file__).resolve().parents[3]


def cnn(image):
    values=torch.from_numpy(np.array(image.convert('RGB'),dtype=np.float32,copy=True)).permute(2,0,1)/255
    return normalize(values)


def normalize(values):
    return (values-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


def fixed_tokens(text):
    values=tokenize_text([text])[0]
    return torch.nn.functional.pad(values,(0,514-len(values)))


def safe_path(root,relative):
    path=(root/relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):raise ValueError('Image escapes dataset root')
    return path


class ImageRows(Dataset):
    def __init__(self,root,rows):self.root,self.rows=root,rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(safe_path(self.root,r['image_path'])) as source:
            image=ImageOps.fit(ImageOps.exif_transpose(source).convert('RGB'),(224,224),method=Image.Resampling.BICUBIC)
            pixels=cnn(image)
        return {'image':pixels,'sample_id':r['sample_id'],'sku':r['product_id']}


class ProductBatches(Sampler):
    """All eight TRAIN views/sku once per epoch, paired positives in each batch."""
    def __init__(self,rows):
        self.groups=defaultdict(list)
        for i,r in enumerate(rows):self.groups[r['product_id']].append(i)
        if set(map(len,self.groups.values()))!={8}:raise ValueError('Expected eight TRAIN views/SKU')
        self.epoch=0
    def set_epoch(self,epoch):self.epoch=epoch
    def __iter__(self):
        rng=random.Random(42+self.epoch)
        groups={k:rng.sample(v,len(v)) for k,v in self.groups.items()}
        pairs=[]
        for round in range(4):
            keys=list(groups);rng.shuffle(keys)
            pairs.extend(groups[k][round*2:round*2+2] for k in keys)
        for start in range(0,len(pairs),16):
            batch=[i for pair in pairs[start:start+16] for i in pair]
            rng.shuffle(batch);yield batch
    def __len__(self):return (4*len(self.groups)+15)//16


def abo_data():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retail_sources import shape_groups
    p=ROOT/'LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/abo200_enrolled_protocol_20260913/protocol.json'
    protocol=json.loads(p.read_text())
    if sha(p)!='f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde':
        raise ValueError('Original ABO200 enrolled split SHA differs')
    root=ROOT/'data/abo_similarity10_data'
    if (protocol['protocol']!='abo200_enrolled_sku_hash8train4query_v1'
            or sha(root/'data/abo_similarity10_manifest.csv')!=protocol['parent_manifest_sha256']):
        raise ValueError('ABO200 original parent/protocol differs')
    groups=shape_groups(protocol['rows'])
    if (len(groups['train']),len(groups['query']))!=(1600,800):raise ValueError('ABO200 count differs')
    for row in protocol['rows']:
        if sha(safe_path(root,row['image_path']))!=row['image_sha256']:raise ValueError('ABO image SHA differs')
    train,test=ImageRows(root,groups['train']),ImageRows(root,groups['query'])
    return train,test,{'protocol_sha256':sha(p),'parent_manifest_sha256':protocol['parent_manifest_sha256'],
        'counts':{'train':1600,'query':800,'gallery':1600},'all_image_sha_verified':True,
        'relevance':'exact_SKU','closed_catalog':True}


class PoseImages(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        row=self.base[i]
        return {'image':cnn(row['image']),**{k:row[k] for k in
            ('heatmaps','keypoints','visible','torso_scale','head_scale','sample_id')}}


def pose_data():
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import load_settings
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import prepare_lsp,LSPPoseDataset
    config=ROOT/'experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_opt2.yaml'
    settings=load_settings(config)
    settings.data_root=ROOT/'data/lsp_pose';settings.download=False
    settings.train_limit=None;settings.test_limit=None
    receipt=json.loads((ROOT/'maintenance/storage/T02_CONTENT_ASSETS_20261004.json').read_text())['official']
    for relative,expected in receipt['annotations'].items():
        if sha(settings.data_root/relative)!=expected['sha256']:raise ValueError('LSP annotation identity differs')
    bundle=prepare_lsp(settings,persist=False)
    if (len(bundle.train),len(bundle.test))!=(10428,1000):raise ValueError('LSP original counts differ')
    train=PoseImages(LSPPoseDataset(bundle.train,settings,training=True))
    test=PoseImages(LSPPoseDataset(bundle.test,settings,training=False))
    return train,test,settings,{'config_sha256':sha(config),**bundle.metadata,
        'preprocessing':'original_pose_crop_and_augmentation_then_ImageNet_normalization'}


class SaliencyImages(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        r=self.base[i]
        return {'image':cnn(r['image']),**{k:r[k] for k in ('density','fixation','sample_id')}}


def saliency_data():
    from LightGenV2.tasks.t03_saliency.settings import load_settings
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import prepare_salicon,SALICONSaliencyDataset
    config=ROOT/'LightGenV2/tasks/t03_saliency/configs/qwen_aligned_head_50.yaml'
    settings=load_settings(config)
    settings.download=False;settings.materialize_density_maps=False
    bundle=prepare_salicon(settings,persist=False)
    if (len(bundle.train_records),len(bundle.validation_records))!=(10000,5000):raise ValueError('SALICON counts differ')
    annotations={split:sha(Path(meta['annotation_path'])) for split,meta in bundle.metadata['splits'].items()} if 'splits' in bundle.metadata else {}
    if annotations!={'train':'d4b5500cde45d556b16144726a31c8d8550adb8fb8892955ab6fa86d13ecfc37',
                     'validation':'c99c8c2850b65f640ff10ce09be0d5add59f151bcc9875e280a927af284a90c8'}:
        raise ValueError('SALICON original fixation annotations differ')
    identity={'config_sha256':sha(config),**bundle.metadata,'annotation_sha256':annotations,
              'maps_materialized':False,'Qwen_features_used':False}
    return (SaliencyImages(SALICONSaliencyDataset(bundle.train_records,settings,training=True)),
            SaliencyImages(SALICONSaliencyDataset(bundle.validation_records,settings,training=False)),settings,identity)


class OpenMojiImages(Dataset):
    def __init__(self,root,rows):self.root,self.rows=root,rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        source=torch.tensor(r['source_grid'],dtype=torch.long)
        target=torch.tensor(r['target_grid'],dtype=torch.long)
        with Image.open(safe_path(self.root,Path(r['relative_dir'])/'source.png')) as image:
            pixels=cnn(image)
        return {'image':pixels,'tokens':fixed_tokens(r['instruction']),'sample_id':r['sample_id'],
                'source_grid':source,'target_grid':target,'edit_grid':source.ne(target).float(),
                'task':r['task'],'instruction':r['instruction'],
                'preserve_grid':source.eq(target).float(),'task_index':r['task_index']}


def openmoji_data():
    root=Path('/DATA/DATA1/guest3/LightGenV2_worktrees/t04_layered_1bc120428/LightGenV2/tasks/t04_semantic_interaction/dataset/openmoji_layered_anchor6_svg_v3')
    summary=json.loads((root/'dataset_summary.json').read_text())
    if sha(root/'dataset_summary.json')!='1ed94474c4f15e4715f65c262f2cc7f42061ada395082ccd65ce6dd7bd5e7f18':
        raise ValueError('Original layered OpenMoji identity differs')
    if summary['type']!='openmoji_layered_anchor6_proportional_svg_v3' or summary['seed']!=73:
        raise ValueError('Wrong OpenMoji layout/source split')
    datasets={};seen=set()
    for split,count in [('train',5000),('test',1000)]:
        path=root/(split+'.jsonl')
        if sha(path)!=summary[split]['sha256']:raise ValueError('OpenMoji manifest SHA differs')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=count:raise ValueError('OpenMoji count differs')
        for row in rows:
            key=json.dumps([row['source_grid'],row['instruction']],sort_keys=True)
            if key in seen:raise ValueError('OpenMoji duplicate input across/within splits')
            seen.add(key);fixed_tokens(row['instruction'])
            safe_path(root,Path(row['relative_dir'])/'source.png')
        datasets[split]=OpenMojiImages(root,rows)
    return datasets['train'],datasets['test'],{'dataset_root':str(root),'summary_sha256':sha(root/'dataset_summary.json'),
        'layout':summary['type'],'train_sha256':summary['train']['sha256'],'test_sha256':summary['test']['sha256'],
        'counts':{'train':5000,'test':1000},'dataset_seed':73,'training_seed':42,'Qwen_features_used':False}


class RGBEditorPairs(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        r=self.base[i]
        return {'image':normalize(r['reference'].add(1).mul(.5)),
                'target':r['target'].add(1).mul(.5),'tokens':fixed_tokens(r['prompt']),
                'sample_id':r['sample_id'],'mode':r['mode']}


def editor_data():
    # Bind original renderer and curated TRAIN catalogue without loading even
    # an empty pretrained cache. Targets/masks are dropped before model input.
    from .editor_dataset import TextOnlyEditorDataset
    root=Path('/DATA/DATA1/guest3/t12_assets/datasets/abo_cleanrender_lamp_table_pillow_256_v1')
    expected={'train':'5e008814c3f8a83409c41744df056235a19d3433f0bd0755e33a454680477264',
              'val':'307b0e5039f5523b29d1121c9840cfe454477d83806dd3fbbfadb0e006abb5b7',
              'test':'331874abbb8d8c7a4ac5ee3d6e030f20611cb5a9e96d3092917b7a5c4d7d5d2f'}
    if any(sha(root/(split+'.jsonl'))!=expected[split] for split in expected):
        raise ValueError('Original T12 source manifests differ')
    raw={split:TextOnlyEditorDataset(root,split) for split in ('train','val','test')}
    if [len(raw[k]) for k in ('train','val','test')]!=[20736,2304,2304]:raise ValueError('T12 counts differ')
    sequence_sets={split:{r['sequence_id'] for r in d.sources} for split,d in raw.items()}
    if any(sequence_sets[a]&sequence_sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('T12 source product leakage')
    identity={'dataset_root':str(root),'source_manifests':{split:sha(root/(split+'.jsonl')) for split in raw},
              'counts':{'train':20736,'val':2304,'test':2304},'pretrained_feature_cache_used':False,
              'metric':'per_image_RGB_0_1_PSNR_then_mean; original_Gaussian_SSIM',
              'loss':'RGB_MSE_plus_0.05_gradient_L1; differs_from_Qwen_latent_loss'}
    return {split:RGBEditorPairs(d) for split,d in raw.items()},identity
