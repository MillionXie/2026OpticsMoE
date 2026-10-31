"""Use the historical TRAIN-only CNN as a training teacher, never LG inference."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from LightGenV2.demo_check.frozen_electronic.model import Electronic
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256
from .train import write


def training_view_codes(augmentation):
    if augmentation=='horizontal_flip':return (0,4)
    if augmentation=='D4':return tuple(range(8))
    raise ValueError('Teacher augmentation requires horizontal_flip or D4')


def transform_teacher_images(images,code):
    if code not in range(8):raise ValueError('D4 code must be 0..7')
    value=torch.rot90(images,code%4,(1,2))
    return value.flip(2) if code>=4 else value


def teacher_view_indices(codes,augmentation):
    choices=training_view_codes(augmentation)
    if not bool(torch.stack([codes==v for v in choices]).any(0).all()):raise ValueError('Uncached teacher view code')
    return codes if augmentation=='D4' else (codes==4).long()


def teacher_KL(student_logits, teacher_logits, temperature):
    if temperature <= 0 or student_logits.shape != teacher_logits.shape:
        raise ValueError('Matching class logits and positive KD temperature required')
    return temperature**2 * F.kl_div(F.log_softmax(student_logits/temperature, 1),
        F.softmax(teacher_logits.detach()/temperature, 1), reduction='batchmean')


@torch.no_grad()
def prepare_original_teacher(raw, cfg, out):
    task=Path(__file__).resolve().parent;info=cfg['teacher'];path=task/info['checkpoint']
    if cfg['dataset']!='eurosat_original_f' or sha256(path)!=info['checkpoint_sha256']:
        raise ValueError('Original F teacher identity differs')
    if cfg['data']['trainval']['data_sha256']!=info['original_TRAIN_data_sha256']:
        raise ValueError('Teacher must have used the same original TRAIN data')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['stage']!='electronic' or payload['epoch']!=19:
        raise ValueError('Expected sealed historical electronic E19')
    state=payload['model'];model=Electronic(state['mean'],state['std']).cuda()
    model.load_state_dict(state,strict=True);model.requires_grad_(False).eval()
    frozen={k:v.detach().clone() for k,v in model.state_dict().items()};values=[];report={}
    view_codes=training_view_codes(cfg['augmentation'])
    for split in ['train','val']:
        part=raw[split];logits=[]
        for code in (view_codes if split=='train' else [0]):
            batches=[]
            for start in range(0,len(part['labels']),128):
                images=torch.from_numpy(part['images'][start:start+128]).cuda()
                batches.append(model(transform_teacher_images(images,code)))
            logits.append(torch.cat(batches))
        prediction=logits[0].argmax(1).cpu().numpy()
        report[split]=dict(accuracy=float((prediction==part['labels']).mean()),
            domains={title:float((prediction[part['domains']==d]==part['labels'][part['domains']==d]).mean()) for d,title in [(0,'RGB'),(1,'SAR')]})
        if split=='train':
            values=torch.stack(logits,1)
            report[split]['accuracy_by_augmentation_code']={str(code):float((z.argmax(1).cpu().numpy()==part['labels']).mean()) for code,z in zip(view_codes,logits)}
    assert all(torch.equal(model.state_dict()[k],v) for k,v in frozen.items())
    np.savez_compressed(out/'TRAIN_teacher_logits.npz',logits=values.cpu().numpy(),
        ids=raw['train']['ids'],labels=raw['train']['labels'],domains=raw['train']['domains'])
    write(out/'teacher_manifest.json',dict(checkpoint_sha256=sha256(path),teacher_parameters=sum(p.numel() for p in model.parameters()),
        training_source=str(Path(__file__).resolve().parents[2]/'demo_check/frozen_electronic/model.py'),
        original_TRAIN_data_sha256=info['original_TRAIN_data_sha256'],teacher_used_TEST=False,
        current_cache_splits=['TRAIN unaugmented','TRAIN horizontal_flip'] if cfg['augmentation']=='horizontal_flip' else [f'TRAIN D4 code{c}' for c in view_codes],training_view_codes=list(view_codes),TEST_cached=False,
        teacher_parameters_bitwise_frozen=True,metrics=report,logits_sha256=sha256(out/'TRAIN_teacher_logits.npz')))
    del model,payload,state,frozen
    return values
