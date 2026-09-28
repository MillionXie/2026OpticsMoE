"""Short TRAIN-only electronic-path continuation; the six-stage model stays sealed.

The student is a separate electronic-only copy. It never changes the physical
checkpoint, masks, SLM encoding or existing CCD files. Selection uses 400
TRAIN-holdout photos whose image identities are excluded from gradients.
"""
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoProcessor

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import OpticalRetrieval
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import inputs, picture
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.cli import autocast

PROJECT = Path('/DATA/DATA1/guest3/2026OpticsMoE')
RUNS = PROJECT/'LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation'
DATA = PROJECT/'data/abo_similarity10_data'
ASSETS = PROJECT/'.codex_tmp/t07_robust_assets_20260926'
PROTOCOL = RUNS/'abo200_enrolled_protocol_20260913/protocol.json'
HOLDOUT = RUNS/'sixhour_strong_20260928/execution.json'
WEIGHT = Path(os.environ['ABO_ELECTRONIC_START'])
OUT = Path(os.environ['ABO_ELECTRONIC_OUTPUT'])
EXPECTED_SHA = 'f413efa865d6b1b7252d2b3babd0cedfbb4415a362f0818feafa36f2843f292e'
STEPS = int(os.environ.get('ABO_ELECTRONIC_STEPS', '80'))
EPOCHS = int(os.environ.get('ABO_ELECTRONIC_EPOCHS', '6'))


def digest_optics(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if '.optics.' in name:
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def augment(image, rng):
    # Preserve the complete product on white; no category-destroying crop/flip.
    image = image.copy()
    if rng.random() < .65:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(.93, 1.07))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(.93, 1.07))
    if rng.random() < .35:
        dx, dy = rng.randint(-2, 2), rng.randint(-2, 2)
        canvas = Image.new('RGB', image.size, (255, 255, 255))
        canvas.paste(image, (dx, dy))
        image = canvas
    if rng.random() < .08:
        image = image.filter(ImageFilter.GaussianBlur(.35))
    return image


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    random.seed(20260928)
    assert torch.cuda.is_available()
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest() == EXPECTED_SHA
    protocol = json.loads(PROTOCOL.read_text())
    holdout = json.loads(HOLDOUT.read_text())['holdout_audit']
    fit_ids, val_ids = set(holdout['fitting_ids']), set(holdout['validation_ids'])
    train_rows = [r for r in protocol['rows'] if r['split'] == 'train']
    fit = [r for r in train_rows if r['sample_id'] in fit_ids]
    val = [r for r in train_rows if r['sample_id'] in val_ids]
    assert len(fit) == 1200 and len(val) == 400 and not fit_ids.intersection(val_ids)
    assert len({r['product_id'] for r in fit}) == len({r['product_id'] for r in val}) == 200
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    payload = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    model = OpticalRetrieval(payload['metadata'])
    model.load_state_dict(payload['state_dict'], strict=True)
    optics_sha = digest_optics(model)
    model.set_remove_optical(True)
    for name, param in model.named_parameters():
        allowed = (name.startswith(('vision.', 'language.')) and
                   ('.blocks.' in name or '.output_norm.' in name or '.output_adapter.' in name)) or name.startswith('readout.')
        param.requires_grad_(bool(allowed))
    model.cuda()
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': [p], 'lr': 5e-5 if name.startswith('readout.') else 3e-5,
         'weight_decay': .03 if p.ndim > 1 else 0.}
        for name, p in params])
    processor = AutoProcessor.from_pretrained(str(ASSETS/'processor'), local_files_only=True)
    images = {r['sample_id']: picture(DATA/r['image_path'],model.metadata.get('input_preprocessing','contain_white'))
              for r in fit+val}
    labels = {r['sample_id']: r['product_id'] for r in fit+val}
    products = {}
    for r in fit:
        products.setdefault(r['product_id'], []).append(r)
    product_ids = sorted(products)
    device = torch.device('cuda')

    def encode(rows, batch_size=16):
        model.eval()
        vectors = []
        with torch.no_grad():
            for start in range(0,len(rows),batch_size):
                batch_rows = rows[start:start+batch_size]
                batch = inputs(processor,[images[r['sample_id']] for r in batch_rows],device)
                with autocast(device):
                    vectors.append(model(batch).float().cpu())
        return F.normalize(torch.cat(vectors),dim=-1)

    teacher = encode(fit)
    teacher_by_id = {r['sample_id']: teacher[index].to(device) for index,r in enumerate(fit)}
    fit_labels = [r['product_id'] for r in fit]
    def validate(bank):
        probes = encode(val)
        nearest = (probes @ bank.T).argmax(1).tolist()
        return sum(fit_labels[index] == row['product_id'] for index,row in zip(nearest,val))/len(val)

    baseline = validate(teacher)
    best = baseline
    best_epoch = 0
    history = [{'epoch':0,'electronic_val_r1':baseline}]
    contract = {'status':'running','source_checkpoint_sha256':EXPECTED_SHA,'optics_sha256':optics_sha,
                'split':'1200 TRAIN fit / 400 TRAIN holdout; original 800 query excluded',
                'holdout_caveat':'Warm-start model had seen these TRAIN identities before this continuation',
                'trainable_parameters':sum(p.numel() for _,p in params),
                'steps_per_epoch':STEPS,'epochs':EPOCHS,'gpu':torch.cuda.get_device_name(),
                'electronic_only':True,'physical_checkpoint_unchanged':True,
                'image_augmentation':'whole-product brightness/contrast, <=2px translation, rare mild blur',
                'selection':'electronic-only R@1 against TRAIN fit bank; no query'}
    (OUT/'status.json').write_text(json.dumps(contract,indent=2))
    def save_checkpoint(path, epoch):
        assert digest_optics(model) == optics_sha
        state = {name: tensor.detach().cpu() for name,tensor in model.state_dict().items()}
        torch.save({'metadata':model.metadata,'state_dict':state,
                    'epoch':epoch,'electronic_val_r1':best,'source_checkpoint_sha256':EXPECTED_SHA,
                    'optics_sha256':optics_sha,'electronic_only':True},path)
    save_checkpoint(OUT/'best.pt',0)
    print(json.dumps(history[-1]),flush=True)
    started = time.time()
    try:
        for epoch in range(1,EPOCHS+1):
            bank = encode(fit).to(device)
            model.train()
            rng = random.Random(20260928+epoch)
            running = 0.
            for step in range(STEPS):
                classes = rng.sample(product_ids,8)
                rows = []
                for sku in classes:
                    rows.extend(rng.sample(products[sku],2))
                ids = [r['sample_id'] for r in rows]
                samples = [augment(images[sid],rng) if rng.random()<.7 else images[sid] for sid in ids]
                batch = inputs(processor,samples,device)
                with autocast(device):
                    z = model(batch).float()
                    logits = z @ bank.T / .1
                    excluded = torch.tensor([[sid==reference['sample_id'] for reference in fit] for sid in ids],device=device)
                    positives = torch.tensor([[labels[sid]==reference['product_id'] for reference in fit] for sid in ids],device=device)
                    positives &= ~excluded
                    logits = logits.masked_fill(excluded,-1e4)
                    assert positives.any(1).all()
                    retrieval = (torch.logsumexp(logits,dim=1) -
                                 torch.logsumexp(logits.masked_fill(~positives,-1e4),dim=1)).mean()
                    targets = torch.stack([teacher_by_id[sid] for sid in ids])
                    anchor = (1-F.cosine_similarity(z,targets,dim=-1)).mean()
                    pair = (1-F.cosine_similarity(z[::2],z[1::2],dim=-1)).mean()
                    loss = retrieval + .4*anchor + .08*pair
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for _,p in params],1.)
                optimizer.step()
                running += float(loss.detach())
            bank = encode(fit)
            score = validate(bank)
            row = {'epoch':epoch,'train_loss':running/STEPS,'electronic_val_r1':score,
                   'elapsed_seconds':time.time()-started}
            history.append(row)
            (OUT/'history.json').write_text(json.dumps(history,indent=2))
            if score > best:
                best = score
                best_epoch = epoch
                save_checkpoint(OUT/'best.pt',epoch)
            contract.update(epoch=epoch,best_epoch=best_epoch,best_val_r1=best)
            (OUT/'status.json').write_text(json.dumps(contract,indent=2))
            print(json.dumps(row),flush=True)
        save_checkpoint(OUT/'last.pt',EPOCHS)
        contract.update(status='complete',best_epoch=best_epoch,best_val_r1=best,
                        last_val_r1=history[-1]['electronic_val_r1'])
        (OUT/'status.json').write_text(json.dumps(contract,indent=2))
    except BaseException as exc:
        contract.update(status='failed',error=repr(exc))
        (OUT/'status.json').write_text(json.dumps(contract,indent=2))
        raise


if __name__ == '__main__':
    main()
