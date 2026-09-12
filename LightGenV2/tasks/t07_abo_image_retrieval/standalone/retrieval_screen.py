"""Small, predeclared instance-retrieval screens; never changes the ABO protocol.

The optical path imports only the existing compact student. Full Qwen is loaded
only by the separate qwen64 command, never by the optical command.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import torch
from torch.nn import functional as F

from .io import sha256, write_json, write_csv, source_commit

COIL_SHA256 = 'fcb4b8b01c0778f416d8d7a09f45082c4f7075cc74743dcddd5c44307bd4afe7'
COIL_URL = 'https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip'
OPTICS_SHA256 = '6490c6ee0ccc7501572fbae722aafbd7d4a016425d21af454019e7b60625433d'
GALLERY_ANGLES = (0, 90, 180, 270)
QUERY_ANGLES = (30, 60, 120, 150, 210, 240, 300, 330)


def coil_records():
    # Membership fixed without looking at images, labels, or model performance.
    order = sorted(range(1, 101), key=lambda n: hashlib.sha256(f'coil100-object42:{n}'.encode()).hexdigest())
    train = set(order[:60])
    rows = []
    for obj in range(1, 101):
        for angle in range(0, 360, 5):
            split = ('train' if obj in train else 'gallery' if angle in GALLERY_ANGLES
                     else 'query' if angle in QUERY_ANGLES else 'unused_test_view')
            rows.append(dict(sample_id=f'obj{obj}__{angle}', product_id=f'obj{obj}',
                             split=split, angle=angle, image_path=f'coil-100/obj{obj}__{angle}.png'))
    return rows


def validate_rows(rows, *, disjoint_products=True):
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate sample identities')
    groups = {split: [r for r in rows if r['split'] == split] for split in ('train', 'gallery', 'query')}
    if any(not group for group in groups.values()):
        raise ValueError('Train, query and gallery must all exist')
    train = {r['product_id'] for r in groups['train']}
    gallery = {r['product_id'] for r in groups['gallery']}
    query = {r['product_id'] for r in groups['query']}
    if (disjoint_products and train & (gallery | query)) or not query <= gallery:
        raise ValueError('Training/test product leakage or missing positives')
    if len({r['image_path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image path across protocol rows')
    return groups


def prepare_coil(archive, root, output):
    if sha256(archive) != COIL_SHA256:
        raise ValueError('COIL official archive identity differs')
    rows = coil_records()
    validate_rows(rows)
    root = root.resolve()
    if output.exists():
        raise FileExistsError(output)
    with ZipFile(archive) as z:
        names = {r['image_path'] for r in rows}
        if not names <= set(z.namelist()):
            raise ValueError('Archive lacks expected 100x72 images')
        for name in sorted(names):
            dest = root / name
            data = z.read(name)  # CRC verified by ZipFile; extract only fixed names.
            if dest.exists():
                if dest.read_bytes() != data:
                    raise ValueError(f'Existing image differs: {dest}')
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
    for row in rows:
        row['image_sha256'] = sha256(root / row['image_path'])
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', dict(schema=1, dataset='COIL-100',
        protocol='heldout40_objects_four_gallery_views_eight_queries_v1',
        source_url=COIL_URL, archive_sha256=COIL_SHA256,
        purpose='Controlled multi-view object instance retrieval, NOT ABO category retrieval or a real retail benchmark',
        split_rule='60/40 objects ordered by sha256(coil100-object42:<integer>)',
        gallery_angles=list(GALLERY_ANGLES), query_angles=list(QUERY_ANGLES),
        minimum_query_gallery_angle_degrees=30,
        train_images=4320, gallery_images=160, query_images=320, unused_test_images=2400,
        validation=False, selection='Predeclared before any model evaluation; no tuning in initial screen',
        relevance='same physical object identity; rank all individual gallery images; no centroid or class prefilter',
        license_note='Official research dataset; redistribution permission not established, do not bundle images',
        source_commit=source_commit(), rows=rows))


def load_screen(manifest, root):
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('schema') != 1:
        raise ValueError('Unknown screen schema')
    if data.get('protocol') not in ('heldout40_objects_four_gallery_views_eight_queries_v1',
                                    'grocery81_official_test_to_iconic_v1'):
        raise ValueError('Unknown predeclared retrieval protocol')
    groups = validate_rows(data['rows'], disjoint_products=data['protocol'] != 'grocery81_official_test_to_iconic_v1')
    # No target images are used for fitting. Check every declared image identity.
    root = root.resolve()
    for row in data['rows']:
        path = (root / row['image_path']).resolve()
        if not path.is_relative_to(root) or sha256(path) != row['image_sha256']:
            raise ValueError('Image path or SHA changed')
    return data, groups


def grocery_records(root):
    """Official images/splits, 81 iconic gallery images, fine-class relevance.

    Here product_id is a RELEVANCE LABEL, not a physical item identifier. The
    official split does not establish unseen physical-product generalization.
    """
    root = root.resolve()
    rows = []
    with (root / 'classes.csv').open(encoding='utf-8', newline='') as f:
        classes = list(csv.DictReader(f))
    labels = {int(r['Class ID (int)']) for r in classes}
    if labels != set(range(81)) or len(classes) != 81:
        raise ValueError('Require all 81 official fine classes')
    for c in classes:
        label = int(c['Class ID (int)'])
        rows.append(dict(sample_id=f'iconic:{label}', product_id=f'fine_class:{label}',
            split='gallery', image_path=c['Iconic Image Path (str)'].lstrip('/'),
            fine_class_id=label, coarse_class_id=int(c['Coarse Class ID (int)'])))
    for name, split in [('train', 'train'), ('test', 'query'), ('val', 'unused_validation')]:
        with (root / f'{name}.txt').open(encoding='utf-8', newline='') as f:
            for row in csv.reader(f, skipinitialspace=True):
                if len(row) != 3:
                    raise ValueError('Invalid official split row')
                path, fine, coarse = row
                fine, coarse = int(fine), int(coarse)
                if fine not in labels:
                    raise ValueError('Unknown fine class')
                rows.append(dict(sample_id=path, product_id=f'fine_class:{fine}', split=split,
                    image_path=path, fine_class_id=fine, coarse_class_id=coarse))
    groups = validate_rows(rows, disjoint_products=False)
    if len(groups['train']) != 2640 or len(groups['query']) != 2485:
        raise ValueError('Official train/test counts changed')
    for row in rows:
        path = (root / row['image_path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Dataset path escapes root')
        row['image_sha256'] = sha256(path)
    # No identical bytes may occur across training, query and gallery roles.
    role_hashes = {s: {r['image_sha256'] for r in g} for s, g in groups.items()}
    for a, b in [('train', 'query'), ('train', 'gallery'), ('query', 'gallery')]:
        if role_hashes[a] & role_hashes[b]:
            raise ValueError(f'Exact image duplication across {a}/{b}; audit, do not silently filter')
    return rows


def prepare_grocery(root, output):
    if output.exists():
        raise FileExistsError(output)
    rows = grocery_records(root)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', dict(schema=1, dataset='GroceryStoreDataset',
        protocol='grocery81_official_test_to_iconic_v1',
        source_url='https://github.com/marcusklasson/GroceryStoreDataset',
        source_files_sha256={n: sha256(root / n) for n in ('classes.csv', 'train.txt', 'test.txt', 'val.txt')},
        purpose='Natural grocery photo to fine-class iconic image retrieval; NOT unseen SKU or physical-object retrieval',
        train_images=2640, query_images=2485, gallery_images=81,
        unused_validation_images=sum(r['split'] == 'unused_validation' for r in rows),
        relevance='same official fine class (81); entire iconic gallery, no coarse-class filtering',
        training_test_classes_overlap=True, physical_product_disjointness='Not provided by official metadata',
        validation=False, selection='All official test images/classes retained, no initial-screen fitting',
        source_commit=source_commit(), rows=rows))


def rank_instances(vectors, rows):
    if vectors.shape != (len(rows), 64) or not torch.isfinite(vectors).all():
        raise ValueError('Expected finite 64D vectors in manifest row order')
    if bool((vectors.float().norm(dim=-1) < 1e-8).any()):
        raise ValueError('Zero descriptor')
    query = [i for i, r in enumerate(rows) if r['split'] == 'query']
    gallery = [i for i, r in enumerate(rows) if r['split'] == 'gallery']
    if not query or not gallery:
        raise ValueError('Missing query/gallery')
    z = F.normalize(vectors.float().cpu(), dim=-1)
    # Stable tie ordering, no label-dependent ranking or gallery filtering.
    ranking = (z[query] @ z[gallery].T).argsort(dim=1, descending=True, stable=True)
    relevant = torch.tensor([[rows[gallery[j]]['product_id'] == rows[i]['product_id']
                             for j in order.tolist()] for i, order in zip(query, ranking)])
    if not bool(relevant.any(1).all()):
        raise ValueError('Query with no positive gallery image')
    from .data import _ranking_metrics
    report = _ranking_metrics(relevant.numpy())
    report.update(query_count=len(query), candidate_count=len(gallery),
                  relevant_candidates_per_query=float(relevant.sum(1).float().mean()))
    predictions = [dict(sample_id=rows[i]['sample_id'], product_id=rows[i]['product_id'],
        top1_sample_id=rows[gallery[int(order[0])]]['sample_id'],
        top1_product_id=rows[gallery[int(order[0])]]['product_id'], hit_at_1=int(hit[0]))
        for i, order, hit in zip(query, ranking, relevant)]
    return report, predictions


def checkpoint_history(payload, manifest_sha):
    """Do not label a newly trained checkpoint as an untrained transfer screen."""
    trained_manifest = payload.get('manifest_sha256')
    same_protocol = trained_manifest == manifest_sha
    updated = same_protocol and int(payload.get('epoch', 0)) > 0 and payload.get('variant') != 'initial'
    selected = same_protocol and bool(payload.get('test_selected', False))
    return dict(fitted_on_this_dataset=updated, test_selected=selected,
        checkpoint_training_manifest_sha256=trained_manifest,
        checkpoint_epoch=payload.get('epoch'), checkpoint_variant=payload.get('variant'),
        checkpoint_origin=('Current-protocol adaptation; fixed-weight reevaluation, no new fitting in this command'
                           if updated else 'Transferred or initial-fallback checkpoint; no current-protocol weight updates established'),
        history_note='test_selected refers to current protocol; other-dataset training/selection is not ruled out')


@torch.inference_mode()
def evaluate(args):
    from PIL import Image, ImageOps
    from transformers import AutoProcessor
    from .io import inputs, picture, verify_assets, evaluation_checkpoint
    from .cli import autocast
    manifest_sha = sha256(args.manifest)
    protocol, groups = load_screen(args.manifest, args.data)
    rows = groups['gallery'] + groups['query']
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required but unavailable')
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.output.mkdir(parents=True)
    identity = dict(source_commit=source_commit(), command=sys.argv, pid=os.getpid(),
        model_kind=args.mode, protocol=protocol['protocol'], manifest_sha256=manifest_sha,
        dataset=protocol['dataset'], fitted_on_this_dataset=False, descriptor_dimension=64,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), python=sys.version,
        torch=torch.__version__, gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
        test_selected=False, inference_teacher_required=False)
    status = dict(identity, status='running')
    write_json(args.output / 'status.json', status)
    model = None
    try:
        if args.mode == 'optical':
            from .model import OpticalRetrieval
            verify_assets(args.assets)
            path, digest = evaluation_checkpoint(args.assets, args.checkpoint, args.expected_checkpoint_sha256)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if sha256(path) != digest:
                raise ValueError('Checkpoint changed while reading')
            if sha256(Path(__file__).with_name('optics.py')) != OPTICS_SHA256:
                raise ValueError('Physical source changed')
            model = OpticalRetrieval(payload['metadata'])
            model.load_state_dict(payload['state_dict'], strict=True)
            audit = model.audit()
            if audit['alpha_bounds'][0] <= .4 or audit['descriptor_dimension'] != 64:
                raise ValueError('Require original high-alpha 64D architecture')
            identity.update(checkpoint_sha256=digest, model_audit=audit, protected_optics_sha256=OPTICS_SHA256,
                            **checkpoint_history(payload, manifest_sha))
            processor = AutoProcessor.from_pretrained(str(args.assets / 'processor'), local_files_only=True)
        else:
            # Baseline only: the student command never imports/loads full Qwen.
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model),
                local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
            processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                min_pixels=50176, max_pixels=50176)
            identity.update(model_path=str(args.model.resolve()), frozen=True, trainable_parameters=0,
                model_config_sha256=sha256(args.model / 'config.json'),
                mrl='last valid token, first 64 dimensions, L2 normalization',
                preprocessing='EXIF RGB, native aspect, min=max pixels 50176')
        model.to(device).eval().requires_grad_(False)
        status.update(identity)
        write_json(args.output / 'status.json', status)
        write_json(args.output / 'execution.json', identity)
        results = {}
        for removed in ([False, True] if args.mode == 'optical' else [False]):
            if args.mode == 'optical':
                model.set_remove_optical(removed)
            vectors = []
            for start in range(0, len(rows), args.batch_size):
                images = []
                for row in rows[start:start + args.batch_size]:
                    path = args.data / row['image_path']
                    if args.mode == 'optical':
                        images.append(picture(path, model.metadata.get('input_preprocessing', 'contain_white')))
                    else:
                        with Image.open(path) as im:
                            images.append(ImageOps.exif_transpose(im).convert('RGB'))
                batch = inputs(processor, images, device)
                with autocast(device):
                    if args.mode == 'optical':
                        vector = model(batch)
                    else:
                        hidden = model.model(**batch, use_cache=False, return_dict=True).last_hidden_state
                        mask = batch['attention_mask'].bool()
                        position = torch.arange(mask.shape[1], device=device)[None].expand_as(mask).masked_fill(~mask, -1).amax(1)
                        vector = F.normalize(hidden[torch.arange(len(mask), device=device), position, :64].float(), dim=-1)
                vectors.append(vector.float().cpu())
                print(f'{args.mode} remove={removed}: {min(start+args.batch_size,len(rows))}/{len(rows)}', flush=True)
            values = torch.cat(vectors)
            metrics, predictions = rank_instances(values, rows)
            name = 'remove_optical' if removed else 'normal'
            results[name] = metrics
            torch.save(dict(manifest_sha256=manifest_sha, ids=[r['sample_id'] for r in rows], vectors=values), args.output / f'{name}_features.pt')
            write_csv(args.output / f'{name}_predictions.csv', predictions)
        if sha256(args.manifest) != manifest_sha:
            raise ValueError('Manifest changed during evaluation')
        report = dict(identity, status='complete', metrics=results,
                      elapsed_seconds=time.time()-args.started,
                      timing_and_power='Not benchmarked; elapsed includes loading and artifact I/O')
        if 'remove_optical' in results:
            report['optical_removal_drop_percentage_points'] = 100*(results['normal']['hit_at_1']-results['remove_optical']['hit_at_1'])
        write_json(args.output / 'final_report.json', report)
        status.update(status='complete')
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        status.update(status='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        status.update(identity)
        write_json(args.output / 'status.json', status)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare-coil', 'prepare-grocery', 'optical', 'qwen64'])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--archive', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--assets', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--expected-checkpoint-sha256')
    p.add_argument('--model', type=Path)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    args.started = time.time()
    if args.mode == 'prepare-grocery':
        prepare_grocery(args.data, args.output)
    elif args.mode == 'prepare-coil':
        if args.archive is None:
            p.error('prepare-coil requires --archive')
        prepare_coil(args.archive, args.data, args.output)
    else:
        if args.manifest is None or args.batch_size < 1:
            p.error('Evaluation requires manifest and positive batch size')
        if args.mode == 'optical' and (args.assets is None or args.checkpoint is None or not args.expected_checkpoint_sha256):
            p.error('Optical screen requires pinned assets/checkpoint/SHA')
        if args.mode == 'qwen64' and args.model is None:
            p.error('Frozen baseline requires --model')
        evaluate(args)


if __name__ == '__main__':
    main()
