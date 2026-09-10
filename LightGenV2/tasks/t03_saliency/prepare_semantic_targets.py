"""CPU-only image-level COCO labels for the already audited extra-image pool.

No saliency/fixation labels are fabricated and no inference model is changed.
COCO category IDs are sparse: consumers must use the stored ordered category map.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .prepare_unlabeled_pool import file_sha, ids_sha, image_index


def semantic_records(annotations, pool):
    rows = pool['images']
    selected = [r['image_id'] for r in rows]
    if (not selected or selected != sorted(set(selected))
            or len(selected) != pool['selected_count']
            or ids_sha(selected) != pool['selected_ids_sha256']):
        raise ValueError('Extra-image manifest identity mismatch')
    categories = sorted(annotations['categories'], key=lambda x: x['id'])
    category_ids = [c['id'] for c in categories]
    if (len(categories) != 80 or len(set(category_ids)) != 80
            or any(type(i) is not int or i <= 0 for i in category_ids)
            or any(not isinstance(c.get('name'), str) or not c['name'] for c in categories)):
        raise ValueError('Expected 80 unique named COCO instance categories')
    image_ids = [r['id'] for r in annotations['images']]
    if len(set(image_ids)) != len(image_ids) or not set(selected) <= set(image_ids):
        raise ValueError('Missing/duplicate annotation image identities')
    labels = {i: set() for i in selected}
    allowed = set(category_ids)
    known_images = set(image_ids)
    seen_annotations = set()
    for row in annotations['annotations']:
        aid, image_id, category_id = row['id'], row['image_id'], row['category_id']
        if aid in seen_annotations or image_id not in known_images or category_id not in allowed:
            raise ValueError('Invalid annotation identity/category reference')
        seen_annotations.add(aid)
        if image_id in labels:
            # Crowd instances still establish category presence. No area threshold.
            labels[image_id].add(category_id)
    records = [{'sample_id': f'unlabeled/coco2017/{i:012d}', 'image_id': i,
                'positive_category_ids': sorted(labels[i])} for i in selected]
    counts = {i: sum(i in value for value in labels.values()) for i in category_ids}
    return {
        'categories': [{'id': c['id'], 'name': c['name']} for c in categories],
        'records': records,
        'category_positive_image_counts': [counts[i] for i in category_ids],
        'images_without_annotated_instances': sum(not labels[i] for i in selected),
    }


def add_spatial_boxes(annotations, data):
    """Add weak object-localization labels; never create fixation/density maps."""
    images = {r['id']: r for r in annotations['images']}
    selected = {r['image_id']: r for r in data['records']}
    for image_id, row in selected.items():
        source = images[image_id]
        size = [source.get('width'), source.get('height')]
        if any(type(v) is not int or v <= 0 for v in size):
            raise ValueError('Invalid annotation image dimensions')
        row['source_size_wh'] = size
        row['boxes_xyxy_unit'] = []
    skipped = clipped = count = 0
    for ann in sorted(annotations['annotations'], key=lambda x: x['id']):
        row = selected.get(ann['image_id'])
        if row is None:
            continue
        box = ann.get('bbox')
        if (not isinstance(box, (list,tuple)) or len(box) != 4
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in box)):
            raise ValueError('Invalid/nonfinite COCO bbox')
        x,y,bw,bh = box
        w,h = row['source_size_wh']
        if bw <= 0 or bh <= 0:
            skipped += 1
            continue
        raw = [x,y,x+bw,y+bh]
        bounds = [max(0.,min(w,raw[0])),max(0.,min(h,raw[1])),
                  max(0.,min(w,raw[2])),max(0.,min(h,raw[3]))]
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            skipped += 1
            continue
        clipped += int(bounds != raw)
        row['boxes_xyxy_unit'].append({'annotation_id':ann['id'], 'category_id':ann['category_id'],
                                      'xyxy':[bounds[0]/w,bounds[1]/h,bounds[2]/w,bounds[3]/h]})
        count += 1
    data.update(spatial_box_count=count, spatial_boxes_clipped=clipped,
                spatial_boxes_skipped_nonpositive_or_outside=skipped,
                additional_human_box_supervision=True,
                spatial_box_policy='COCO xywh -> clipped continuous xyxy / source width,height; crowd included; no area threshold; boxes are NOT segmentations or fixation labels',
                image_resize_contract='direct anisotropic RGB BICUBIC resize to 224x224; no crop/EXIF transpose')
    return data


def prepare(annotation_file, annotation_sha256, image_manifest, manifest_sha256,
            salicon_root, output, include_boxes=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite semantic targets: {output}')
    raw_manifest = Path(image_manifest).read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != manifest_sha256:
        raise ValueError('Image manifest SHA mismatch')
    pool = json.loads(raw_manifest)
    if (pool.get('schema_version') != 1
            or pool.get('status') != 'audited_images_only_not_teacher_predictions'):
        raise ValueError('Unsupported audited extra-image pool')
    excluded = set()
    for split, folder in [('train', 'train'), ('test', 'val')]:
        index = image_index(Path(salicon_root)/'images'/folder)
        if (len(index) != pool[f'salicon_{split}_count']
                or ids_sha(index) != pool[f'salicon_{split}_ids_sha256']):
            raise ValueError('SALICON exclusion split drift')
        if excluded & set(index):
            raise ValueError('SALICON train/test overlap')
        excluded.update(index)
    if {r['image_id'] for r in pool['images']} & excluded:
        raise ValueError('Semantic supervision pool overlaps SALICON')
    # Hash exactly the bytes parsed, not a separate read of a mutable file.
    annotation_bytes = Path(annotation_file).read_bytes()
    if hashlib.sha256(annotation_bytes).hexdigest() != annotation_sha256:
        raise ValueError('COCO annotation SHA mismatch')
    annotations = json.loads(annotation_bytes)
    data = semantic_records(annotations, pool)
    if include_boxes:
        data = add_spatial_boxes(annotations, data)
        # Original image geometry must agree with annotations before producing
        # normalized coordinates. Training still checks exact image content SHA.
        from PIL import Image
        root = Path(pool['coco_root']).resolve()
        for item, record in zip(pool['images'], data['records']):
            name = item['image_file']
            image_file = (root/name).resolve()
            if Path(name).name != name or image_file.parent != root:
                raise ValueError('Image filename escapes audited root')
            with Image.open(image_file) as image:
                if list(image.size) != record['source_size_wh']:
                    raise ValueError(f'Annotation/image dimensions mismatch: {record["sample_id"]}')
    payload = {
        'schema_version': 1,
        'purpose': ('training_only_auxiliary_object_regions_not_saliency_ground_truth' if include_boxes
                    else 'training_only_auxiliary_object_presence_not_saliency_ground_truth'),
        'annotation_sha256': annotation_sha256,
        'image_manifest_sha256': manifest_sha256,
        'selected_ids_sha256': pool['selected_ids_sha256'],
        'sample_count': pool['selected_count'],
        'train_overlap': 0, 'test_overlap': 0,
        'category_order': 'ascending original COCO category ID, not ID minus one',
        'empty_labels': 'image exists in annotation images; no annotated object of these 80 categories',
        'crowd_instances_included': True,
        'no_saliency_or_fixation_targets_created': True,
        'additional_human_semantic_supervision': True,
        **data,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(',', ':'))
        handle.write('\n')
    return {'path': str(output.resolve()), 'sha256': file_sha(output),
            'bytes': output.stat().st_size, 'samples': payload['sample_count'],
            'categories': len(payload['categories']),
            'images_without_annotated_instances': payload['images_without_annotated_instances'],
            **({k:payload[k] for k in ('spatial_box_count','spatial_boxes_clipped',
                 'spatial_boxes_skipped_nonpositive_or_outside')} if include_boxes else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('annotation-file', 'annotation-sha256', 'image-manifest',
                 'manifest-sha256', 'salicon-root', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--include-boxes', action='store_true',
                        help='Prepare separately identified weak spatial labels; does not run training')
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
