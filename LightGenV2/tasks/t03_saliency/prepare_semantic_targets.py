"""CPU-only image-level COCO labels for the already audited extra-image pool.

No saliency/fixation labels are fabricated and no inference model is changed.
COCO category IDs are sparse: consumers must use the stored ordered category map.
"""
import argparse
import hashlib
import json
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


def prepare(annotation_file, annotation_sha256, image_manifest, manifest_sha256,
            salicon_root, output):
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
    data = semantic_records(json.loads(annotation_bytes), pool)
    payload = {
        'schema_version': 1,
        'purpose': 'training_only_auxiliary_object_presence_not_saliency_ground_truth',
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
            'images_without_annotated_instances': payload['images_without_annotated_instances']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('annotation-file', 'annotation-sha256', 'image-manifest',
                 'manifest-sha256', 'salicon-root', 'output'):
        parser.add_argument('--'+name, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
