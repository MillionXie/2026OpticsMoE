"""CPU-only pinned query identities for the two original T08 directions.

This is adapter preparation, not a measured run. Never reuse the T07 enrolled
gallery, its64-D descriptor, white-contain transform or its prompt here.
"""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path

DIRECTIONS = ('image_to_text', 'text_to_image')
MODEL_CONFIG_SHA = '9172f55b0b9cce70b7f67b10c58a408ccf3ec15c587e6efd4d5f41631237fded'
MODEL_WEIGHTS_SHA = 'c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1'
DATA_SHA = {
    'titles.csv': 'c97e8da3c6c608a706f2cc0d284aeafb873fc6cc17d04a7c9f9f66b2135846c1',
    'train.csv': 'f742fc9e2314b7fb2858ad486505c943f7b7e37493e5d601d8dcbf7740b5707b',
    'test.csv': '4ca5834abf4bb4d614b41f05b5b2481b3d41720dd78c1c37b05923c97657dfb8',
    'manifest.csv': '880b4f34a411df1bfbd98adc91b1c19651480323ee43e62ae4b84c36ca06c543',
}
BASELINE_QUERY = {
    'image_to_text': 'Retrieve the product title that best describes this product image.',
    'text_to_image': 'Retrieve product images that match the following product description.',
}
DOCUMENT_INSTRUCTION = "Represent the user's input."
FULL_REFERENCE = {'image_to_text': .7358333333333333, 'text_to_image': .82}
TEXT_BASELINE_BANK_SHA = '8f9bc7d1e2276f59b720e6666b8f24d7207df75d4e7ff57c3867e0cc10a447fc'
TEXT_BASELINE_SOURCE = ('1e218991a894923906f1e70eebd41377083252dd',
    'LightGenV2/tasks/t08_abo_image_text_retrieval/text_to_image_baseline.py')


def validate_sha256(digest):
    if len(digest)!=64 or set(digest)-set('0123456789abcdef'):
        raise ValueError('SHA256 must be exactly64 lowercase hexadecimal characters')
    return digest


for _pin in (MODEL_CONFIG_SHA,MODEL_WEIGHTS_SHA,TEXT_BASELINE_BANK_SHA,*DATA_SHA.values()):
    validate_sha256(_pin)


def validate_rows(titles, train, test):
    if (len(titles), len(train), len(test)) != (100, 4800, 2400):
        raise ValueError('Original100 titles, TRAIN4800 and TEST2400 required')
    if [int(r['label']) for r in titles] != list(range(100)):
        raise ValueError('Title order differs')
    products = {r['product_id']: int(r['label']) for r in titles}
    if len(products) != 100 or len({r['title'] for r in titles}) != 100:
        raise ValueError('Non-unique title candidates')
    train_ids = {r['sample_id'] for r in train}
    test_ids = {r['sample_id'] for r in test}
    if len(train_ids) != 4800 or len(test_ids) != 2400 or train_ids & test_ids:
        raise ValueError('Duplicate or overlapping sample identities')
    counts = {pid: 0 for pid in products}
    for row in test:
        if products.get(row['product_id']) != int(row['label']):
            raise ValueError('Product/label mapping differs')
        counts[row['product_id']] += 1
    if set(counts.values()) != {24}:
        raise ValueError('Every title needs24 exact-SKU TEST positives')


def load_contract(data_root):
    root = Path(data_root).resolve()
    for name, expected in DATA_SHA.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Original CSV SHA differs: '+name)
    def read(name):
        with (root/name).open(encoding='utf8', newline='') as stream:
            return list(csv.DictReader(stream))
    titles, train, test = read('titles.csv'), read('train.csv'), read('test.csv')
    validate_rows(titles, train, test)
    for row in test:
        p = (root/row['image_path']).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError('Invalid original TEST image path')
    return titles, train, test


def timed_calls(direction, titles, test):
    if direction not in DIRECTIONS:
        raise ValueError('Unknown retrieval direction')
    if len(titles) != 100 or len(test) != 2400:
        raise ValueError('Wrong original workload')
    if direction == 'image_to_text':
        # Approved new sealed order, not the old class-balanced timing subset.
        return [dict(index=i, pass_index=0, unique_query_index=i, row=row)
                for i, row in enumerate(test[:200])]
    # Exactly two full traversals of100 titles, not200 distinct queries.
    return [dict(index=100*p+i, pass_index=p, unique_query_index=i, row=row)
            for p in (0, 1) for i, row in enumerate(titles)]


def cost_contract(direction):
    if direction not in DIRECTIONS:
        raise ValueError('Unknown retrieval direction')
    image = direction == 'image_to_text'
    return dict(direction=direction, calls=200, unique_queries=200 if image else 100,
                baseline_descriptor_dimensions=2048, student_descriptor_dimensions=64,
                baseline_query_instruction=BASELINE_QUERY[direction],
                baseline_document_instruction=DOCUMENT_INSTRUCTION,
                baseline_full_test_reference=FULL_REFERENCE[direction],
                online_optical_passes=6 if image else 3,
                offline_candidate_optical_passes=3 if image else 6,
                candidate_count=100 if image else 2400,
                exact_sku_positives_per_query=1 if image else 24,
                baseline_ranking_stable=not image,
                baseline_image_preprocessing='original EXIF/RGB and dynamic aspect processor min=max50176 pixels',
                student_image_preprocessing='original GroceryRetrievalDataset RGB, bicubic ImageOps.fit224 centered',
                no_explicit_inference_warmup=True, first_call_retained=True,
                all199_inter_call_gaps_included=True,
                gallery_build='separate offline stage, bound source/model/input/precision/SHA; not a dummy query warmup',
                fp16_saved_title_bank_is_not_proof_of_original_fp32_bank=True,
                state='original adapter identity preparation, not yet officially measured')
