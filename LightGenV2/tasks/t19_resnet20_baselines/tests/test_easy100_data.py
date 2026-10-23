import copy

from PIL import Image
import pytest
import torch

from LightGenV2.tasks.t19_resnet20_baselines.easy100_data import (
    Easy100, Easy100Images, validate_rows,
)


def fixture_rows():
    # Reversed title order detects accidental use of list position as label/SKU.
    titles = [{'label': '1', 'product_id': 'b', 'title': 'blue mug'},
              {'label': '0', 'product_id': 'a', 'title': 'red bag'}]
    train, test = [], []
    for split, output in [('train', train), ('test', test)]:
        for label, sku in enumerate(['a', 'b']):
            output.append({'sample_id': f'{split}-{sku}', 'split': split,
                           'label': str(label), 'product_id': sku,
                           'image_path': f'{split}-{sku}.png'})
    return train + test, train, test, titles


def check(rows):
    validate_rows(*rows, products=2, train_per_product=1, test_per_product=1)


def test_title_order_is_preserved_and_split_overlap_rejected():
    rows = fixture_rows()
    check(rows)
    rows[2][0]['sample_id'] = rows[1][0]['sample_id']
    with pytest.raises(ValueError):
        check(rows)


@pytest.mark.parametrize('fault', ['sku_mapping', 'missing_product', 'long_title'])
def test_rejects_semantically_wrong_metadata(fault):
    rows = copy.deepcopy(fixture_rows())
    if fault == 'sku_mapping':
        # Change manifest and split together: consistency alone must not pass.
        rows[1][0]['label'] = '1'
    elif fault == 'missing_product':
        rows[2].pop()
        rows[0].pop(2)
    else:
        rows[3][0]['title'] = 'x' * 513
    with pytest.raises(ValueError):
        check(rows)


def test_pixels_metadata_and_path_boundary(tmp_path):
    Image.new('RGB', (320, 160), (255, 0, 0)).save(tmp_path / 'red.png')
    row = {'image_path': 'red.png', 'product_id': 'a', 'sample_id': 'original-id'}
    sample = Easy100Images(tmp_path, [row])[0]
    assert sample['image'].shape == (3, 224, 224)
    assert sample['image'].dtype == torch.float32
    assert torch.isfinite(sample['image']).all()
    assert sample['image'][0, 0, 0].item() == pytest.approx((1 - .485) / .229)
    assert sample['sku'] == 'a' and sample['sample_id'] == 'original-id'
    for path in ['../outside.png', 'C:/outside.png', '/outside.png', 'dir\\outside.png']:
        with pytest.raises(ValueError):
            Easy100Images(tmp_path, [dict(row, image_path=path)])[0]


def test_official_loader_rejects_unpinned_csv_before_decoding(tmp_path):
    (tmp_path / 'manifest.csv').write_text('sample_id\nforged\n', encoding='utf-8')
    with pytest.raises(ValueError, match='CSV content identity'):
        Easy100(tmp_path)
