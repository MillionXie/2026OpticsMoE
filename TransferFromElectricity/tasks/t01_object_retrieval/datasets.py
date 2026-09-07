"""Official CIFAR-100 splits for the unchanged class-prototype retrieval graph."""
import hashlib
import json
import random
from pathlib import Path
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import GrocerySample, GroceryRetrievalBundle


def class_partition(targets, class_id, seed, gallery_count):
    indices = [i for i, y in enumerate(targets) if int(y) == class_id]
    random.Random(f'{seed}:cifar100:train:{class_id}').shuffle(indices)
    if len(indices) <= gallery_count:
        raise ValueError('Insufficient training examples')
    return indices[gallery_count:], indices[:gallery_count]


def prepare_cifar100(config, root, output, seed):
    from torchvision.datasets import CIFAR100
    dataset_root = (root / config['root']).resolve()
    train_source = CIFAR100(str(dataset_root), train=True, download=False)
    test_source = CIFAR100(str(dataset_root), train=False, download=False)
    classes = config.get('class_ids') or list(range(100))
    if len(set(classes)) != len(classes) or any(c < 0 or c >= 100 for c in classes):
        raise ValueError('Invalid CIFAR100 fine-class selection')
    names = tuple(train_source.classes[c] for c in classes)
    records = {'train': [], 'test': [], 'gallery': []}
    image_root = dataset_root / 'rgb_png'
    for local_id, original_id in enumerate(classes):
        train_indices, gallery_indices = class_partition(train_source.targets, original_id, seed, config['gallery_per_class'])
        test_indices = [i for i,y in enumerate(test_source.targets) if int(y) == original_id]
        for split, indices in [('train',train_indices),('gallery',gallery_indices),('test',test_indices)]:
            official = 'official_test' if split == 'test' else 'official_train'
            source = test_source if split == 'test' else train_source
            folder = image_root / official / str(original_id)
            folder.mkdir(parents=True, exist_ok=True)
            for index in indices:
                path = folder / f'{index:05d}.png'
                if not path.exists():
                    picture, label = source[index]
                    assert int(label) == original_id
                    # Reusable data artifact, not a second training source tree.
                    picture.convert('RGB').save(path)
                records[split].append(GrocerySample(f'cifar100:{official}:{index:05d}',path,
                    original_id,names[local_id],local_id,split,official,split=='gallery'))
    manifest = [s.manifest_record() for split in records.values() for s in split]
    digest = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    metadata = {'dataset':'CIFAR-100','fine_class_ids':classes,'class_names':names,'data_seed':seed,
        'gallery_per_class':config['gallery_per_class'],'manifest_sha256':digest,
        'policy':'Gallery and adaptation validation are held out from official train; all selected official test images are retained',
        'original_files_sha256':{name:hashlib.sha256((dataset_root/'cifar-100-python'/name).read_bytes()).hexdigest() for name in ('train','test','meta')}}
    (output/'dataset.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return GroceryRetrievalBundle(tuple(records['train']),tuple(records['test']),tuple(records['gallery']),names,digest,metadata)
