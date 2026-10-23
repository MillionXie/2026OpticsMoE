"""Lossless CPU cache for deterministic RGBEditorPairs, never learned features.

Only the renderer's uint8 RGB and complete byte tokens are stored. Loading
replays the original float32 arithmetic in its original order, rather than
replacing it with uint8/255. Every row must pass exact equality while building.
Blobs are deduplicated, memory mapped read-only and never hashed per getitem.
An incomplete cache resumes from its last durable journal checkpoint.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import time

import numpy as np
import PIL
import torch
from torch.utils.data import Dataset


TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
FORMAT = 't19_rgb_editor_uint8_pairs_v1'
ROW_KEYS = {'image', 'target', 'tokens', 'sample_id', 'mode'}
STORE_NAMES = ('reference', 'target', 'tokens')
# The v1 builder at this published commit wrote the same uint8/tokens format
# and float32 reconstruction contract. Only this module may differ when the
# compatible RAM/vectorized reader is used; renderer/data identities stay exact.
COMPATIBLE_BUILDERS = {
    '5521db609a053889eb7bd27b35dc9a85333bf23c91435cb4c8b12190cac7cae5': '487beceba',
}
DEFAULT_SOURCE_FILES = [TASK / name for name in ('performance_cache.py', 'other_data.py',
    'editor_dataset.py', 'models.py')] + [ROOT / 'LightGenV2/tasks/t12_text_to_image' / name for name in (
    'product_unified_edit_data_v2.py', 'product_unified_edit_data.py',
    'product_global_redesign_data.py', 'product_object_replace_data.py',
    'product_scene_replace_data.py', 'product_scene_data.py')]
_DATASET = None
_CACHED_DATASET = None


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _prefix_hasher(path, byte_count):
    digest = hashlib.sha256()
    if byte_count == 0:
        return digest
    with Path(path).open('rb') as stream:
        remaining = byte_count
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError('Incomplete cache lost committed blob/index bytes')
            digest.update(block)
            remaining -= len(block)
    return digest


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _source_identity(files):
    return {str(Path(path).resolve()): _file_sha(path) for path in files}


def _restore_rgb(raw):
    # This exactly matches _tensor(PIL RGB) followed by RGBEditorPairs scaling.
    return raw.float().div(127.5).sub(1).add(1).mul(.5)


def _restore_row(reference, target, tokens, record):
    image = _restore_rgb(reference)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    return {'image': (image - mean) / std, 'target': _restore_rgb(target),
            'tokens': tokens, 'sample_id': record['sample_id'], 'mode': record['mode']}


def _assert_equal_rows(expected, actual, index):
    if set(expected) != ROW_KEYS or set(actual) != ROW_KEYS:
        raise ValueError('RGBEditorPairs cache requires exactly its original five fields')
    for name in ROW_KEYS:
        left, right = expected[name], actual[name]
        equal = (left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
                 if torch.is_tensor(left) and torch.is_tensor(right) else left == right)
        if not equal:
            raise ValueError(f'Cache cannot exactly reproduce sample {index} field {name}')


def _encode_row(index, row):
    if set(row) != ROW_KEYS:
        raise ValueError('Only deterministic RGBEditorPairs rows can use this cache')
    image, target, tokens = row['image'], row['target'], row['tokens']
    if (image.dtype != torch.float32 or target.dtype != torch.float32
            or image.shape != target.shape or image.ndim != 3 or image.shape[0] != 3
            or tokens.dtype != torch.int64 or tokens.ndim != 1):
        raise ValueError('Unexpected editor tensor dtype/shape')
    if any(value.device.type != 'cpu' for value in (image, target, tokens)):
        raise ValueError('Cache construction must use CPU dataset tensors')
    if not torch.isfinite(image).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite editor pixels')
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    reference_raw = ((image * std + mean) * 255).round().clamp(0, 255).to(torch.uint8)
    target_raw = (target * 255).round().clamp(0, 255).to(torch.uint8)
    raw = {'reference': reference_raw, 'target': target_raw, 'tokens': tokens}
    _assert_equal_rows(row, _restore_row(reference_raw, target_raw, tokens, row), index)
    record = {'index': index, 'sample_id': row['sample_id'], 'mode': row['mode'],
              'original_tensor_sha256': {name: _tensor_sha(row[name]) for name in ('image', 'target', 'tokens')},
              'stored_tensor_sha256': {name: _tensor_sha(value) for name, value in raw.items()}}
    stores = {name: (value.numpy().dtype.str, list(value.shape), value.numpy().tobytes())
              for name, value in raw.items()}
    return record, stores


def _initialize_worker(dataset, cache_dir=None):
    global _DATASET, _CACHED_DATASET
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    _DATASET = dataset
    _CACHED_DATASET = CachedEditorPairs(cache_dir) if cache_dir is not None else None


def _process_index(index):
    row = _DATASET[index]
    if _CACHED_DATASET is not None:
        _assert_equal_rows(row, _CACHED_DATASET[index], index)
        return index
    return _encode_row(index, row)


def _process_rows(dataset, indices, workers, cache_dir=None):
    if workers < 0:
        raise ValueError('Worker count must be nonnegative')
    if workers == 0:
        cached = CachedEditorPairs(cache_dir) if cache_dir is not None else None
        for index in indices:
            row = dataset[index]
            if cached is not None:
                _assert_equal_rows(row, cached[index], index)
                yield index
            else:
                yield _encode_row(index, row)
        return
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
        initializer=_initialize_worker, initargs=(dataset, str(cache_dir) if cache_dir is not None else None))
    pending = deque()
    iterator = iter(indices)
    try:
        for _ in range(workers * 2):
            index = next(iterator, None)
            if index is None:
                break
            pending.append(executor.submit(_process_index, index))
        while pending:
            yield pending.popleft().result()
            index = next(iterator, None)
            if index is not None:
                pending.append(executor.submit(_process_index, index))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


@contextmanager
def _exclusive_builder(cache_dir):
    # An OS lock is released on crashes; the persistent marker is not a lease.
    stream = (cache_dir / '.builder.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            if stream.read(1) == b'':
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def build_editor_cache(dataset, cache_dir, source_identity, source_files=None, workers=4,
                       resume=False, checkpoint_interval=64):
    """Build/resume a private cache without CUDA or changing dataset/sample order.

    Source code/data identities must match on resume. Only the cache's own
    uncommitted trailing bytes are truncated; original source assets are read-only.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError('Cache construction cannot share an initialized CUDA process')
    if len(dataset) < 1 or checkpoint_interval < 1:
        raise ValueError('Cache requires a nonempty dataset and positive checkpoint interval')
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = {'format': FORMAT, 'dataset_length': len(dataset), 'source_identity': source_identity,
        'dataset_split': getattr(getattr(dataset, 'base', None), 'split', None),
        'source_code_sha256': _source_identity(DEFAULT_SOURCE_FILES if source_files is None else source_files),
        'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'pillow': PIL.__version__}}
    with _exclusive_builder(cache_dir):
        manifest_path = cache_dir / 'manifest.json'
        if manifest_path.exists():
            if not resume:
                raise FileExistsError('Existing cache requires explicit resume=True')
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if any(_canonical(manifest.get(key)) != _canonical(value) for key, value in expected.items()):
                raise ValueError('Cannot resume a different dataset/source/cache contract')
            if manifest['status'] == 'complete':
                verify_cache_integrity(cache_dir)
                return manifest
        else:
            unexpected = [path.name for path in cache_dir.iterdir() if path.name != '.builder.lock']
            if unexpected:
                raise ValueError(f'Refuse to overwrite unrecognized cache files: {unexpected}')
            manifest = dict(expected, status='building', completed_samples=0, index_bytes=0,
                index_prefix_sha256=hashlib.sha256().hexdigest(),
                stores={}, started_unix=time.time(), all_rows_exactly_verified=False)
            _write_json(manifest_path, manifest)
        completed = manifest['completed_samples']
        index_path = cache_dir / 'index.jsonl'
        index_hasher = _prefix_hasher(index_path, manifest['index_bytes'])
        if index_hasher.hexdigest() != manifest['index_prefix_sha256']:
            raise ValueError('Incomplete cache committed sample index SHA differs')
        if index_path.exists():
            with index_path.open('r+b') as stream:
                stream.truncate(manifest['index_bytes'])
        elif completed:
            raise ValueError('Incomplete cache lost its committed sample index')
        records = [json.loads(line) for line in index_path.read_text(encoding='utf-8').splitlines()] if index_path.exists() else []
        if len(records) != completed or any(record['index'] != i for i, record in enumerate(records)):
            raise ValueError('Incomplete cache sample journal identity differs')
        lookup = {name: {} for name in STORE_NAMES}
        for record in records:
            for name in STORE_NAMES:
                lookup[name][record['stored_tensor_sha256'][name]] = record['stores'][name]
        streams = {}
        store_hashers = {}
        index_stream = index_path.open('ab')
        try:
            for name in STORE_NAMES:
                path = cache_dir / (name + '.bin')
                if not path.exists() and name in manifest['stores'] and manifest['stores'][name]['count']:
                    raise ValueError('Incomplete cache lost a committed blob')
                streams[name] = path.open('r+b' if path.exists() else 'w+b')
                size = manifest['stores'].get(name, {}).get('bytes', 0)
                store_hashers[name] = _prefix_hasher(path, size)
                if store_hashers[name].hexdigest() != manifest['stores'].get(name, {}).get(
                        'prefix_sha256', hashlib.sha256().hexdigest()):
                    raise ValueError(f'Incomplete cache committed blob SHA differs: {name}')
                streams[name].truncate(size)
                streams[name].seek(size)

            def commit(status, count):
                for name, stream in streams.items():
                    stream.flush(); os.fsync(stream.fileno())
                    if name in manifest['stores']:
                        manifest['stores'][name]['bytes'] = stream.tell()
                        manifest['stores'][name]['prefix_sha256'] = store_hashers[name].hexdigest()
                index_stream.flush(); os.fsync(index_stream.fileno())
                manifest.update(status=status, completed_samples=count, index_bytes=index_stream.tell(),
                                index_prefix_sha256=index_hasher.hexdigest(),
                                last_checkpoint_unix=time.time(), all_rows_exactly_verified=count == len(dataset))
                manifest.pop('error', None)
                _write_json(manifest_path, manifest)

            count = completed
            for record, raw in _process_rows(dataset, range(completed, len(dataset)), workers):
                record['stores'] = {}
                for name, (dtype, shape, payload) in raw.items():
                    spec = manifest['stores'].setdefault(name, {'dtype': dtype, 'shape': shape, 'count': 0, 'bytes': 0})
                    if (spec['dtype'], spec['shape']) != (dtype, shape):
                        raise ValueError('Editor cached tensor shapes changed within split')
                    digest = record['stored_tensor_sha256'][name]
                    if digest not in lookup[name]:
                        lookup[name][digest] = spec['count']
                        streams[name].write(payload)
                        store_hashers[name].update(payload)
                        spec['count'] += 1
                    record['stores'][name] = lookup[name][digest]
                serialized = (_canonical(record) + '\n').encode('utf-8')
                index_stream.write(serialized)
                index_hasher.update(serialized)
                count += 1
                if count % checkpoint_interval == 0:
                    commit('building', count)
            commit('building', count)
            for name in STORE_NAMES:
                manifest['stores'][name]['sha256'] = _file_sha(cache_dir / (name + '.bin'))
            manifest['index_sha256'] = _file_sha(index_path)
            manifest.update(status='complete', finished_unix=time.time())
            _write_json(manifest_path, manifest)
            return manifest
        except BaseException as error:
            # Reload the last durable checkpoint, not mutable uncommitted counts.
            durable = json.loads(manifest_path.read_text(encoding='utf-8'))
            durable.update(status='interrupted', error=repr(error), interrupted_unix=time.time())
            _write_json(manifest_path, durable)
            raise
        finally:
            index_stream.close()
            for stream in streams.values():
                stream.close()


def verify_cache_integrity(cache_dir):
    """Once-per-cache file verification; deliberately excluded from getitem."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['format'] != FORMAT or manifest['status'] != 'complete':
        raise ValueError('Cache is not complete')
    if _file_sha(cache_dir / 'index.jsonl') != manifest['index_sha256']:
        raise ValueError('Cache sample index SHA differs')
    for name, spec in manifest['stores'].items():
        path = cache_dir / (name + '.bin')
        if path.stat().st_size != spec['bytes'] or _file_sha(path) != spec['sha256']:
            raise ValueError(f'Cache blob identity differs: {name}')
    return manifest


class CachedEditorPairs(Dataset):
    """Drop-in Dataset, with optional verified CPU RAM preload/batched reads.

    ``preload()`` sequentially reads and hashes the actual immutable store
    bytes once. ``get_batch(indices)`` then performs no file access or hashing.
    The source manifest and cache blobs are never edited by this reader.
    """

    def __init__(self, cache_dir, source_identity=None, verify_integrity=False):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / 'manifest.json').read_text(encoding='utf-8'))
        manifest = self.manifest
        if (manifest.get('format') != FORMAT or manifest.get('status') != 'complete'
                or not manifest.get('all_rows_exactly_verified')):
            raise ValueError('Only a complete, exactly verified editor cache can be read')
        if source_identity is not None and _canonical(source_identity) != _canonical(manifest['source_identity']):
            raise ValueError('Editor cache source data identity differs')
        reader_path = Path(__file__).resolve()
        reader_sha = _file_sha(reader_path)
        accepted_builder = None
        compatibility_commit = None
        for path, expected in manifest['source_code_sha256'].items():
            actual = reader_sha if Path(path).resolve() == reader_path else _file_sha(path)
            if actual == expected:
                if Path(path).resolve() == reader_path:
                    accepted_builder = expected
                continue
            if Path(path).resolve() == reader_path and expected in COMPATIBLE_BUILDERS:
                accepted_builder = expected
                compatibility_commit = COMPATIBLE_BUILDERS[expected]
            else:
                raise ValueError('Editor cache renderer/source code identity differs')
        if verify_integrity:
            verify_cache_integrity(self.cache_dir)
        index_bytes = (self.cache_dir / 'index.jsonl').read_bytes()
        self._index_sha256_at_load = hashlib.sha256(index_bytes).hexdigest()
        self.records = [json.loads(line) for line in index_bytes.decode('utf-8').splitlines()]
        if len(self.records) != manifest['dataset_length'] or any(row['index'] != i for i, row in enumerate(self.records)):
            raise ValueError('Editor cache sample index identity differs')
        for name in STORE_NAMES:
            spec = manifest['stores'][name]
            expected_bytes = int(np.prod(spec['shape'])) * np.dtype(spec['dtype']).itemsize * spec['count']
            if spec['bytes'] != expected_bytes or (self.cache_dir / (name + '.bin')).stat().st_size != expected_bytes:
                raise ValueError('Editor cache blob shape/size differs')
        self._maps = None
        self._ram = None
        self._row_lookup = torch.tensor([[record['stores'][name] for name in STORE_NAMES]
                                        for record in self.records], dtype=torch.int64)
        self.reader_provenance = {
            'reader_code_sha256': reader_sha,
            'accepted_builder_sha256': accepted_builder,
            'compatible_builder_commit': compatibility_commit,
            'cache_format': FORMAT,
            'cache_manifest_sha256': _file_sha(self.cache_dir / 'manifest.json'),
            'cache_index_sha256': manifest['index_sha256'],
            'cache_identity_preserved': True,
            'ram_preloaded': False,
        }

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_maps'] = None
        state['_ram'] = None
        state['reader_provenance'] = dict(self.reader_provenance, ram_preloaded=False)
        return state

    def __getitem__(self, index):
        record = self.records[index]
        if self._ram is not None:
            values = {name: self._ram[name][record['stores'][name]].clone() for name in STORE_NAMES}
            return _restore_row(values['reference'], values['target'], values['tokens'], record)
        if self._maps is None:
            self._maps = {name: np.memmap(self.cache_dir / (name + '.bin'), mode='r',
                dtype=spec['dtype'], shape=(spec['count'], *spec['shape']))
                for name, spec in self.manifest['stores'].items()}
        values = {name: torch.from_numpy(self._maps[name][record['stores'][name]].copy())
                  for name in STORE_NAMES}
        return _restore_row(values['reference'], values['target'], values['tokens'], record)

    def preload(self):
        """Read immutable uint8/token stores sequentially into verified CPU RAM.

        Hash the loaded arrays themselves, so this is sufficient integrity
        verification even when the constructor used verify_integrity=False.
        No GPU is used and no cache file or manifest is changed.
        """
        if self._ram is not None:
            return self
        if self._index_sha256_at_load != self.manifest['index_sha256']:
            raise ValueError('Cache sample index SHA differs')
        arrays = {}
        for name in STORE_NAMES:
            spec = self.manifest['stores'][name]
            path = self.cache_dir / (name + '.bin')
            if path.stat().st_size != spec['bytes']:
                raise ValueError(f'Cache blob identity differs: {name}')
            count = int(np.prod(spec['shape'])) * spec['count']
            values = np.fromfile(path, dtype=spec['dtype'], count=count)
            if values.size != count:
                raise ValueError(f'Cache blob read was incomplete: {name}')
            if hashlib.sha256(memoryview(values).cast('B')).hexdigest() != spec['sha256']:
                raise ValueError(f'Cache blob identity differs: {name}')
            arrays[name] = torch.from_numpy(values.reshape(spec['count'], *spec['shape']))
        self._ram = arrays
        self._maps = None
        self.reader_provenance.update(ram_preloaded=True,
            ram_store_bytes=sum(spec['bytes'] for spec in self.manifest['stores'].values()),
            ram_store_sha256_verified={name: self.manifest['stores'][name]['sha256'] for name in STORE_NAMES},
            batched_reconstruction='original_float32_operation_order; exact_equal_to_per_row_collation')
        return self

    def get_batch(self, indices):
        """Gather one full/partial/repeated-index batch from RAM without I/O."""
        if self._ram is None:
            raise RuntimeError('Call preload() before get_batch()')
        indices = torch.as_tensor(indices)
        if indices.device.type != 'cpu' or indices.ndim != 1 or indices.numel() == 0:
            raise ValueError('Batch indices must be a nonempty one-dimensional CPU integer array')
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError('Batch indices must be integers')
        indices = indices.to(torch.int64)
        if int(indices.min()) < 0 or int(indices.max()) >= len(self):
            raise IndexError('Cached editor batch index is out of range')
        lookup = self._row_lookup.index_select(0, indices)
        values = {name: self._ram[name].index_select(0, lookup[:, column])
                  for column, name in enumerate(STORE_NAMES)}
        image = _restore_rgb(values['reference'])
        mean = torch.tensor([.485, .456, .406])[None, :, None, None]
        std = torch.tensor([.229, .224, .225])[None, :, None, None]
        records = [self.records[index] for index in indices.tolist()]
        return {'image': (image - mean) / std, 'target': _restore_rgb(values['target']),
                'tokens': values['tokens'], 'sample_id': [row['sample_id'] for row in records],
                'mode': [row['mode'] for row in records]}


def verify_editor_cache(cache_dir, dataset, mode='full', sample_count=64, workers=4, seed=42):
    """Compare actual cached rows to fresh original rendering with torch.equal."""
    manifest = verify_cache_integrity(cache_dir)
    if len(dataset) != manifest['dataset_length']:
        raise ValueError('Verification dataset length differs')
    if mode == 'full':
        indices = list(range(len(dataset)))
    elif mode == 'random':
        if sample_count < 1:
            raise ValueError('sample_count must be positive')
        indices = sorted(np.random.default_rng(seed).choice(len(dataset), min(sample_count, len(dataset)), replace=False).tolist())
    else:
        raise ValueError('Verification mode must be full or random')
    checked = sum(1 for _ in _process_rows(dataset, indices, workers, cache_dir=cache_dir))
    receipt = {'mode': mode, 'checked_samples': checked, 'dataset_length': len(dataset),
        'all_checked_rows_exactly_equal': True, 'indices': indices if mode == 'random' else 'all',
        'cache_index_sha256': manifest['index_sha256'], 'verified_unix': time.time()}
    _write_json(Path(cache_dir) / ('verification_' + mode + '.json'), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/DATA/DATA1/guest3/t12_assets/datasets/abo_cleanrender_lamp_table_pillow_256_v1'))
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify', choices=['full', 'random', 'none'], default='full')
    args = parser.parse_args()
    if not args.cache_root.resolve().is_relative_to((TASK / 'runs').resolve()):
        raise ValueError('Cache must remain in T19 runs')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    from .other_data import editor_data
    datasets, identity = editor_data()
    if args.data_root.resolve() != Path(identity['dataset_root']).resolve():
        raise ValueError('CLI cache requires the original verified T19 editor dataset root')
    for split in args.splits:
        source = datasets[split]
        cache_dir = args.cache_root / split
        manifest = build_editor_cache(source, cache_dir, identity, workers=args.workers, resume=args.resume)
        print(json.dumps({'split': split, 'status': manifest['status'], 'samples': manifest['completed_samples'],
                          'stores': manifest['stores']}, ensure_ascii=False), flush=True)
        if args.verify != 'none':
            print(json.dumps(verify_editor_cache(cache_dir, source, mode=args.verify, workers=args.workers)), flush=True)


if __name__ == '__main__':
    main()
