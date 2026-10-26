"""Identity-checked reuse of a completed TEST evaluation; no model inference."""
import hashlib
import json


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def verified_cached_result(run, stem, identity):
    result=run/f'{stem}_result.json'
    if not result.exists():return None
    payload=json.loads(result.read_text())
    if not payload.get('locked_result_cache'):return None
    if payload.get('state')!='complete':raise ValueError('Cached TEST is incomplete')
    for key,value in identity.items():
        if payload.get(key)!=value:raise ValueError(f'Cached TEST identity mismatch: {key}')
    predictions=run/f'{stem}_predictions.npz'
    if not predictions.exists() or file_sha(predictions)!=payload.get('predictions_sha256'):
        raise ValueError('Cached TEST predictions missing or changed')
    return payload
