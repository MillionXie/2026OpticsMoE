"""One bounded, SHA-verified GLM download recovery; preserve failed partial bytes.

Uses small exact HTTP ranges instead of one multi-GB connection. At most three
attempts for transient network errors only; never retry an identity/range error.
Only pinned public model data, never executable patches or repository source.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import smolvlm2_temporal_baseline as shared

PROFILE = shared.TASK / 'configs/baselines/glm46v_flash_temporal_4f.json'
CHUNK = 256 * 1024 * 1024


def validate_range(response, start, end, total):
    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
        raise RuntimeError(f"Server did not honor exact range: status={response.status_code}, "
                           f"Content-Range={response.headers.get('Content-Range')!r}; preserve partial")
    if int(response.headers.get('Content-Length', -1)) != end - start + 1:
        raise RuntimeError('Unexpected HTTP range content length')


def range_request(requests, url, start, end, total, deadline, output):
    for attempt in range(1, 4):
        if time.time() >= deadline:
            raise TimeoutError('Original asset deadline reached')
        response = None
        try:
            response = requests.get(url, params={'range_start': start, 'range_end': end},
                headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                stream=True, timeout=(20, 60))
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f'Transient HTTP {response.status_code}')
            validate_range(response, start, end, total)
            return response
        except requests.RequestException as e:
            if response is not None:
                response.close()
            shared.frozen.write_json(output / f'network_{start}_{attempt}.json',
                dict(start=start, end=end, attempt=attempt, error=repr(e), at_unix=time.time(),
                     retry_after=response.headers.get('Retry-After') if response is not None else None))
            if attempt == 3:
                raise
            retry_after = response.headers.get('Retry-After') if response is not None else None
            from email.utils import parsedate_to_datetime
            try:
                delay = max(0, float(retry_after)) if retry_after is not None else 30 * attempt
            except ValueError:
                delay = max(0, parsedate_to_datetime(retry_after).timestamp() - time.time())
            if time.time() + delay >= deadline:
                raise TimeoutError('Server retry delay exceeds authorized deadline')
            # Respect Retry-After, reduce request frequency; never change endpoint/IP.
            until = time.time() + delay
            while time.time() < until:
                time.sleep(min(60, until - time.time()))
        except BaseException:
            if response is not None:
                response.close()
            raise


def main():
    import requests
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--resume-partial', choices=('recovery02', 'recovery03'), help='Preserve verified failed recovery bytes')
    args = p.parse_args()
    if not time.time() < args.deadline_unix <= time.time() + 3600:
        raise ValueError('Recovery must retain original asset deadline, at most one hour')
    c = json.loads(PROFILE.read_text())
    if c['model_id'] != 'zai-org/GLM-4.6V-Flash' or c['model_revision'] != '411bb4d77144a3f03accbf4b780f5acb8b7cde4e':
        raise ValueError('Only explicitly approved official GLM revision')
    model, output = args.model.resolve(), args.output.resolve()
    if model != Path('/DATA/DATA1/guest3/models/glm46v-flash-411bb4d7'):
        raise ValueError('Only canonical GLM official model directory')
    shared.new_output(output)
    output.mkdir(parents=True)
    identity = dict(source_sha256=shared.frozen.sha256(Path(__file__)),
        profile_sha256=shared.frozen.sha256(PROFILE), model_id=c['model_id'], revision=c['model_revision'],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=shared.REPO, text=True).strip(),
        command=sys.argv, deadline_unix=args.deadline_unix, pid=os.getpid(), gpu_initialized=False)
    shared.frozen.write_json(output / 'run_identity.json', identity)
    preserved = []
    try:
        # Small metadata must already be verified from the completed first stage.
        for name, spec in c['official_assets'].items():
            if not name.endswith('.safetensors') and not shared.asset_valid(model / name, spec):
                raise ValueError('Existing official metadata mismatch: ' + name)
        for name, spec in c['official_assets'].items():
            if not name.endswith('.safetensors'):
                continue
            target = model / name
            if target.exists():
                if not shared.asset_valid(target, spec):
                    raise ValueError('Preserve conflicting existing target: ' + name)
                continue
            prior = model / (name + '.' + args.resume_partial + '.partial') if args.resume_partial else None
            failed = prior if prior is not None and prior.exists() else model / (name + '.partial')
            suffix = '.recovery04.partial' if args.resume_partial == 'recovery03' else '.recovery03.partial' if args.resume_partial else '.recovery02.partial'
            working = model / (name + suffix)
            if working.exists():
                raise FileExistsError('No blind recovery restart: ' + str(working))
            if failed.exists():
                preserved.append(dict(path=str(failed), bytes=failed.stat().st_size,
                    sha256=shared.frozen.sha256(failed)))
                shutil.copyfile(failed, working)
                if shared.frozen.sha256(working) != preserved[-1]['sha256']:
                    raise RuntimeError('Preserved partial copy verification failed')
            else:
                with working.open('xb'):
                    pass
            start, total = working.stat().st_size, spec['size']
            if start > total:
                raise ValueError('Partial exceeds pinned official file size')
            shared.frozen.write_json(output / 'preserved_partial_manifest.json', preserved)
            while start < total:
                if time.time() >= args.deadline_unix:
                    raise TimeoutError('Original asset deadline reached, no extension')
                if shared.frozen.sha256(PROFILE) != identity['profile_sha256']:
                    raise RuntimeError('Pinned profile changed')
                end = min(start + CHUNK, total) - 1
                url = f"https://hf-mirror.com/{c['model_id']}/resolve/{c['model_revision']}/{name}"
                # Cache keys must distinguish ranges on public mirrors/CDNs.
                with range_request(requests, url, start, end, total, args.deadline_unix, output) as response:
                    count = 0
                    with working.open('ab') as f:
                        for block in response.iter_content(4 * 1024 * 1024):
                            if time.time() >= args.deadline_unix:
                                raise TimeoutError('Original asset deadline reached')
                            if block:
                                if count + len(block) > end - start + 1:
                                    raise RuntimeError('Oversized response; preserve evidence')
                                f.write(block)
                                count += len(block)
                    if count != end - start + 1:
                        raise RuntimeError('Truncated range; no automatic retry')
                start += count
                shared.frozen.write_json(output / 'status.json', dict(status='downloading', file=name,
                    bytes=start, total=total, updated_at_unix=time.time(), gpu_initialized=False))
                print(f'[verified-range] {name} {start}/{total}', flush=True)
            if not shared.asset_valid(working, spec):
                raise RuntimeError('Final official LFS SHA mismatch; preserve partial')
            if target.exists():
                raise FileExistsError('Target appeared during recovery; no overwrite')
            working.rename(target)
        hashes = shared.model_files(model, c)
        shared.frozen.write_json(output / 'model_asset_manifest.json', dict(identity=identity,
            sha256=hashes, preserved_failed_partials=preserved, verified_official_git_blobs_and_lfs_sha=True))
        shared.frozen.write_json(output / 'status.json', dict(status='complete', updated_at_unix=time.time(), gpu_initialized=False))
    except BaseException as e:
        shared.frozen.write_json(output / 'status.json', dict(status='failed', error=repr(e),
            updated_at_unix=time.time(), gpu_initialized=False))
        raise


if __name__ == '__main__':
    main()
