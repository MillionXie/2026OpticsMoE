"""Read-only EuroSAT image-release verification; never extracts or renames data."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile


def verify(archive, export_report, split_path, image_manifest, expected_images=53784):
    from PIL import Image
    archive = Path(archive)
    report = json.loads(Path(export_report).read_text(encoding='utf-8'))
    if not report['passed'] or report['state'] != 'complete':
        raise ValueError('Export is not complete')
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if archive.stat().st_size != report['archive_bytes'] or digest.hexdigest() != report['archive_sha256']:
        raise ValueError('Archive byte identity mismatch')
    expected_split = json.loads(Path(split_path).read_text(encoding='utf-8'))['records']
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
            raise ValueError('Duplicate or unsafe archive member')
        records = json.loads(package.read('SPLIT.json'))['records']
        if records != expected_split:
            raise ValueError('Split differs from retained training split')
        raw_manifest = package.read('IMAGE_MANIFEST.json')
        if hashlib.sha256(raw_manifest).hexdigest() != report['image_manifest_sha256'] or raw_manifest != Path(image_manifest).read_bytes():
            raise ValueError('Image manifest byte identity mismatch')
        manifest = json.loads(raw_manifest)
        expected = {r['path'] for r in records}
        actual = {n for n in names if n.startswith('images/') and n.endswith('.png')}
        if len(records) != expected_images or len(expected) != expected_images or actual != expected or set(manifest) != expected:
            raise ValueError('Image membership/count mismatch')
        for record in records:
            name = record['path']
            raw = package.read(name)
            if hashlib.sha256(raw).hexdigest() != manifest[name]['sha256']:
                raise ValueError('Image file SHA mismatch: ' + name)
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != (56, 56) or image.mode != 'RGB' or hashlib.sha256(image.tobytes()).hexdigest() != manifest[name]['pixel_sha256']:
                    raise ValueError('Image pixel contract mismatch: ' + name)
        for name in names:
            if name not in actual:
                # Stream CRC checks instead of extracting ancillary members.
                with package.open(name) as stream:
                    while stream.read(8 * 1024 * 1024):
                        pass
    return {'passed': True, 'images': len(records), 'archive_sha256': digest.hexdigest(),
            'split_matches_training': True, 'read_only': True, 'extracted': False,
            'scope': 'Archive/file/pixel/split identity only; not model accuracy'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('archive', 'export-report', 'split', 'image-manifest'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.export_report, args.split, args.image_manifest), indent=2))


if __name__ == '__main__':
    main()
