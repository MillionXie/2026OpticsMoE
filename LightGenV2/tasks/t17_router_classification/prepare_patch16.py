"""Acquire the original CC BY 4.0 skin pathology release; no image sampling."""
import argparse,hashlib,json,time,zipfile
from pathlib import Path
import requests

SOURCE='https://heidata.uni-heidelberg.de'
PID='doi:10.11588/data/7QCR8S'

def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=True)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    status=a.out/'download_status.json'
    try:
        r=requests.get(SOURCE+'/api/datasets/:persistentId/',params={'persistentId':PID},timeout=60)
        r.raise_for_status();record=r.json();version=record['data']['latestVersion']
        terms=version.get('termsOfUse','')
        if 'creativecommons.org/licenses/by/4.0/' not in terms:raise ValueError('CC BY 4.0 data terms missing')
        write(a.out/'source_metadata.json',record)
        files={f['dataFile']['filename']:f for f in version['files']}
        info=files['data.zip'];meta=info['dataFile']
        if info.get('restricted'):raise ValueError('restricted dataset')
        target=a.out/'data.zip';part=a.out/'data.zip.part'
        if not target.exists():
            offset=part.stat().st_size if part.exists() else 0
            response=requests.get(SOURCE+f'/api/access/datafile/{meta["id"]}',stream=True,
                headers={'Range':f'bytes={offset}-'} if offset else {},timeout=(60,120))
            response.raise_for_status()
            if offset and response.status_code!=206:offset=0
            last=0.;start=time.monotonic()
            with part.open('ab' if offset else 'wb') as f:
                for block in response.iter_content(4<<20):
                    f.write(block);offset+=len(block)
                    if time.monotonic()-last>10:
                        write(status,dict(state='downloading',bytes=offset,total=meta['filesize'],seconds=time.monotonic()-start))
                        print(f'downloaded {offset}/{meta["filesize"]}',flush=True);last=time.monotonic()
            if offset!=meta['filesize']:raise ValueError('archive length mismatch')
            if digest(part,'md5')!=meta['checksum']['value']:raise ValueError('official archive MD5 mismatch')
            part.rename(target)
        if target.stat().st_size!=meta['filesize'] or digest(target,'md5')!=meta['checksum']['value']:
            raise ValueError('existing archive does not match original release')
        with zipfile.ZipFile(target) as z:
            small=[n for n in z.namelist() if n.lower().endswith(('.csv','.json','.md'))]
            for name in small:
                data=z.read(name)
                if len(data)<30_000_000:(a.out/Path(name).name).write_bytes(data)
            write(a.out/'archive_inventory.json',dict(entries=len(z.infolist()),metadata_files=small,
                first_entries=z.namelist()[:20]))
        write(a.out/'download_receipt.json',dict(source=SOURCE,identifier=PID,license='CC BY 4.0',
            official_md5=meta['checksum']['value'],sha256=digest(target),bytes=target.stat().st_size,
            source_file_id=meta['id'],source_version=f'{version["versionNumber"]}.{version["versionMinorNumber"]}'))
        write(status,dict(state='verified_download',bytes=target.stat().st_size));print('verified original archive',flush=True)
    except Exception as exc:
        write(status,dict(state='failed',error=repr(exc)));raise

if __name__=='__main__':main()
