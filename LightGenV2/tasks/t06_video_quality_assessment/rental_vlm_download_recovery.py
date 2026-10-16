"""One bounded, exact-range continuation of the two public rental asset downloads.

Original partial bytes are preserved as an immutable recorded prefix, then only
appended. Only new transport chunk files created here are removed after fsync.
The complete official LFS SHA must pass before a shard is accepted.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import time
import requests
from .rental_vlm_assets import MODELS,check,sha,write

CHUNK=32*1024*1024
WORKERS=12
DEADLINE=1792181100.0 #03:45 +0800; original user release remains05:40.

def range_plan(offset,total):
    if not 0<=offset<=total:raise ValueError("Invalid preserved prefix")
    return [(s,min(s+CHUNK,total)-1) for s in range(offset,min(offset+WORKERS*CHUNK,total),CHUNK)]

def run(key,attempt=1):
    root=Path("/data/models")/key
    recovery=root/f"range_recovery{attempt:02d}";recovery.mkdir(exist_ok=False)
    priors=[root/f"range_recovery{i:02d}" for i in range(1,attempt)]
    metadata=json.loads((root/"official_metadata.json").read_text())
    if (metadata["model"],metadata["revision"])!=MODELS[key]:raise RuntimeError("Pinned metadata changed")
    records=[]
    for spec in metadata["assets"]:
        name=spec["rfilename"];target=root/name;partial=root/(name+".partial")
        if target.exists():
            if not check(target,spec):raise RuntimeError("Preserve mismatching existing official asset")
            continue
        if not name.endswith(".safetensors"):raise RuntimeError("Metadata must already be verified")
        prefix=partial.stat().st_size if partial.exists() else 0
        if prefix>spec["size"]:raise RuntimeError("Oversized old partial")
        prefix_sha=sha(partial) if partial.exists() else None
        record=dict(file=name,preserved_prefix_bytes=prefix,preserved_prefix_sha256=prefix_sha)
        records.append(record);write(recovery/"prefix_receipt.json",records)
        offset=prefix
        url=f"https://hf-mirror.com/{metadata['model']}/resolve/{metadata['revision']}/{name}"
        def download(bounds):
            start,end=bounds
            p=recovery/f"{name}.{start}_{end}.chunk"
            old=[directory/p.name for directory in priors if (directory/p.name).exists()]
            if old:shutil.copyfile(max(old,key=lambda f:f.stat().st_size),p)
            received=p.stat().st_size if p.exists() else 0
            if received>end-start+1:raise RuntimeError("Oversized preserved transport piece")
            if received==end-start+1:return p
            failures=0;slow_continuations=0
            # Progressing bounded bodies are new exact ranges, not failed retries.
            # Transport errors/429 retain a three-request failure budget; even
            # progress resumes stop after24 pieces or the original absolute deadline.
            while failures<3 and slow_continuations<24:
                if time.time()>=DEADLINE:raise TimeoutError("Rental continuation deadline reached")
                cursor=start+received
                response=None
                try:
                    response=requests.get(url,headers={"Range":f"bytes={cursor}-{end}"},stream=True,timeout=(15,8))
                    if response.status_code==429:
                        value=response.headers.get("Retry-After","30")
                        response.close();delay=int(value) if value.isdigit() else 30
                        if time.time()+delay>=DEADLINE:raise TimeoutError("Retry-After exceeds finite deadline")
                        time.sleep(delay);failures+=1;continue
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {cursor}-{end}/{spec['size']}":
                        response.close();raise RuntimeError("Exact HTTP range mismatch; do not adopt bytes")
                    request_started=time.monotonic()
                    with p.open("ab") as stream:
                        for block in response.iter_content(64*1024):
                            if time.time()>=DEADLINE:raise TimeoutError("Finite transfer deadline")
                            if block:
                                received+=len(block)
                                if received>end-start+1:raise RuntimeError("HTTP range overflow")
                                stream.write(block)
                            if time.monotonic()-request_started>45 and received<end-start+1:
                                raise requests.Timeout("Bounded slow-body request; continue from received bytes")
                    response.close()
                    if p.stat().st_size!=end-start+1:raise RuntimeError("Incomplete exact range")
                    return p
                except requests.Timeout as error:
                    if str(error)=="Bounded slow-body request; continue from received bytes":
                        slow_continuations+=1
                    else:
                        failures+=1
                        if failures==3:raise
                    print(f"[resume piece] {key}/{name} offset={start+received}",flush=True)
                    time.sleep(1)
                except requests.RequestException:
                    failures+=1
                    if failures==3:raise
                    time.sleep(3*failures)
                finally:
                    if response is not None:response.close()
            raise RuntimeError("Finite HTTP retry budget exhausted")
        with partial.open("ab") as output, ThreadPoolExecutor(max_workers=WORKERS) as pool:
            if partial.stat().st_size!=offset:raise RuntimeError("Old downloader still writing; refuse overlap")
            bounds=[(s,min(s+CHUNK,spec["size"])-1) for s in range(offset,spec["size"],CHUNK)]
            futures={pool.submit(download,b):b for b in bounds};finished={}
            for future in as_completed(futures):
                start,end=futures[future]
                try:finished[start]=(end,future.result())
                except BaseException as error:
                    write(recovery/"status.json",dict(status="failed",file=name,error=repr(error),time=time.time()))
                    for pending in futures:pending.cancel()
                    raise
                while offset in finished:
                    end,chunk=finished.pop(offset)
                    with chunk.open("rb") as source:
                        for block in iter(lambda:source.read(4*1024*1024),b""):output.write(block)
                    output.flush();os.fsync(output.fileno());offset=end+1
                    # Only this worker's redundant transport bytes; original prefix remains intact.
                    chunk.unlink()
                write(recovery/"status.json",dict(status="continuing",file=name,bytes=offset,total=spec["size"],
                    completed_chunks=sum(f.done() and not f.cancelled() for f in futures),total_chunks=len(futures),time=time.time()))
                print(f"[continued] {key}/{name} {offset}/{spec['size']}",flush=True)
        if not check(partial,spec):raise RuntimeError("Complete official LFS SHA mismatch; preserve full partial")
        partial.rename(target)
        record.update(status="official_sha_verified",complete_bytes=spec["size"],sha256=spec["lfs"]["sha256"])
        write(recovery/"prefix_receipt.json",records)
        print(f"[verified] {key}/{name}",flush=True)
    files={spec["rfilename"]:sha(root/spec["rfilename"]) for spec in metadata["assets"]}
    write(root/"asset_receipt.json",dict(status="complete",model=MODELS[key][0],revision=MODELS[key][1],
        files=files,completed=time.time(),continuation="recorded original prefixes, exact ranges, full official SHA"))
    write(recovery/"status.json",dict(status="complete",time=time.time()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--attempt",type=int,choices=(1,2,3),default=1)
    args=p.parse_args()
    try:run(args.model,args.attempt)
    except BaseException as error:
        root=Path("/data/models")/args.model/f"range_recovery{args.attempt:02d}";root.mkdir(exist_ok=True)
        write(root/"status.json",dict(status="failed",error=repr(error),time=time.time()))
        raise

if __name__=="__main__":main()
