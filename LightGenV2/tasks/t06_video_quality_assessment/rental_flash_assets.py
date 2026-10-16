"""Fetch one official, ABI-pinned FlashAttention wheel with exact HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

NAME="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
SIZE=187826523
ASSET=224784686

def main():
    root=Path("/data/flash_attention_official_ranges01");root.mkdir(exist_ok=False)
    api="https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.7.4.post1"
    response=requests.get(api,timeout=30);response.raise_for_status()
    metadata=response.json()
    record=next(x for x in metadata["assets"] if x["id"]==ASSET)
    if record["name"]!=NAME or record["size"]!=SIZE:raise RuntimeError("Official release asset mismatch")
    url=record["browser_download_url"]
    (root/"official_asset.json").write_text(json.dumps({k:record.get(k) for k in
        ("id","name","size","digest","browser_download_url")},indent=2))
    chunk=(SIZE+15)//16
    def download(index):
        start=index*chunk;end=min(SIZE,start+chunk)-1
        if start>end:return None
        path=root/f"range{index:02d}.bin"
        for attempt in range(3):
            try:
                r=requests.get(url,headers={"Range":f"bytes={start}-{end}"},stream=True,timeout=(20,60))
                if r.status_code==429:
                    delay=r.headers.get("Retry-After","30");r.close()
                    time.sleep(min(int(delay) if delay.isdigit() else 30,60));continue
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{SIZE}":
                    raise RuntimeError("Exact official range mismatch")
                with path.open("wb" if attempt else "xb") as output:
                    for block in r.iter_content(1024*1024):
                        if block:output.write(block)
                r.close()
                if path.stat().st_size!=end-start+1:raise RuntimeError("Official range size mismatch")
                print(f"[official wheel range] {index}/15 verified-size",flush=True)
                return path
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(5*(attempt+1))
        raise RuntimeError("Finite rate-limit retries exhausted")
    with ThreadPoolExecutor(max_workers=16) as executor:
        parts=list(executor.map(download,range(16)))
    digest=hashlib.sha256()
    with (root/NAME).open("xb") as output:
        for path in parts:
            if path:
                with path.open("rb") as source:
                    for block in iter(lambda:source.read(4*1024*1024),b""):
                        output.write(block);digest.update(block)
    official_digest=record.get("digest")
    if official_digest and official_digest!="sha256:"+digest.hexdigest():raise RuntimeError("Official wheel SHA mismatch")
    (root/"receipt.json").write_text(json.dumps(dict(status="complete",size=SIZE,sha256=digest.hexdigest(),
        official_digest=official_digest,transport="16 exact HTTPS ranges from official GitHub release, not a third-party wheel",
        official_release_api=api),indent=2))

if __name__=="__main__":main()
