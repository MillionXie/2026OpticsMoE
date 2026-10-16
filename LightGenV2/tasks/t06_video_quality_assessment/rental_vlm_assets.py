"""Pinned public assets for two authorized rental models, entirely under /data."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

MODELS = {
    "deepseek": ("deepseek-ai/deepseek-vl2", "f363772d1c47f4239dd844015b4bd53beb87951b"),
    "kimi": ("moonshotai/Kimi-VL-A3B-Instruct", "398eede0903cd983a2bfa0cc634e9ac1d843f375"),
}

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)

def check(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    if spec.get("lfs"):
        return sha(path) == spec["lfs"]["sha256"]
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == spec["blobId"]

def fetch(key, root, deadline):
    model, revision = MODELS[key]
    root.mkdir(parents=True, exist_ok=False)
    endpoint = "https://hf-mirror.com"
    response = requests.get(f"{endpoint}/api/models/{model}/revision/{revision}", params={"blobs":"true"}, timeout=45)
    response.raise_for_status(); metadata = response.json()
    if metadata["sha"] != revision:
        raise RuntimeError("Fixed official revision mismatch")
    assets = [f for f in metadata["siblings"] if "/" not in f["rfilename"] and not f["rfilename"].startswith(".")
              and (f["rfilename"].endswith((".json", ".py", ".safetensors", ".jinja", ".model")))]
    write(root/"official_metadata.json", dict(model=model, revision=revision, assets=assets))
    def download(spec):
        name = spec["rfilename"]; target = root/name
        partial = root/(name+".partial")
        offset = 0
        write(root/(name+".status.json"), dict(status="downloading", bytes=0, total=spec["size"]))
        with partial.open("xb") as stream:
            while offset < spec["size"]:
                end = min(offset + 256*1024*1024, spec["size"]) - 1
                for attempt in range(3):
                    if time.time() >= deadline:
                        raise TimeoutError("Rental asset deadline")
                    try:
                        r = requests.get(f"{endpoint}/{model}/resolve/{revision}/{name}",
                            headers={"Range": f"bytes={offset}-{end}"}, stream=True, timeout=(20,60))
                        if r.status_code == 429:
                            retry = r.headers.get("Retry-After", "30")
                            r.close(); time.sleep(min(float(retry) if retry.isdigit() else 30,60)); continue
                        r.raise_for_status()
                        if r.status_code == 206:
                            if r.headers.get("Content-Range", "").split("/")[0] != f"bytes {offset}-{end}":
                                raise RuntimeError("Exact HTTP range mismatch")
                        elif not (r.status_code == 200 and offset == 0 and end == spec["size"]-1):
                            raise RuntimeError("Server ignored range")
                        for chunk in r.iter_content(4*1024*1024):
                            if chunk:
                                if offset + len(chunk) > end+1:
                                    raise RuntimeError("Range overflow")
                                stream.write(chunk); offset += len(chunk)
                                write(root/(name+".status.json"), dict(status="downloading", bytes=offset,total=spec["size"], updated=time.time()))
                        r.close()
                        if offset != end+1:
                            raise RuntimeError("Incomplete range")
                        break
                    except (requests.RequestException, RuntimeError):
                        if attempt == 2:
                            raise
                        stream.flush()
                        # Resume only received bytes; SHA rejects any corrupt stream.
                        end = min(offset + 256*1024*1024,spec["size"])-1
                        time.sleep(5*(attempt+1))
                else:
                    raise RuntimeError("Finite HTTP retries exhausted")
        if not check(partial, spec):
            raise RuntimeError("Official asset hash mismatch: " + name)
        partial.rename(target)
        write(root/(name+".status.json"), dict(status="verified",bytes=offset,sha256=sha(target),updated=time.time()))
        print(f"[verified] {key}/{name} {offset}", flush=True)
    metadata_assets=[f for f in assets if not f["rfilename"].endswith(".safetensors")]
    weight_assets=[f for f in assets if f["rfilename"].endswith(".safetensors")]
    for item in metadata_assets:
        download(item)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, weight_assets))
    write(root/"asset_receipt.json",dict(status="complete",model=model,revision=revision,
        files={f["rfilename"]:sha(root/f["rfilename"]) for f in assets}, completed=time.time()))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--deadline",type=float,required=True)
    args=p.parse_args(); root=Path("/data/models")/args.model
    try:
        fetch(args.model,root,args.deadline)
    except BaseException as error:
        root.mkdir(parents=True,exist_ok=True)
        write(root/"failure.json",dict(error=repr(error),time=time.time()))
        raise

if __name__=="__main__": main()
