"""Original 150-pixel CC-BY-4.0 Kather2016 data; fixed full-class image split."""
import argparse,hashlib,io,json,subprocess,sys,time,urllib.request,zipfile
from pathlib import Path
import numpy as np
from PIL import Image

URL='https://zenodo.org/records/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1'
MD5='0ddbebfc56344752028fda72602aaade'
def digest(p,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);a.data_root.mkdir(parents=True,exist_ok=True);archive=a.data_root/'Kather_texture_2016_image_tiles_5000.zip'
    metadata=dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=digest(Path(__file__)),url=URL,license='CC BY 4.0',license_evidence='https://www.nature.com/articles/srep27988#Sec4',dataset_doi='10.5281/zenodo.53169',split_seed=20260917,patient_independence_verified=False)
    (a.out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    if not archive.exists():
        part=archive.with_suffix('.download');req=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0 academic reproduction'})
        with urllib.request.urlopen(req,timeout=120) as src,part.open('wb') as dst:
            for chunk in iter(lambda:src.read(2**20),b''):dst.write(chunk)
        assert digest(part,'md5')==MD5;part.rename(archive)
    assert digest(archive,'md5')==MD5;images=[];labels=[];ids=[];seen={};duplicates=[]
    with zipfile.ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if n.lower().endswith('.tif') and '__MACOSX' not in n)
        classes=sorted({Path(n).parent.name for n in names});assert len(classes)==8 and len(names)==5000,(classes,len(names))
        for name in names:
            label=classes.index(Path(name).parent.name);im=np.array(Image.open(io.BytesIO(z.read(name))).convert('RGB'));assert im.shape==(150,150,3)
            h=hashlib.sha256(im.tobytes()).hexdigest()
            if h in seen:
                assert seen[h]['label']==label,'Conflicting duplicate labels';duplicates.append(dict(removed=name,kept=seen[h]['id']));continue
            seen[h]=dict(id=name,label=label);images.append(im);labels.append(label);ids.append(name)
    x=np.stack(images);y=np.array(labels,dtype=np.int64);ids=np.array(ids);rng=np.random.default_rng(20260917);indices={k:[] for k in ['train','val','test']}
    for label in range(8):
        ix=rng.permutation(np.flatnonzero(y==label));n=len(ix);nt=int(.7*n);nv=(n-nt)//2
        for split,selected in zip(indices,[ix[:nt],ix[nt:nt+nv],ix[nt+nv:]]):indices[split].extend(selected.tolist())
    arrays={};manifests={};supports={}
    for split,ix in indices.items():
        ix=np.array(sorted(ix));arrays.update({split+'_images':x[ix],split+'_labels':y[ix],split+'_ids':ids[ix]});manifests[split]=ids[ix].tolist();supports[split]=np.bincount(y[ix],minlength=8).tolist()
    cache=a.data_root/'kather2016_fixed_split.npz';assert not cache.exists();np.savez_compressed(cache,**arrays)
    manifest=dict(metadata,classes=classes,raw_images=len(names),unique_images=len(x),duplicates=duplicates,split_ids=manifests,supports=supports,source_archive_sha256=digest(archive),source_archive_md5=MD5,cache_sha256=digest(cache),patient_id_policy='No unverified patient ID inferred from arbitrary image names. Image-level split, exact duplicates removed before split.',shape=[150,150,3])
    (a.out/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');(a.data_root/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(dict(classes=classes,supports=supports,duplicates=len(duplicates),cache=str(cache))),flush=True)
if __name__=='__main__':main()
