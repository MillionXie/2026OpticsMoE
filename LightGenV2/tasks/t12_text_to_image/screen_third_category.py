"""CPU-only visual screening; no training, TEST images, weights or optical access."""
import argparse,collections,hashlib,json,os
from pathlib import Path

from PIL import Image,ImageDraw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','data','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    from .product_object_replace_data import ProductObjectReplacementDataset
    from .qwen_vl_turbo_data import _pixels_hash,sha256,write_json
    profile=json.loads(args.profile.read_text(encoding='utf-8'));raw={};manifest_sha={}
    for split in ('train','val'):
        path=args.data/(split+'.jsonl');manifest_sha[split]=sha256(path)
        raw[split]={r['sample_id']:r for line in path.read_text(encoding='utf-8').splitlines() if (r:=json.loads(line))}
    owners={};selected=[];panel=Image.new('RGB',(1024,572),'white');draw=ImageDraw.Draw(panel)
    for i,spec in enumerate(profile['samples']):
        r=dict(raw[spec['split']][spec['sample_id']])
        if r['category']!=profile['category'] or r['license']!='CC BY 4.0':raise ValueError('Wrong category/license')
        if r['sequence_id'] in profile['excluded_source_products']:raise ValueError('Excluded product')
        for key in ('image_path','mask_path'):r[key]=str((args.data/r[key]).resolve())
        digest=_pixels_hash(r)
        for identity in (r['sequence_id'],digest):
            if owners.setdefault(identity,spec['split'])!=spec['split']:raise ValueError('Cross-role identity overlap')
        fg,alpha=ProductObjectReplacementDataset._load_product(r,256)
        image=Image.composite(fg,Image.new('RGB',(256,256),'white'),alpha)
        x=i%4*256;y=i//4*286;panel.paste(image,(x,y+30))
        draw.text((x+3,y+5),spec['split'].upper()+' '+spec['sample_id'].split('-',1)[1],fill='black')
        selected.append({**spec,'product_id':r['sequence_id'],'rgba_sha256':digest,'raw':r})
    args.output.mkdir(parents=True,exist_ok=True);panel.save(args.output/'selected_ottoman_candidates.png')
    write_json(args.output/'selection.json',{'profile':profile,'profile_sha256':sha256(args.profile),
        'source_manifest_sha256':manifest_sha,'selected':selected,'counts':dict(collections.Counter(x['split'] for x in selected)),
        'split_inherited':True,'prompt_uses_brand_or_product_id':False,'trained':False,
        'test_images_opened':False,'legacy_test_metadata_read_in_broad_inventory':True,
        'note':'This is candidate screening, not a new independent TEST claim. No TEST image or product was reassigned. Preview uses the existing genuine-alpha foreground normalization, not model outputs.'})
    write_json(args.output/'screen_manifest.json',{p.name:sha256(p) for p in args.output.iterdir() if p.is_file() and p.name in ('selected_ottoman_candidates.png','selection.json')})
    print('Screened',len(selected),'candidates; original TRAIN/VAL assignment retained; CPU only')


if __name__=='__main__':main()
