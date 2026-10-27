"""Content-reviewed, split-local development pairs; original data stay immutable."""
import argparse,collections,json,os,random,re,sys,time
from pathlib import Path

from PIL import Image,ImageDraw

from .qwen_vl_turbo_data import (_pixels_hash,add_descriptions,render_pair,seed,
                                sha256,validate_roles,write_json)


def reviewed_pairs(selected,descriptions,profile):
    from .product_scene_replace_data import COMBINATIONS
    pairs=[]
    for split,products in selected.items():
        groups=collections.defaultdict(list)
        for row in products:groups[row['category']].append(row)
        for source in products:
            canonical={}
            for r in groups[source['category']]:canonical.setdefault(r['product_id'],r)
            targets=sorted((r for r in canonical.values() if r['product_id']!=source['product_id']),
                           key=lambda r:r['sample_id'])
            if not targets:raise ValueError('Need another reviewed product in each category/split')
            rng=random.Random(seed(profile['seed'],source['sample_id'],'reviewed-targets'));rng.shuffle(targets)
            base=seed(profile['seed'],source['sample_id'],'reviewed-scene')%len(COMBINATIONS)
            for mode in ('background','object','joint'):
                count=profile.get('edits_per_mode',{}).get(split,2)
                for j in range(count):
                    target=targets[j%len(targets)]
                    source_scene=COMBINATIONS[(base+7*j)%len(COMBINATIONS)]
                    pairs.append({'sample_id':f"{split}:reviewed:{source['sample_id']}:{mode}:{j}",
                        'split':split,'category':source['category'],'mode':mode,'source':source,
                        'target':source if mode=='background' else target,
                        'source_scene':source_scene,
                        'target_scene':source_scene if mode=='object' else COMBINATIONS[(base+11+13*j)%len(COMBINATIONS)],
                        'noise_seed':profile['seed'],'explicit_category_teacher':True})
    rows=add_descriptions(pairs,descriptions)
    validate_roles(rows)
    for r in rows:
        if re.search(r'\bB0[A-Z0-9]{8}\b|\bSKU\b|\bAmazon\b',r['prompt'],re.I):
            raise ValueError('Product identity or brand is not permitted in prompts')
    return rows


def prior_exposure(rows,prior_rows):
    train_products={r[role]['product_id'] for r in prior_rows if r['split']=='train' for role in ('source','target')}
    train_pixels={r[role]['rgba_sha256'] for r in prior_rows if r['split']=='train' for role in ('source','target')}
    for r in rows:
        if r['split']!='val':continue
        for role in ('source','target'):
            if r[role]['product_id'] in train_products or r[role]['rgba_sha256'] in train_pixels:
                raise ValueError('New VAL was exposed to parent TRAIN: '+r[role]['product_id'])
    current_train={r[role]['product_id'] for r in rows if r['split']=='train' for role in ('source','target')}
    return {'checked_parent_train_roles':True,'new_val_parent_train_overlap':False,
            'current_train_products_seen_by_parent':sorted(current_train&train_products),
            'current_train_products_not_in_parent':sorted(current_train-train_products)}


def prepare(data,profile_path,out,category_data=None,parent_data=None):
    profile=json.loads(profile_path.read_text(encoding='utf-8'))
    if profile.get('requires_parent_exposure_audit') and parent_data is None:
        raise ValueError('This profile requires the full historical parent TRAIN exposure manifest')
    if (out/'pairs_manifest.json').exists():raise ValueError('Prepared data already exist; reuse, never overwrite')
    selected={};descriptions={};hashes={};category_data=category_data or {};loaded={}
    for split,approved in profile['products'].items():
        products=[];seen=set()
        for spec in approved:
            source_root=category_data.get(spec['category'],data)
            manifest=(source_root/(split+'.jsonl')).resolve()
            if manifest not in loaded:
                hashes[str(manifest)]=sha256(manifest)
                loaded[manifest]={r['sample_id']:r for line in manifest.read_text(encoding='utf-8').splitlines() if (r:=json.loads(line))}
            raw=loaded[manifest]
            sku=spec['product_id']
            if sku in seen or sku in profile['excluded_products']:raise ValueError('Repeated/excluded product')
            seen.add(sku)
            if len(set(spec['views']))!=len(spec['views']):raise ValueError('Repeated approved view')
            for view in spec['views']:
                sid=f"{spec['category']}-{sku}-{view:02d}"
                r=dict(raw[sid])
                if r['license']!='CC BY 4.0' or r['sequence_id']!=sku or r['category']!=spec['category']:raise ValueError('Raw identity/category/license differs')
                for key in ('image_path','mask_path'):r[key]=str((source_root/r[key]).resolve())
                r.update(product_id=sku,family_id=r.get('parent_id') or r.get('family_id') or sku)
                r['rgba_sha256']=_pixels_hash(r);products.append(r)
                descriptions[sid]=spec['description']+', entirely visible'
        selected[split]=products
    rows=reviewed_pairs(selected,descriptions,profile)
    audit=validate_roles(rows);audit.update({'profile_sha256':sha256(profile_path),
        'raw_manifest_sha256':hashes,'inventories':{s:{'products':len({r['product_id'] for r in v}),'views':len(v)} for s,v in selected.items()},
        'selection_basis':profile['selection_basis'],'manual_prompt_descriptions':profile.get('manual_prompt_descriptions',True),
        'annotation_method':profile.get('annotation_method','Manually reviewed product descriptions'),
        'evaluation_label':profile.get('evaluation_label','validation'),
        'test_selection_authorized':profile.get('test_selection_authorized',False),
        'target_view_policy':profile.get('target_view_policy','Object/joint targets use the first approved view per SKU; background edits retain the actual reference view.'),
        'source_scene_policy':f"Independent reference backgrounds per view: {profile.get('edits_per_mode',{'train':2,'val':2})}; no identical reference/prompt paired with conflicting target views",
        'family_scope':'Upstream family where present, otherwise SKU; no claim all similar-family products were independently audited',
        'prior_exposure':profile.get('prior_exposure','Warm-start parent trained on a superset of TRAIN products. This smaller profile does not erase prior training exposure.'),
        'metrics_scope':'New development VAL cohort. Rescore every comparison on these same pairs; do not compare its mean to old 144-VAL means.'})
    if parent_data:
        audit['parent_data_manifest']=str(parent_data.resolve())
        audit['parent_data_sha256']=sha256(parent_data)
        audit['parent_exposure_audit']=prior_exposure(rows,json.loads(parent_data.read_text(encoding='utf-8')))
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'pairs_manifest.json',{'config':profile,'audit':audit,'pairs':rows})
    write_json(out/'pilot_pairs.json',rows);write_json(out/'annotated_pairs.json',rows)
    write_json(out/'product_descriptions.json',descriptions);write_json(out/'data_audit.json',audit)
    for split,products in selected.items():
        sheet=Image.new('RGB',(6*180,((len(products)+5)//6)*210),'white');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(products):
            im=Image.open(r['image_path']).convert('RGB');im.thumbnail((180,180))
            x=i%6*180;y=i//6*210;sheet.paste(im,(x,y+30));draw.text((x+3,y+4),r['sample_id'],fill='black')
        sheet.save(out/(split+'_reviewed_products.png'))
    sheet=Image.new('RGB',(512,len(profile['categories'])*3*286),'white');draw=ImageDraw.Draw(sheet)
    for ci,cat in enumerate(profile['categories']):
        for mi,mode in enumerate(('background','object','joint')):
            r=next(x for x in rows if x['split']=='val' and (x['category'],x['mode'])==(cat,mode))
            y=(ci*3+mi)*286;draw.text((3,y+3),cat+'/'+mode+'  reference | target',fill='black')
            for j,im in enumerate(render_pair(r)):sheet.paste(im,(256*j,y+30))
    sheet.save(out/'reviewed_val_pairs.png')
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--qwen',type=Path)
    parser.add_argument('--sd',type=Path);parser.add_argument('--gpu-uuid');parser.add_argument('--stage',choices=('prepare','cache'),required=True)
    parser.add_argument('--category-data',action='append',default=[],metavar='CATEGORY=DIR')
    parser.add_argument('--parent-data',type=Path,help='Full historical TRAIN/VAL roles for exposure checks; never TEST')
    args=parser.parse_args()
    category_data={cat:Path(path) for cat,path in (value.split('=',1) for value in args.category_data)}
    if args.stage=='prepare':print(json.dumps(prepare(args.data,args.profile,args.output,category_data,args.parent_data),indent=2));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:raise ValueError('Select the authorized GPU explicitly')
    if not args.qwen or not args.sd:raise ValueError('Public Qwen and SD assets required')
    import torch
    from .qwen_vl_turbo_run import cache,git_identity,pretrained_manifest
    torch.set_num_threads(4);torch.manual_seed(1042);torch.cuda.set_per_process_memory_fraction(.65,0)
    audit=json.loads((args.output/'data_audit.json').read_text())
    if sha256(args.profile)!=audit['profile_sha256']:raise ValueError('Reviewed profile changed after preparation')
    if audit.get('parent_data_manifest') and sha256(Path(audit['parent_data_manifest']))!=audit['parent_data_sha256']:
        raise ValueError('Historical parent exposure manifest changed')
    for source,digest in audit['raw_manifest_sha256'].items():
        path=args.data/(source+'.jsonl') if source in ('train','val') else Path(source)
        if sha256(path)!=digest:raise ValueError('Raw split changed')
    cfg={'image_size':256,'use_all_pairs':True,'uint8_image_cache':True,'clip_cache_batch_size':16}
    rows=json.loads((args.output/'pilot_pairs.json').read_text());validate_roles(rows)
    write_json(args.output/'cache_execution.json',{'git':git_identity(),'source_sha256':sha256(Path(__file__)),
        'argv':sys.argv,'pid':os.getpid(),'gpu_uuid':args.gpu_uuid,'profile_sha256':sha256(args.profile),'started_unix':time.time()})
    write_json(args.output/'pretrained_manifest.json',{'qwen':pretrained_manifest(args.qwen),'sd_turbo':pretrained_manifest(args.sd)})
    digest=cache(args,cfg,rows,torch.device('cuda:0'))
    write_json(args.output/'cache_complete.json',{'complete':True,'sha256':digest,'samples':len(rows)})


if __name__=='__main__':main()
