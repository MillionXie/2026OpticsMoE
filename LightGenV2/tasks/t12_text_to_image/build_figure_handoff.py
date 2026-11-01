"""Package immutable T12 TEST triplets and raw metrics for scientific figure work.

No training, scoring, pixel editing, dataset filtering or network access. The
curated examples never replace the complete500 denominator.
"""
import argparse,csv,hashlib,json,shutil,statistics,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

SIM_SHA='aa2d277bd743d4dd831ec33852629dbf3001e772b55a8ac436a0d2a557a06fd8'
QWEN_SHA='68e04d5f378e2bc0a495363b45d7d8c68e81c41cf4169329749712d23d8998f6'
METRICS=('CLIP_T','DINOv2_I','CLIP_I')
MODELS=('Qwen','Simulation','Experiment')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def write_csv(path,headers,rows):
 with path.open('w',encoding='utf-8-sig',newline='')as f:
  w=csv.writer(f);w.writerow(headers);w.writerows(rows)

def join_records(post,experiment):
 original=read(post/'Qwen_reference_metrics/per_image.json')
 groups=[{r['sample_id']:r for r in original if r['checkpoint_sha256']==s}for s in (QWEN_SHA,SIM_SHA)]
 quality=post/('quality_'+experiment.removeprefix('EXP_'))
 groups.append({r['sample_id']:r for r in read(quality/'per_image.json')if r['run_id']==experiment})
 if any(len(g)!=500 for g in groups)or not all(g.keys()==groups[0].keys()for g in groups):raise ValueError('All three complete500 cohorts required')
 rows=[]
 for sid in groups[1]:
  values=[g[sid]for g in groups];ref=values[1]
  if not all(all(v[k]==ref[k]for k in ('input_sha256','prompt','seed','initial_latent_sha256'))for v in values):raise ValueError('Input/prompt/seed/noise identity changed')
  rows.append(dict(sample_id=sid,category=ref['category'],prompt=ref['prompt'],reference_id=ref['reference_id'],
                   seed=ref['seed'],models=dict(zip(MODELS,values))))
 if {c:sum(r['category']==c for r in rows)for c in ('lamp','pillow')}!={'lamp':250,'pillow':250}:raise ValueError('Changed category denominator')
 return rows

def aggregates(rows):
 result=[]
 for cohort in ('all','lamp','pillow'):
  group=[r for r in rows if cohort=='all'or r['category']==cohort]
  for metric in METRICS:
   values={model:statistics.mean(r['models'][model][metric]for r in group)for model in MODELS}
   delta=values['Experiment']/values['Simulation']-1
   result.append(dict(cohort=cohort,metric=metric,n=len(group),**values,experiment_relative_difference=delta))
 return result

def selection_rule(overall):
 group=[r for r in overall if r['cohort']=='all'];above=sum(r['experiment_relative_difference']>1e-12 for r in group)
 return dict(overall_items_above_simulation=above,overall_at_most_one_above_and_within5=above<=1 and all(abs(r['experiment_relative_difference'])<=.05+1e-12 for r in group),
             overall_max_abs_relative_difference=max(abs(r['experiment_relative_difference'])for r in group))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('post','selection','out'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--experiment',required=True);p.add_argument('--source-commit',required=True)
 p.add_argument('--allow-unmet',action='store_true');a=p.parse_args()
 if len(a.source_commit)!=40:raise ValueError('Published packaging source commit required')
 post=a.post.resolve();out=a.out.resolve()
 if not out.is_relative_to(post)or out==post:raise ValueError('Package must remain inside the existing private run post directory')
 selection=read(a.selection)
 if not selection.get('complete')or selection.get('reviewed_experiment')!=a.experiment:raise ValueError('Exactly this experimental version must be visually reviewed first')
 selected=selection['rows']
 if len(selected)!=20 or len({r['sample_id']for r in selected})!=20:raise ValueError('Twenty distinct curated requests required')
 rows=join_records(post,a.experiment);by_id={r['sample_id']:r for r in rows};overall=aggregates(rows);rule=selection_rule(overall)
 if not rule['overall_at_most_one_above_and_within5']and not a.allow_unmet:raise ValueError('Candidate does not satisfy the authorized overall rule; never relabel as accepted')
 for r in selected:
  if r['sample_id']not in by_id or r.get('prompt_review')!='all_three_pass':raise ValueError('Missing explicit three-model prompt review')
 out.mkdir(parents=True,exist_ok=True);assets=[]
 folders={'Qwen':'Qwen_reference','Simulation':'original_SIM_reference','Experiment':a.experiment}
 indices={r['sample_id']:str(i+1).zfill(2)for i,r in enumerate(selected)}
 compact=[]
 for row in rows:
  item={k:row[k]for k in ('sample_id','category','prompt','reference_id','seed')};item['example_id']=indices.get(row['sample_id'],'');item['scores']={};item['paths']={}
  for model in MODELS:
   source=row['models'][model];name=source['generated_PNG']
   if Path(name).name!=name:raise ValueError('Unsafe image filename')
   src=post/folders[model]/name;digest=sha(src)
   if digest!=source['generated_sha256']:raise ValueError('Generated image SHA changed')
   with Image.open(src)as im:
    if im.mode!='RGB'or im.size!=(512,512):raise ValueError('Expected original512 RGB')
    im.verify()
   relative=Path('all_test')/model/name;target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():
    if sha(target)!=digest:raise ValueError('Unknown existing image protected')
   else:shutil.copy2(src,target)
   item['scores'][model]={m:source[m]for m in METRICS};item['paths'][model]=relative.as_posix()
   assets.append(dict(path=relative.as_posix(),SHA256=digest,bytes=src.stat().st_size))
  compact.append(item)
 fonts={name:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)for name,size in [('prompt',25),('label',25),('metric',22)]}
 examples=[]
 for i,pick in enumerate(selected):
  row=by_id[pick['sample_id']];idx=str(i+1).zfill(2);folder=out/'selected20'/idx;folder.mkdir(parents=True,exist_ok=True)
  card=Image.new('RGB',(1608,770),'white');draw=ImageDraw.Draw(card)
  for n,line in enumerate(textwrap.wrap(row['prompt'],116)):draw.text((24,14+31*n),line,font=fonts['prompt'],fill='#111111')
  for j,model in enumerate(MODELS):
   source=row['models'][model];path=out/'all_test'/model/source['generated_PNG'];target=folder/(model+'.png')
   if target.exists()and sha(target)!=sha(path):raise ValueError('Existing selected image differs')
   if not target.exists():shutil.copy2(path,target)
   x=24+j*528;draw.text((x,115),{'Qwen':'Qwen3-VL','Simulation':'LightGen-II (simulation)','Experiment':'LightGen-II (experiment)'}[model],font=fonts['label'],fill='#111111')
   with Image.open(path)as im:card.paste(im,(x,157))
   draw.text((x,687),f"CLIP-T {source['CLIP_T']:.4f}   DINO-I {source['DINOv2_I']:.4f}",font=fonts['metric'],fill='#222222')
   draw.text((x,720),f"CLIP-I {source['CLIP_I']:.4f}",font=fonts['metric'],fill='#222222')
  card.save(folder/'comparison.png')
  (folder/'prompt.txt').write_text(row['prompt']+'\n',encoding='utf-8')
  write_csv(folder/'metrics.csv',['Group',*METRICS],[[m,*[row['models'][m][k]for k in METRICS]]for m in MODELS])
  examples.append(dict(example_id=idx,sample_id=row['sample_id'],category=row['category'],prompt=row['prompt'],prompt_review='all_three_pass',
    panel=f'selected20/{idx}/comparison.png',scores={m:{k:row['models'][m][k]for k in METRICS}for m in MODELS}))
 data=dict(TEST=500,lamp=250,pillow=250,metrics=METRICS,models=MODELS,overall=overall,rule=rule,rows=compact,examples=examples,
  simulation_SHA256=SIM_SHA,Qwen_SHA256=QWEN_SHA,experiment_SHA256=read(post/a.experiment/'report.json')['checkpoint_sha256'],
  experiment_run=a.experiment,fixed_original_SIM=True,same_checkpoint_comparison=False,TEST_selection_development_only=True,
  example_selection='Curated visual examples; not a random sample and not used to compute overall metrics',packaging_source_commit=a.source_commit)
 dump(out/'results.json',data)
 headers=['Sample ID','Category','Prompt']+[model+' '+metric for model in MODELS for metric in METRICS]+[model+' image'for model in MODELS]+['Example ID']
 flat=[[r['sample_id'],r['category'],r['prompt'],*[r['scores'][m][k]for m in MODELS for k in METRICS],*[r['paths'][m]for m in MODELS],r['example_id']]for r in compact]
 write_csv(out/'TEST500_per_image.csv',headers,flat)
 write_csv(out/'overall_metrics.csv',['Cohort','Metric','N',*MODELS,'Experiment / simulation - 1'],[[r['cohort'],r['metric'],r['n'],*[r[m]for m in MODELS],r['experiment_relative_difference']]for r in overall])
 dump(out/'image_manifest.json',dict(complete=True,raw_images=1500,rows=assets,source_commit=a.source_commit,raw_pixels_unchanged=True))
 (out/'README.txt').write_text('T12 TEST500 figure materials\n\nQwen3-VL is the electronic-system reference, not a bare-Qwen image generator.\nSimulation is the frozen original pre-adaptation aa2 model. Experiment uses the stated calibrated electronic head with actual CCD replay; this is not a same-checkpoint comparison.\n\nT12_TEST_results.xlsx: overall means,20 curated examples and all500 per-image records.\nselected20:20 comparison panels with the original prompt and the three raw metrics per image; original PNGs are included.\nall_test: all500 PNGs for each of Qwen, simulation and experiment. No raw image pixels were edited.\nCLIP-T evaluates text-image agreement. DINO-I and CLIP-I compare appearance with the reference image. These are similarity proxies, not literal prompt-compliance accuracy.\nThe20 examples were manually selected and do not replace the complete500 metric denominator. All TEST-based candidate selection is development selection, not independent generalization.\n',encoding='utf-8')
 print(json.dumps(dict(complete=True,experiment=a.experiment,TEST=500,examples=len(examples),raw_images=len(assets),rule=rule,output=str(out))))

if __name__=='__main__':main()
