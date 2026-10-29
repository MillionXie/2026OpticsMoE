"""One frozen DeepSeek-VL2-Tiny ABO T2I pass, AFTER the authorized T06 queue.

Reuses the historical decoder/processor and prompt roles, without training,
calibration, generated text or a fitted quality head. Not a latency/energy run.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
GPU='GPU-30fecb94-0222-a3d5-d952-a91f2cab2f43'
CSV_SHA={'titles.csv':'c97e8da3c6c608a706f2cc0d284aeafb873fc6cc17d04a7c9f9f66b2135846c1',
         'test.csv':'4ca5834abf4bb4d614b41f05b5b2481b3d41720dd78c1c37b05923c97657dfb8'}
PROTOCOL='ABO_easy100_frozen_DeepSeekVL2Tiny_T2I_100x2400_A10080_20261029'
INPUT_MANIFEST_SHA='87d4213653d1df3051b982de712f8cabbc8e1ad4bb9d33efcaaf84b8a447c8dc'
ASSET_MANIFEST_SHA='96707edfffe966f91e095e2d9fcb7aeefe5590de7adddaf59025b507e9ac3fbd'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=True,indent=2)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def checked_relative(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!='images':raise ValueError('Unsafe original image path')
    return p.as_posix()


def validate_rows(titles,rows):
    if len(titles)!=100 or len(rows)!=2400 or len({r['sample_id'] for r in rows})!=2400:raise ValueError('Full original100/2400 contract')
    if [int(r['label']) for r in titles]!=list(range(100)):raise ValueError('Original title order')
    if Counter(int(r['label']) for r in rows)!=Counter({i:24 for i in range(100)}):raise ValueError('Exactly24 positives per title')
    for r in rows:
        if r['split']!='test' or titles[int(r['label'])]['product_id']!=r['product_id']:raise ValueError('Exact SKU TEST identity')
        checked_relative(r['image_path'])


def verify_inputs(data,manifest):
    for name,digest in CSV_SHA.items():
        if sha(data/name)!=digest:raise ValueError('Original CSV bytes required')
    titles=read_csv(data/'titles.csv');rows=read_csv(data/'test.csv');validate_rows(titles,rows)
    records=manifest['files']
    if [r['sample_id'] for r in records]!=[r['sample_id'] for r in rows]:raise ValueError('Transport order differs')
    for row,r in zip(rows,records):
        p=data/checked_relative(r['path'])
        if (r['path']!=row['image_path'] or r['product_id']!=row['product_id']
                or not p.resolve().is_relative_to(data.resolve())
                or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']):raise ValueError('Actual TEST bytes differ')
    return titles,rows


def verify_assets(model,manifest):
    if manifest['model_id']!='deepseek-ai/deepseek-vl2-tiny':raise ValueError('Tiny, not LGVQ non-Tiny')
    for r in manifest['files']:
        if Path(r['name']).name!=r['name']:raise ValueError('Flat original model assets')
        p=model/r['name']
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Original lab Tiny asset bytes differ')
    config=json.loads((model/'config.json').read_text(encoding='utf8'))
    if config['language_config']['hidden_size']!=1280:raise ValueError('Original Tiny hidden width required')


def predecessor():
    root=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/rental_timing_only_20261029_02'
    state=json.loads((root/'suite_status.json').read_text(encoding='utf8'))
    expected=[f'{m}_cap{c}_repeat{r:02d}' for m in ('kimi','deepseek') for c in (50,100) for r in (1,2,3)]
    if state['status']!='complete' or state['completed']!=expected or state.get('mps_released') is not True:raise ValueError('T06 must finish completely before ABO GPU admission')
    if Path('/proc/'+str(state['pid'])).exists():raise ValueError('T06 controller must truly exit')
    for name in expected:
        p=root/name
        if json.loads((p/'cpu_audit.json').read_text(encoding='utf8'))['status']!='passed':raise ValueError('T06 CPU audit incomplete')
    r=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',text=True,capture_output=True,timeout=5,
        env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY='/data/tmp/rental_timing_only_20261029_02'))
    if r.returncode!=1:raise ValueError('Old private MPS must be released')
    return dict(root=str(root),suite_sha256=sha(root/'suite_status.json'),all12_CPU_audits_passed=True)


class Guard:
    def __init__(self,deadline):self.deadline=deadline;self.events=[];self.error=None;self.stop=threading.Event();self.thread=None
    def check(self,allow_self):
        if time.time()>=self.deadline:raise TimeoutError('Finite ABO deadline; no paid-time extension')
        r=subprocess.run(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
        if r.returncode:raise RuntimeError('Cannot verify GPU process exclusion')
        processes=[line.strip().split(',') for line in r.stdout.splitlines() if line.strip()]
        self.events.append(dict(time=time.time(),processes=processes))
        if any(len(p)!=2 or p[1].strip()!=GPU or not allow_self or p[0].strip()!=str(os.getpid()) for p in processes):raise RuntimeError('Foreign GPU process; no signal to it and no automatic retry')
    def fast(self):
        if self.error:raise RuntimeError(self.error)
        if time.time()>=self.deadline:raise TimeoutError('Finite ABO deadline')
    def start(self):
        def monitor():
            while not self.stop.wait(1):
                try:self.check(True)
                except BaseException as error:
                    self.error=repr(error);os.kill(os.getpid(),signal.SIGINT);return
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=10)


def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'):raise ValueError('One exact full GPU, no inherited MPS cap')
    if not time.time()<a.deadline<=1793278441.7376518:raise ValueError('Within original rental hard cutoff only')
    if a.output.exists():raise FileExistsError('Fresh once-only output; no hidden feature resume')
    commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT']
    paths=[Path(__file__).resolve(),HERE/'abo_backbone_baselines.py']
    sources={p.relative_to(REPO).as_posix():sha(p) for p in paths}
    for name,digest in sources.items():
        blob=subprocess.check_output(['git','show',commit+':'+name],cwd=REPO,timeout=10)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Published Git source binding')
    evidence=predecessor()
    if sha(a.input_manifest)!=INPUT_MANIFEST_SHA or sha(a.asset_manifest)!=ASSET_MANIFEST_SHA:raise ValueError('Exact original lab transport manifests required')
    titles,rows=verify_inputs(a.data,json.loads(a.input_manifest.read_text(encoding='utf8')))
    assets=json.loads(a.asset_manifest.read_text(encoding='utf8'));verify_assets(a.model,assets)
    import fcntl
    with (REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard=Guard(a.deadline);bundle=None;a.output.mkdir(parents=True)
        identity=dict(protocol_id=PROTOCOL,source_commit=commit,sources=sources,command=sys.argv,pid=os.getpid(),
            gpu_uuid=GPU,predecessor=evidence,no_training=True,trainable_parameters=0,energy_qualified=False,
            performance_only_not_formal_timing=True,asset_manifest_sha256=sha(a.asset_manifest),input_manifest_sha256=sha(a.input_manifest),
            csv_sha256=CSV_SHA,model_assets=assets,reference_model='historical frozen DeepSeek-VL2-Tiny; not LGVQ DeepSeek-VL2')
        def status(value,**kw):
            # Append-only events retain every progress/failure, never delete a bad row.
            with (a.output/'status_events.jsonl').open('a') as f:f.write(json.dumps(dict(status=value,time=time.time(),pid=os.getpid(),**kw),ensure_ascii=True)+'\n')
        try:
            for _ in range(3):guard.check(False);time.sleep(10)
            gpu=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=uuid,name,memory.total,power.limit','--format=csv,noheader'],text=True,timeout=8)
            if GPU not in gpu or 'A100' not in gpu or '400.00 W' not in gpu:raise ValueError('Original A10080/400W identity')
            identity['hardware']=gpu
            identity['host_snapshot']=dict(uname=list(os.uname()),cpuinfo=Path('/proc/cpuinfo').read_text(),
                cgroup={str(p):p.read_text() for p in (Path('/sys/fs/cgroup/cpu.max'),Path('/sys/fs/cgroup/memory.max')) if p.exists()},
                host_topology_is_not_container_exclusive=True)
            guard.start();status('loading')
            import torch
            from LightGenV2.tasks.t08_abo_image_text_retrieval import abo_backbone_baselines as base
            identity['packages']={n:importlib.metadata.version(n) for n in ('torch','torchvision','transformers','numpy','xformers','accelerate','opencv-python','Pillow','scipy')}
            if identity['packages']['transformers']!='4.38.2' or identity['packages']['numpy']!='1.26.4':raise ValueError('Verified rental DS numeric/runtime packages')
            bundle=base._load_deepseek(a.model,torch.device('cuda:0'))
            if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).multi_processor_count!=108:raise ValueError('Actual full A100108SM required')
            if bundle.metadata['backbone_total_parameters']!=3370501440 or bundle.image_width!=1280 or bundle.metadata['backbone_trainable_parameters']!=0:raise ValueError('Exact frozen Tiny; no fitted head')
            identity['model_metadata']=bundle.metadata
            identity['loaded_deepseek_source_SHA']={name:dict(path=str(Path(m.__file__).resolve()),sha256=sha(Path(m.__file__))) for name,m in list(sys.modules.items()) if name.startswith('deepseek_vl2') and getattr(m,'__file__',None) and str(m.__file__).endswith('.py')}
            write(a.output/'identity.json',identity)
            image_features=[];text_features=[]
            with torch.inference_mode():
                for i,row in enumerate(rows):
                    guard.fast();images=base._load_images([row],a.data)
                    image_features.append(bundle.encode_images(images,base.I2T_QUERY).float().cpu())
                    for image in images:image.close()
                    if (i+1)%100==0:status('encoding_gallery',samples=i+1)
                for i,row in enumerate(titles):
                    guard.fast();text_features.append(bundle.encode_texts([row['title']],base.T2I_QUERY).float().cpu())
                    if (i+1)%25==0:status('encoding_titles',samples=i+1)
            images=torch.cat(image_features);texts=torch.cat(text_features);guard.check(True)
            scores=base._normalized(texts)@base._normalized(images).T
            metrics,order=base._gallery_metrics(scores,torch.tensor([int(r['label']) for r in titles]),torch.tensor([int(r['label']) for r in rows]))
            torch.save(dict(image_features=images,title_features=texts,scores=scores,full_order=order,
                gallery_ids=[r['sample_id'] for r in rows],gallery_product_ids=[r['product_id'] for r in rows],
                query_product_ids=[r['product_id'] for r in titles]),a.output/'features_and_rankings_fp32.pt')
            predictions=[]
            for q,row in enumerate(titles):
                ranking=order[q].tolist();first=next(i+1 for i,k in enumerate(ranking) if rows[k]['product_id']==row['product_id'])
                predictions.append(dict(query_index=q,product_id=row['product_id'],title=row['title'],first_positive_rank=first,
                    top10=[dict(rank=i+1,sample_id=rows[k]['sample_id'],product_id=rows[k]['product_id'],image_path=rows[k]['image_path'],score=float(scores[q,k]),correct=rows[k]['product_id']==row['product_id']) for i,k in enumerate(ranking[:10])]))
            write(a.output/'predictions.json',predictions)
            guard.fast();write(a.output/'report.json',dict(status='complete',**identity,query_count=100,gallery_count=2400,positives_per_query=24,
                metrics=metrics,features_sha256=sha(a.output/'features_and_rankings_fp32.pt'),predictions_sha256=sha(a.output/'predictions.json'),
                image_instruction=base.I2T_QUERY,title_instruction=base.T2I_QUERY,sort='FP32 L2 normalized cosine; stable descending original TEST order',
                GPU_mean_power=None,energy_j=None))
            status('complete')
        except BaseException as error:status('failed',error=repr(error));raise
        finally:
            guard.close();write(a.output/'process_audit.json',guard.events)
            if bundle:bundle.close()


def audit_saved_run(folder):
    """CPU-only replay of all100 complete stored FP32 rankings and SKU metrics."""
    folder=Path(folder);report=json.loads((folder/'report.json').read_text(encoding='utf8'))
    events=[json.loads(line) for line in (folder/'status_events.jsonl').read_text(encoding='utf8').splitlines()]
    if not events or events[-1]['status']!='complete':raise ValueError('Terminal complete status required')
    if (report['status']!='complete' or report['protocol_id']!=PROTOCOL or report['no_training'] is not True
            or report['trainable_parameters']!=0 or report['query_count']!=100 or report['gallery_count']!=2400
            or report['energy_qualified'] is not False or report['GPU_mean_power'] is not None or report['energy_j'] is not None):raise ValueError('Only complete frozen performance-only ABO run')
    if sha(folder/'features_and_rankings_fp32.pt')!=report['features_sha256'] or sha(folder/'predictions.json')!=report['predictions_sha256']:raise ValueError('Original saved score bytes differ')
    import torch
    from LightGenV2.tasks.t08_abo_image_text_retrieval import abo_backbone_baselines as base
    values=torch.load(folder/'features_and_rankings_fp32.pt',map_location='cpu',weights_only=False)
    predictions=json.loads((folder/'predictions.json').read_text(encoding='utf8'));gallery=values['gallery_ids']
    if len(gallery)!=2400 or len(set(gallery))!=2400 or len(predictions)!=100:raise ValueError('Complete IDs required')
    order=values['scores'].argsort(dim=1,descending=True,stable=True)
    if not torch.equal(order,values['full_order']):raise ValueError('Exact full stable ordering differs')
    # The stored gallery SKU mapping is part of the same frozen raw evidence.
    labels=values['gallery_product_ids'];queries=values['query_product_ids']
    unique=sorted(set(queries));mapping={v:i for i,v in enumerate(unique)}
    metrics,replayed=base._gallery_metrics(values['scores'],torch.tensor([mapping[v] for v in queries]),torch.tensor([mapping[v] for v in labels]))
    if metrics!=report['metrics']:raise ValueError('All100 metric replay differs')
    for event in json.loads((folder/'process_audit.json').read_text(encoding='utf8')):
        if any(str(p[0]).strip()!=str(report['pid']) or str(p[1]).strip()!=GPU for p in event['processes']):raise ValueError('Foreign GPU process in original audit')
    for q,row in enumerate(predictions):
        if row['product_id']!=queries[q] or [v['sample_id'] for v in row['top10']]!=[gallery[k] for k in replayed[q,:10].tolist()]:raise ValueError('Saved drawing Top10 differs')
    return dict(status='passed',protocol_id=PROTOCOL,query_count=100,gallery_count=2400,metrics=metrics,
        energy_qualified=False,report_sha256=sha(folder/'report.json'),exact_full_order_replayed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','model','input-manifest','asset-manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);a=p.parse_args();run(a)


if __name__=='__main__':main()
