"""Persistent model and hardware; complete TEST one optical stage at a time."""
import argparse,json,sys,time,csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .ccd_bridge import attach
from .bounded_amplitude import save_bmps
from .export_samples import export
from ..sealed_editor import build_sealed
from ..audited_unified import architecture_report
from ..product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from ..qwen_mini_small import PromptEmbeddingLookup
from .run_full import write

STAGES=['language_router','language_expert','language_global','vision_router','vision_expert','vision_global']
class StageDone(Exception):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--abo-project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse',type=Path,required=True);p.add_argument('--selftest',action='store_true')
    p.add_argument('--split',choices=('test','val'),default='test')
    p.add_argument('--max-samples',type=int)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    sys.path.insert(0,str(a.abo_project/'lab_dvp8um'))
    import four_image_flow as flow
    from shs_physical2400 import SHSBench,CORNERS
    flow.BASE_CORNERS=CORNERS.copy()
    contract=json.loads((a.project/'assets/contract.json').read_text())
    assert flow.sha(a.project/'assets/small.pt')==contract['checkpoint_sha256']
    model=build_sealed(torch.load(a.project/'assets/small.pt',map_location='cpu',weights_only=False)).cuda().eval().requires_grad_(False)
    assert architecture_report(model)['counted_parameters']==9958098
    assert model.bounded_amplitude==dict(kind='tanh',scale=.5)
    data=a.project/'assets/datasets'
    dataset=ExpandedUnifiedProductEditDataset(data/'abo_cleanrender_lamp_table_pillow_256_v1',a.split,256,data/'abo_unified_expanded_instructions_qwen2_v2.pt')
    lookup=PromptEmbeddingLookup(data/'abo_unified_expanded_qwen_embeddings_v2.pt')
    assert len(dataset)==2304
    selected_indices=list(range(len(dataset))) if a.max_samples is None else torch.linspace(0,len(dataset)-1,min(a.max_samples,len(dataset))).long().tolist()
    total=len(selected_indices)
    phase_dir=a.output/'phase';phase_dir.mkdir(exist_ok=True)
    reused={}
    for start in range(0,total,6):
        folder=a.reuse/f'batch_{start:05d}_{min(start+6,total):05d}'
        if not (folder/'report.json').exists():break
        if a.split!='test' or a.max_samples is not None:raise ValueError('Subset/VAL captures cannot reuse original TEST capture folders')
        report=json.loads((folder/'report.json').read_text())
        assert report['contract']['checkpoint_sha256']==contract['checkpoint_sha256'] and report['status']=='complete'
        for index in range(start,min(start+6,total)):reused[index]=(folder,index-start)
    def inputs(start,stop):
        rows=[dataset[selected_indices[i]] for i in range(start,stop)]
        embeddings,mask,_=lookup.batch([r['prompt'] for r in rows],torch.device('cpu'))
        ref=torch.stack([r['reference'] for r in rows])
        return dict(reference=ref,target=torch.stack([r['target'] for r in rows]),embeddings=embeddings,mask=mask,
            noise=torch.stack([torch.randn(ref[0].shape,generator=torch.Generator().manual_seed(1042+selected_indices[i])) for i in range(start,stop)]),
            indices=selected_indices[start:stop],metadata=[{k:r[k] for k in ('sample_id','prompt','category','mode')} for r in rows],scope=a.split+' members; fixed indices recorded')
    def ccd_path(stage,index):
        if index in reused:
            folder,row=reused[index];return folder/'ccd'/stage/f'pilot_{row:03d}.png'
        return a.output/'ccd'/stage/f'test_{index:05d}.png'
    def load(stage,start,stop):
        return torch.from_numpy(np.stack([np.asarray(Image.open(ccd_path(stage,i))).copy() for i in range(start,stop)])).cuda().float()
    state=dict(stage=None,start=0,stop=0,bench=None,ideal={},selftest=a.selftest)
    def callback(stage,amplitude,phase,ideal):
        start,stop=state['start'],state['stop']
        if state['selftest']:
            if state['stage'] is None:return ideal
            if STAGES.index(stage)<STAGES.index(state['stage']):return state['ideal'][stage]
            state['ideal'][stage]=ideal.detach().clone();raise StageDone()
        if state['stage'] is None:return load(stage,start,stop)
        if STAGES.index(stage)<STAGES.index(state['stage']):return load(stage,start,stop)
        assert stage==state['stage']
        if float((phase-phase[:1]).abs().max())>1e-5:raise ValueError('Nonshared phase')
        phase_path=phase_dir/(stage+'.bmp')
        native=flow.phase_gray(phase[0].detach().cpu().numpy(),'hv',True)
        if phase_path.exists():assert np.array_equal(np.asarray(Image.open(phase_path)),native)
        else:Image.fromarray(native).save(phase_path)
        ids=[f'test_{i:05d}' for i in range(start,stop)]
        paths,_=save_bmps(amplitude.detach().cpu().numpy(),a.output,stage,ids,flow)
        try:state['bench'].capture(stage,phase_path,paths,ids,'flip_v')
        finally:
            for path in paths:path.unlink(missing_ok=True)
        raise StageDone()
    restore=attach(model,callback)
    def forward(x):return model(*[x[k].cuda().float() if k=='embeddings' else x[k].cuda() for k in ('reference','embeddings','mask','noise')])
    if a.selftest:
        x=inputs(0,6)
        with torch.inference_mode():baseline=forward(x)
        for stage in STAGES:
            state['stage']=stage
            try:
                with torch.inference_mode():forward(x)
            except StageDone:pass
        state['stage']=None
        # Replay ideal cache through identical sequential callbacks.
        state['selftest']=False
        for stage,value in state['ideal'].items():
            folder=a.output/'ccd'/stage;folder.mkdir(parents=True,exist_ok=True)
            torch.save(value,folder/'ideal.pt')
        def replay(stage,amplitude,phase,ideal):return state['ideal'][stage]
        restore();restore=attach(model,replay)
        with torch.inference_mode():actual=forward(x)
        error=float((actual-baseline).abs().max());assert error<1e-6,error
        write(a.output/'report.json',dict(status='complete',selftest=True,max_error=error,stages=STAGES));restore();return
    started=time.time();bench=None
    try:
        for attempt in range(3):
            try:bench=SHSBench(a.output,400,240,{});bench.__enter__();break
            except RuntimeError:
                if attempt==2:raise
                time.sleep(1)
        state['bench']=bench
        for stage in STAGES:
            state['stage']=stage
            for start in range(0,total,6):
                stop=min(start+6,total);state.update(start=start,stop=stop)
                if all(ccd_path(stage,i).exists() for i in range(start,stop)):continue
                write(a.output/'progress.json',dict(status='capturing',stage=stage,stage_completed=start,total=total,reused_complete_samples=len(reused),elapsed_seconds=time.time()-started))
                try:
                    with torch.inference_mode():forward(inputs(start,stop))
                except StageDone:pass
            write(a.output/'progress.json',dict(status='stage_complete',stage=stage,stage_completed=total,total=total,elapsed_seconds=time.time()-started))
    finally:
        if bench is not None:bench.__exit__(*sys.exc_info())
    state['stage']=None;all_rows=[]
    for start in range(0,total,6):
        stop=min(start+6,total);name=f'batch_{start:05d}_{stop:05d}';folder=a.output/name
        state.update(start=start,stop=stop)
        if start in reused:
            old=reused[start][0];rows=json.loads((old/'sample_metrics.json').read_text())
            for i,row in zip(range(start,stop),rows):
                row['test_index']=i;row['sample_id']=f'test_{i:05d}'
                for label in ('reference','target','simulation','physical'):row[label+'_image']=str(old/row[label+'_image'])
        else:
            folder.mkdir(exist_ok=True);x=inputs(start,stop)
            if not (folder/'report.json').exists():
                with torch.inference_mode():actual=forward(x)
                restore()
                with torch.inference_mode():simulation=forward(x)
                restore=attach(model,callback)
                for j in range(stop-start):
                    for label,value in [('reference',x['reference']),('target',x['target']),('simulation',simulation),('physical',actual)]:
                        gray=value[j].detach().float().cpu().add(1).mul(127.5).clamp(0,255).byte().permute(1,2,0).numpy()
                        Image.fromarray(gray).save(folder/f'pilot_{j:03d}_{label}.png')
                torch.save(dict(actual=actual.cpu(),simulation=simulation.cpu()),folder/'outputs.pt')
                inp=folder/'inputs.pt';torch.save(x,inp)
                write(folder/'report.json',dict(status='complete',selftest=False,sample_count=stop-start,sample_metadata=x['metadata'],indices=x['indices'],stages=STAGES,contract=contract,scope=a.split+' members; layerwise capture'))
                export(a.project,folder,inp);inp.unlink()
            rows=json.loads((folder/'sample_metrics.json').read_text())
            for i,row in zip(range(start,stop),rows):
                row[a.split+'_index']=selected_indices[i];row['dataset_split']=a.split;row['sample_id']=f'{a.split}_{selected_indices[i]:05d}'
                for label in ('reference','target','simulation','physical'):row[label+'_image']=name+'/'+row[label+'_image']
        all_rows.extend(rows);write(a.output/'progress.json',dict(status='decoding',completed=stop,total=total))
    restore();assert len(all_rows)==total
    write(a.output/'sample_metrics.json',all_rows)
    with (a.output/'sample_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    metrics={key:sum(row[key] for row in all_rows)/total for key in ('physical_mse_0_1','physical_mae_0_1','physical_psnr_db','physical_ssim','simulation_mse_0_1','simulation_mae_0_1','simulation_psnr_db','simulation_ssim')}
    write(a.output/'report.json',dict(status='complete',scope=a.split+' fixed members; layerwise capture',split=a.split,indices=selected_indices,contract=contract,sample_count=total,reused_samples=len(reused),metrics=metrics,elapsed_seconds=time.time()-started))
    write(a.output/'progress.json',dict(status='complete',completed=total,total=total))

if __name__=='__main__':main()
