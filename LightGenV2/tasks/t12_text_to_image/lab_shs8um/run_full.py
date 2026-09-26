"""Sequential complete TEST acquisition, bounded GPU memory, resumable batches."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import torch


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--abo-project',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=6)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    from LightGenV2.tasks.t12_text_to_image.product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
    from LightGenV2.tasks.t12_text_to_image.qwen_mini_small import PromptEmbeddingLookup
    data=a.project/'assets/datasets'
    dataset=ExpandedUnifiedProductEditDataset(data/'abo_cleanrender_lamp_table_pillow_256_v1','test',256,
        data/'abo_unified_expanded_instructions_qwen2_v2.pt')
    lookup=PromptEmbeddingLookup(data/'abo_unified_expanded_qwen_embeddings_v2.pt')
    all_rows=[]
    for start in range(0,len(dataset),a.batch_size):
        stop=min(start+a.batch_size,len(dataset));folder=a.output/f'batch_{start:05d}_{stop:05d}'
        if not (folder/'report.json').exists():
            if folder.exists():raise RuntimeError(f'Incomplete batch retained: {folder}; inspect before resuming')
            items=[dataset[i] for i in range(start,stop)]
            embeddings,mask,_=lookup.batch([r['prompt'] for r in items],torch.device('cpu'))
            reference=torch.stack([r['reference'] for r in items])
            # Stable per-source noise independent of batch size or resume.
            noise=torch.stack([torch.randn(reference[0].shape,generator=torch.Generator().manual_seed(1042+i)) for i in range(start,stop)])
            inputs=a.output/'active_inputs.pt'
            torch.save(dict(reference=reference,target=torch.stack([r['target'] for r in items]),
                embeddings=embeddings,mask=mask,noise=noise,indices=list(range(start,stop)),
                metadata=[{k:r[k] for k in ['sample_id','prompt','category','mode']} for r in items],
                scope='complete TEST member; aggregate is valid only after all batches complete'),inputs)
            write(a.output/'progress.json',dict(status='running',completed=start,total=len(dataset),active_batch=str(folder)))
            subprocess.run([sys.executable,'-u','-m','LightGenV2.tasks.t12_text_to_image.lab_shs8um.run_pilot',
                '--project',str(a.project),'--abo-project',str(a.abo_project),'--output',str(folder),'--inputs',str(inputs)],check=True)
            inputs.unlink()
        report=json.loads((folder/'report.json').read_text())
        assert report['status']=='complete' and report['sample_count']==stop-start
        assert report['indices']==list(range(start,stop))
        rows=json.loads((folder/'sample_metrics.json').read_text())
        for index,row in zip(range(start,stop),rows):
            row['test_index']=index;row['sample_id']=f'test_{index:05d}'
            for label in ('reference','target','simulation','physical'):
                row[label+'_image']=folder.name+'/'+row[label+'_image']
        all_rows.extend(rows)
        write(a.output/'progress.json',dict(status='running',completed=stop,total=len(dataset)))
    assert len(all_rows)==len(dataset)
    write(a.output/'sample_metrics.json',all_rows)
    with (a.output/'sample_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    metrics={key:sum(row[key] for row in all_rows)/len(all_rows) for key in
        ('physical_mse_0_1','physical_mae_0_1','physical_psnr_db','physical_ssim',
         'simulation_mse_0_1','simulation_mae_0_1','simulation_psnr_db','simulation_ssim')}
    write(a.output/'report.json',dict(status='complete',scope='entire original TEST split, no sample selection',
        sample_count=len(all_rows),metrics=metrics,noise_seed='1042+test_index',native_image_size=[256,256]))
    write(a.output/'progress.json',dict(status='complete',completed=len(all_rows),total=len(dataset)))


if __name__=='__main__':main()
