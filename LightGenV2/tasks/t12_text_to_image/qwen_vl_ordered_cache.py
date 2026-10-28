"""Cache actual full-Qwen text-first features on the unchanged TRAIN/DEV cohort."""
import argparse,json,os
from pathlib import Path
import sys,time
import torch

from .qwen_vl_hybrid_run import load_data,load_qwen
from .qwen_vl_ordered_input import ORDER,ordered_inputs
from .qwen_vl_turbo_cache import save_with_digest
from .qwen_vl_turbo_data import sha256,write_json
from .qwen_vl_turbo_run import tensor_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','qwen','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid or a.output.exists():raise ValueError('Single approved GPU and new cache required')
    payload=load_data(a.data)
    if payload.get('frontend_input_order'):raise ValueError('Do not recache an already reordered cohort')
    a.output.mkdir(parents=True);write_json(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),
        'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'source_sha256':sha256(Path(__file__)),'started_unix':time.time()})
    torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.8,0)
    qwen,processor=load_qwen(a.qwen,device);image_id=json.loads((a.qwen/'config.json').read_text())['image_token_id']
    hidden=[];checks=[]
    with torch.no_grad():
        for i,row in enumerate(payload['rows']):
            native,perm,canonical=ordered_inputs(processor,tensor_image(payload['reference'][i]),row['prompt'])
            positions=torch.where(canonical['input_ids'][0]==image_id)[0].tolist()
            if positions!=list(range(4,68)) or canonical['input_ids'].shape[1]!=len(payload['hidden'][i]):raise ValueError('Audited canonical layout changed')
            out=qwen.model(**native.to(device),use_cache=False,return_dict=True).last_hidden_state[0]
            value=out[perm.to(device)].bfloat16().cpu()
            if value.shape!=payload['hidden'][i].shape or not torch.isfinite(value).all():raise ValueError('Invalid native Qwen output')
            hidden.append(value)
            if i in (0,len(payload['rows'])-1):checks.append({'sample_id':row['sample_id'],'native_positions':torch.where(native['input_ids'][0]==image_id)[0].tolist(),'canonical_positions':positions})
            if (i+1)%24==0:write_json(a.output/'progress.json',{'stage':'full_native_text_first_Qwen','done':i+1,'total':len(payload['rows'])})
    result=dict(payload);result['hidden']=hidden;result['frontend_input_order']=ORDER
    digest=save_with_digest(result,a.output/'frozen_features.pt')
    for name in ('annotated_pairs.json','pilot_pairs.json','product_descriptions.json','data_audit.json','pairs_manifest.json'):
        if (a.data/name).exists():(a.output/name).write_bytes((a.data/name).read_bytes())
    source=json.loads((a.data/'report.json').read_text())
    source.update(complete=True,frontend_input_order=ORDER,original_data_run=str(a.data),
        source_commit=os.environ.get('T12_SOURCE_COMMIT'),source_sha256=sha256(Path(__file__)),
        input_helper_sha256=sha256(Path(__file__).with_name('qwen_vl_ordered_input.py')),
        original_data_cache_sha256=json.loads((a.data/'cache_complete.json').read_text())['sha256'],
        cache_sha256=digest,model_predictions_are_images=False,independent_TEST_opened=False,
        readout_permutation_is_parameter_free=True,checks=checks,
        input_contract='Same original reference RGB and full instruction; native text-first Qwen24V/28L; only final readout states reordered to existing canonical head positions')
    write_json(a.output/'cache_complete.json',{'complete':True,'sha256':digest,'frontend_input_order':ORDER})
    write_json(a.output/'report.json',source)


if __name__=='__main__':main()
