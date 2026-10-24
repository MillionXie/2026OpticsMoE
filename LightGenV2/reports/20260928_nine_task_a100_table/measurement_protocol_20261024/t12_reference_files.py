"""Shared original T12 first200 reference-RGB/UTF8 files, not cached NN input.

Offline dataset materialization only: no model/embedding/accuracy inference.
Verify exact original reference pixels, then both systems read the same files
inside their formal cost windows. Never export GT as an inference input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from t12_live_frontend import TASK,ASSETS,DATA,TEST_SHA,PINS,reference_input,sha,write
CONTRACT='original_T12_first200_reference_RGB_and_UTF8_v1'


def validate_records(records):
    if len(records)!=200 or [r['index'] for r in records]!=list(range(200)):
        raise ValueError('Original first200 distinct TEST indices required')
    if len({r['sample_id'] for r in records})!=200:
        raise ValueError('Duplicate original editing sample')
    for i,r in enumerate(records):
        if (r['image']!=f'reference_{i:03d}.png' or r['text']!=f'instruction_{i:03d}.txt'
                or r['image_size']!=[256,256] or r['reference_bitwise_equal'] is not True
                or r['online_inputs_contain_target'] is not False):
            raise ValueError('Wrong original reference-only input contract')
        for key in ('png_sha256','utf8_sha256'):
            if len(r[key])!=64:raise ValueError('Original input SHA missing')


def load_bundle(root):
    root=Path(root).resolve();p=root/'input_manifest.json';value=json.loads(p.read_text(encoding='utf8'))
    if (value['contract']!=CONTRACT or value['status']!='complete'
            or value['original_test_manifest_sha256']!=TEST_SHA
            or value['no_neural_model_loaded'] is not True):
        raise ValueError('Complete original RGB/UTF8 materialization proof required')
    validate_records(value['records'])
    # Only stat, never pre-read PNG/UTF8 bytes before a formal cost window.
    for r in value['records']:
        if ((root/r['image']).stat().st_size!=r['png_bytes']
                or (root/r['text']).stat().st_size!=r['utf8_bytes']):
            raise ValueError('Materialized input file size changed')
    return value


def read_input(root,record):
    import numpy as np
    import torch
    from PIL import Image
    root=Path(root)
    with Image.open(root/record['image']) as image:
        if image.mode!='RGB' or image.size!=(256,256):raise ValueError('Original RGB256 PNG required')
        array=np.asarray(image,dtype=np.uint8).copy()
    text=(root/record['text']).read_bytes().decode('utf8')
    reference=torch.from_numpy(array).permute(2,0,1).float().div(127.5).sub(1)
    return reference,text


def verify_original_bytes(root,value):
    """Call after the cost window; SHA is not a selective input warmup."""
    root=Path(root)
    for r in value['records']:
        if sha(root/r['image'])!=r['png_sha256'] or sha(root/r['text'])!=r['utf8_sha256']:
            raise ValueError('Original RGB/UTF8 changed during measurement')


def materialize(output):
    if output.exists() or not output.resolve().is_relative_to(TASK/'runs/simulation'):
        raise ValueError('New immutable canonical input bundle required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('No-CUDA input preparation required')
    instruction=ASSETS/'datasets/abo_unified_expanded_instructions_qwen2_v2.pt'
    if sha(DATA/'test.jsonl')!=TEST_SHA or sha(instruction)!=PINS[instruction.name]:
        raise ValueError('Original input dataset/instruction identity differs')
    import torch
    from PIL import Image
    from LightGenV2.tasks.t12_text_to_image.product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
    torch.set_num_threads(4)
    dataset=ExpandedUnifiedProductEditDataset(DATA,'test',256,instruction)
    if len(dataset)!=2304:raise ValueError('Original complete TEST2304 identity required')
    output.mkdir(parents=True);records=[]
    modules=[Path(__file__),HERE/'t12_live_frontend.py',TASK/'product_unified_edit_data_v2.py',
        TASK/'product_object_replace_data.py',TASK/'product_global_redesign_data.py',
        TASK/'product_scene_replace_data.py',TASK/'product_unified_edit_data.py']
    frozen={p.relative_to(REPO).as_posix():sha(p) for p in modules}
    try:
        for i in range(200):
            # __getitem__ makes offline supervised targets as part of its
            # original dataset implementation. They are neither exported nor
            # passed to a model, and are outside BOTH systems' online costs.
            gold=dataset[i];live=reference_input(dataset,i)
            if (not torch.equal(gold['reference'],live['reference']) or gold['prompt']!=live['prompt']
                    or gold['sample_id']!=live['sample_id']):raise ValueError('Original reference-only reconstruction differs')
            pixels=gold['reference'].clamp(-1,1).add(1).mul(127.5).round().byte().permute(1,2,0).numpy()
            image=output/f'reference_{i:03d}.png';text=output/f'instruction_{i:03d}.txt'
            with image.open('xb') as stream:Image.fromarray(pixels).save(stream,format='PNG')
            with text.open('xb') as stream:stream.write(gold['prompt'].encode('utf8'))
            record=dict(index=i,sample_id=gold['sample_id'],image=image.name,text=text.name,
                png_bytes=image.stat().st_size,utf8_bytes=text.stat().st_size,png_sha256=sha(image),utf8_sha256=sha(text),
                image_size=[256,256],reference_bitwise_equal=True,online_inputs_contain_target=False)
            restored,prompt=read_input(output,record)
            if not torch.equal(restored,gold['reference']) or prompt!=gold['prompt']:
                raise ValueError('PNG/UTF8 file input does not preserve original float reference/prompt')
            records.append(record)
            if i==0 or (i+1)%25==0:write(output/'status.json',dict(status='materializing',samples=len(records),no_gpu=True))
        validate_records(records)
        if torch.cuda.is_initialized() or {p.relative_to(REPO).as_posix():sha(p) for p in modules}!=frozen:
            raise ValueError('CPU/source input preparation contract changed')
        value=dict(contract=CONTRACT,status='complete',records=records,
            original_test_manifest_sha256=TEST_SHA,instruction_cache_sha256=PINS[instruction.name],
            no_neural_model_loaded=True,no_embedding_lookup=True,no_GPU_inference=True,
            cuda_initialized=False,not_a_formal_cost_or_accuracy_evaluation=True,
            offline_GT_construction_only_for_original_dataset_reference_check=True,
            GT_files_exported=0,online_model_input_fields=['reference_RGB_file','UTF8_instruction_file'],
            sources=frozen,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            created_unix=time.time())
        write(output/'input_manifest.json',value)
        write(output/'status.json',dict(status='complete',samples=200,no_gpu=True,input_manifest_sha256=sha(output/'input_manifest.json')))
        print(json.dumps(dict(status='complete',samples=200,files=400,no_neural_model_loaded=True,
            input_manifest_sha256=sha(output/'input_manifest.json'))))
    except BaseException as error:
        write(output/'status.json',dict(status='failed',samples=len(records),error=repr(error),no_gpu=True));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    materialize(parser.parse_args().output.resolve())
