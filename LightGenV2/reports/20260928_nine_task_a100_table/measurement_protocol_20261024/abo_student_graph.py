"""Original adopted .8600 T08 graph and explicit neural-span preparation.

No inference cache, synthetic CCD or T07 standalone model is used. This module
is not a formal200x3 measurement; its executable mode is an actual-title CPU
original/instrumented numerical contract and records the original module ledger.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from measure_abo_text_qwen import TASK,MODEL_CONFIG_SHA,MODEL_WEIGHTS_SHA,sha,write,model_assets
from measure_salicon import one_index,shared

BODY_SHA='cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae'
PROFILE=TASK/'configs/text_to_image_10cm_adopted_eval.yaml'


def patch_optics(h,meter,mode):
    """Same original balanced optical path for Vision and Language; no math rewrite."""
    b=h.optical_branch;c=b.core
    for name in ('input_adapter','input_norm','output_norm','output_adapter'):
        meter.wrap(getattr(h,name),'forward',mode+'.'+name,family='adapter')
    meter.wrap(h,'_fuse',mode+'.fusion_audit',family='fusion',charged=False)
    for index,block in enumerate(h.blocks):
        meter.wrap(block,'forward',f'{mode}.parallel_residual{index+1}',family='parallel_residual',charged=False)
    body=shared.meter_lib.statements(c,'encode_groups');ix=shared.meter_lib.assign_index(body,'projected')
    meter.patch_statements(c,'encode_groups',[(mode+'.amplitude_linear_norm_softplus','amplitude_encoding_nn',True,ix,ix)])
    body=shared.meter_lib.statements(b,'_encode_input_fields');ix=shared.meter_lib.assign_index(body,'rms')
    meter.patch_statements(b,'_encode_input_fields',[(mode+'.amplitude_rms','amplitude_encoding_nn',True,ix,len(body)-1)])
    meter.wrap(c,'_direct_amplitude_load',mode+'.field_layout',family='layout',charged=False)
    meter.wrap(c.router,'_simulate',mode+'.router_simulation',family='simulation_only',charged=False)
    body=shared.meter_lib.statements(c.router,'forward');first=shared.meter_lib.assign_index(body,'energy')
    last=one_index(body,lambda s:s.startswith('weights, selected, indices ='))
    meter.patch_statements(c.router,'forward',[(mode+'.router_energy_top2','router',True,first,last)])
    meter.wrap(b,'_simulate_detector_roi',mode+'.expert_global_simulation',family='simulation_only',charged=False)
    body=shared.meter_lib.statements(b.ccd_normalizer,'forward');first=shared.meter_lib.assign_index(body,'value')
    next_nn=one_index(body,lambda s:s.startswith('frame_mean =')) if any(s.startswith('frame_mean =') for s in body) else len(body)-1
    groups=[(mode+'.ccd_value','ccd_nonlinearity',True,first,first),
        (mode+'.ccd_normalization','ccd_nonlinearity',True,next_nn,len(body)-1)]
    if first==next_nn:groups=[(mode+'.ccd_normalization','ccd_nonlinearity',True,first,len(body)-1)]
    meter.patch_statements(b.ccd_normalizer,'forward',groups)
    for name,readout,output in (('expert',b.expert_readout,b.expert_output_adapter),('global',c.readout,c.output_adapter)):
        body=shared.meter_lib.statements(readout,'forward_intensity')
        first=one_index(body,lambda s:s.startswith('detector_intensity =') and '.clamp_min(' in s)
        meter.patch_statements(readout,'forward_intensity',[(f'{mode}.{name}_ccd_pool_norm_nonlinearity','ccd_readout',True,first,len(body)-1)])
        meter.wrap(output,'forward',f'{mode}.{name}_ccd_output_linear',family='ccd_readout')


def patch(loaded,replacement,readout,meter):
    visual=loaded.model.model.visual;language=loaded.model.model.language_model
    meter.wrap(visual.patch_embed,'forward','frontend.patch_embedding',family='frontend')
    for name in ('fast_pos_embed_interpolate','rot_pos_emb'):
        meter.wrap(visual,name,'frontend.'+name,family='frontend')
    meter.wrap(visual.merger,'forward','bridge.native_merger',family='bridge')
    for index,merger in enumerate(visual.deepstack_merger_list):
        meter.wrap(merger,'forward',f'bridge.deepstack_merger{index}',family='bridge')
    meter.wrap(language.embed_tokens,'forward','frontend.token_embedding',family='frontend')
    meter.wrap(language.norm,'forward','language.native_final_norm',family='head')
    for mode in ('vision','language'):patch_optics(getattr(replacement,mode+'_surrogate').core,meter,mode)
    meter.wrap(replacement.language_surrogate,'retrieval_detector_features','head.detector_mean_max_pool',family='head')
    meter.wrap(readout,'forward','head.retrieval_readout',family='head')


def load(args,device):
    import torch
    import yaml
    from LightGenV2.tasks.t08_abo_image_text_retrieval.text_to_image import configuration
    from LightGenV2.tasks.t08_abo_image_text_retrieval import reverse_runtime as original
    if sha(args.checkpoint)!=BODY_SHA:raise ValueError('Only adopted .8600 body is allowed')
    weights=list(args.model.glob('*.safetensors'))
    if sha(args.model/'config.json')!=MODEL_CONFIG_SHA or len(weights)!=1 or sha(weights[0])!=MODEL_WEIGHTS_SHA:
        raise ValueError('Original native Embedding2B identity differs')
    # Only asset locations are rebound by the original fixed-profile entry.
    # The teacher cache is not consumed in online student inference.
    raw=configuration(PROFILE,model=args.model,data_root=args.data,checkpoint=args.checkpoint,
        teacher_cache=TASK/'runs/shared/abo_easy100_qwen64_true_text_to_image.pt',run_dir=args.output)
    config=args.output/'resolved_input.yaml';config.write_text(yaml.safe_dump(raw,sort_keys=False),encoding='utf8')
    settings=original.load_settings(config);settings.inference_batch_size=1
    settings.router_optimization_seed=42;settings.random_seed=42
    original.seed_everything(42);contract=original.load_contract(args.data);prompts=original._prompt_contract(raw)
    loaded=original.load_backbone(settings,torch.device(device));replacement,readout=original.build_student(loaded,settings)
    payload=original.load_checkpoint(args.checkpoint,replacement,readout)
    loaded.model.eval().requires_grad_(False);replacement.use_student()
    replacement.vision_surrogate.eval().requires_grad_(False);replacement.language_surrogate.eval().requires_grad_(False)
    replacement.set_phase_dropout_active(False);readout.eval().requires_grad_(False)
    info=dict(body_sha256=BODY_SHA,profile_sha256=sha(PROFILE),model_config_sha256=MODEL_CONFIG_SHA,
        model_weights_sha256=MODEL_WEIGHTS_SHA,model_assets=model_assets(args.model),epoch=payload.get('epoch'),
        original_full100_reference_hit1=.86,prompt_contract=vars(prompts),
        resolved_settings=json.loads(json.dumps(vars(settings),default=str)),
        original_module_sha256=sha(TASK/'reverse_runtime.py'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    return loaded,replacement,readout,settings,contract,prompts,info


def title_forward(loaded,replacement,readout,settings,text,prompts):
    import torch
    from LightGenV2.tasks.t08_abo_image_text_retrieval import reverse_runtime as original
    inputs=original._text_inputs(loaded.processor,[text],prompts.title_instruction)
    original._validate_text_token_budget(inputs,settings)
    with torch.autocast(device_type=loaded.device.type,dtype=torch.bfloat16,
            enabled=settings.amp_enabled and loaded.device.type=='cuda'):
        embedding,features=original.student_embeddings(loaded.model,replacement,readout,original.move_inputs(inputs,loaded.device))
    return embedding,features,inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(TASK/'runs/simulation'):
        raise ValueError('New immutable task contract run required')
    os.environ['CUDA_VISIBLE_DEVICES']='';args.output.mkdir(parents=True)
    write(args.output/'status.json',dict(status='loading',pid=os.getpid(),command=sys.argv,no_gpu=True))
    import torch
    from collections import Counter
    torch.set_num_threads(8)
    meter=shared.meter_lib.StageMeter();replacement=None
    try:
        loaded,replacement,readout,settings,contract,prompts,info=load(args,'cpu')
        from LightGenV2.tasks.t08_abo_image_text_retrieval import reverse_runtime as original
        checks=[]
        with torch.inference_mode():
            for index in (0,99):
                title=contract.titles[index]
                meter.close()
                expected=original._encode_titles_student(loaded,replacement,readout,[title],settings,prompts)
                patch(loaded,replacement,readout,meter);meter.sample_index=index
                actual,features,inputs=title_forward(loaded,replacement,readout,settings,title.text,prompts)
                if not torch.equal(expected,actual):raise ValueError('Original/instrumented full title NN is not bitwise equal')
                counts=Counter(r['stage'] for r in meter.rows if r['sample_index']==index)
                if counts['language.router_simulation']!=1 or counts['language.expert_global_simulation']!=2:
                    raise ValueError('Pure-title online three optical passes were not observed')
                if any(counts[n] for n in ('frontend.patch_embedding','bridge.native_merger','vision.router_simulation')):
                    raise ValueError('Unexpected online visual path in pure-title query')
                checks.append(dict(title_index=index,product_id=title.product_id,bitwise_equal=True,
                    input_shapes={k:list(v.shape) for k,v in inputs.items()},feature_shape=list(features.shape),
                    embedding_shape=list(actual.shape),module_call_counts=dict(counts)))
        if torch.cuda.is_initialized():raise ValueError('CPU contract created CUDA')
        # Keep both raw declarations; the unique ledger is convenient for readers.
        unique={json.dumps(entry,sort_keys=True):entry for entry in meter.ledger}
        write(args.output/'contract.json',dict(status='passed',mode='real_title_original_full_student_CPU_bitwise',
            no_gpu=True,checks=checks,identity=info,stage_ledger=meter.ledger,unique_stage_ledger=list(unique.values()),
            not_a_formal_timing_or_energy_run=True))
        write(args.output/'status.json',dict(status='complete',pid=os.getpid(),real_titles=2,no_gpu=True))
        print(json.dumps(dict(status='passed',real_titles=2,no_gpu=True)))
    except BaseException as error:
        write(args.output/'status.json',dict(status='failed',pid=os.getpid(),error=repr(error),no_gpu=True));raise
    finally:
        meter.close()
        if replacement is not None:replacement.close()


if __name__=='__main__':main()
