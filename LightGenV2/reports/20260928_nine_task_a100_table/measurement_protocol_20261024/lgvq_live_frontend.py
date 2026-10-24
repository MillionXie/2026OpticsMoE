"""Original .8044/.6710 LGVQ live-video front and scoped CPU numerical proof.

No cached field is an online input. Only the original executed native patch,
position and embedding components are instantiated, not unused Transformers.
Formal full-TEST accuracy and 200x3 costs are NOT produced by this CPU entry.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
TASK = REPO / 'LightGenV2/tasks/t06_video_quality_assessment'
BACKEND = REPO / 'experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54'
OLD = REPO / 'experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa'
BINDING = TASK / 'runs/smoke/lgvq_frontend_identity_recovery_20261024_01/front_parameter_binding_02.json'
BINDING_SHA = '6bb38fe494d35cc568fc3e00e1bd9901ce6f9e88e5e5dd8d3d7041292d8d4ed0'
STEM_SHA = 'd357fe51b888ecace74c050096febebc09c08abc21d6b42f533ecc3cf1f4de55'
MANIFEST_SHA = '607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc'
MANIFEST = OLD / 'artifacts/lgvq_train2250_test558.csv'
STEM = OLD / 'runs/lgvq_oeo109_alpha50_kd300_center100_v3/best_observed_test_checkpoint.pt'
PINS = {
    'temporal': ('multivideo16x4_rank_s163', '5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c', 7, 16),
    'spatial': ('spatial_readout_1m_srcc067', '95e12397ccf8c960fa30ba9dfb400b69d2c2ebd02288ac6a4879870ab592828b', 14, 1),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def field_indices(test_count, videos_per_field, field_index):
    if test_count < videos_per_field or videos_per_field not in (1, 16):
        raise ValueError('Original 1/16-video field required')
    indices = list(range(test_count)); valid = [True]*test_count
    padding = (-test_count) % videos_per_field
    indices += indices[:padding]; valid += [False]*padding
    start = field_index * videos_per_field
    if not 0 <= start < len(indices): raise IndexError(field_index)
    return indices[start:start+videos_per_field], valid[start:start+videos_per_field]


def residual_name(name):
    return name.split('.')[0] in {
        'vision_routes', 'language_routes', 'custom_conv_electronic_correction',
        'electronic_quality_norm', 'tiny_rgb_electronic_adapter', 'vgg_correction',
        'resnet_electronic_correction', 'mobilenet_electronic_correction',
    }


def restore_since(meter, start):
    for owner, method, original in reversed(meter._restore[start:]):
        setattr(owner, method, original)
    del meter._restore[start:]


def sources():
    paths = [Path(__file__), HERE/'stage_meter.py', HERE/'bind_lgvq_front.py',
        TASK/'lab_runtime.py', TASK/'models/multivideo9x4.py', TASK/'multivideo_settings.py',
        TASK/'multivideo_data.py', BACKEND/'modeling.py', BACKEND/'settings.py',
        BACKEND/'cache_qwen_front.py', BACKEND/'cache_quality_stem.py', OLD/'modeling.py',
        OLD/'settings.py', OLD/'data.py']
    return {p.relative_to(REPO).as_posix(): sha(p) for p in paths}


class LiveFront:
    def __init__(self, device='cpu'):
        import torch
        from torch import nn
        from safetensors import safe_open
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed, Qwen3VLVisionModel
        from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as front
        from bind_lgvq_front import native_key
        if sha(BINDING) != BINDING_SHA: raise ValueError('Original qualified component binding differs')
        self.binding = json.loads(BINDING.read_text())
        self.model_path = Path(self.binding['model_candidate'])
        self.device = torch.device(device)
        for item in self.binding['original_artifact_checks']:
            path = self.model_path/item['path']
            if path.stat().st_size != item['bytes']: raise ValueError('Native asset size differs')
            if item['sha256'] and sha(path) != item['sha256']: raise ValueError('Native config differs')
        config = AutoConfig.from_pretrained(str(self.model_path), local_files_only=True)

        class ExecutedNativeVisionFront(nn.Module):
            # Original method, with its original config and learned position table.
            # This is not Identity or a synthetic stand-in for executed layers.
            fast_pos_embed_interpolate = Qwen3VLVisionModel.fast_pos_embed_interpolate

            def __init__(self):
                super().__init__()
                self.config = config.vision_config
                self.patch_embed = Qwen3VLVisionPatchEmbed(self.config).to(torch.bfloat16)
                self.pos_embed = nn.Embedding(self.config.num_position_embeddings, self.config.hidden_size,
                                             dtype=torch.bfloat16)
                self.num_grid_per_side = int(self.config.num_position_embeddings**0.5)

        self.visual = ExecutedNativeVisionFront()
        method_sha = hashlib.sha256(inspect.getsource(Qwen3VLVisionModel.fast_pos_embed_interpolate).encode()).hexdigest()
        if method_sha != self.binding['vision_position_method_sha256']:
            raise ValueError('Original position method differs')
        visual_state = {}; language = None; self.tensor_checks = []
        with safe_open(str(self.model_path/'model.safetensors'), framework='pt', device='cpu') as handle:
            keys = list(handle.keys())
            for item in self.binding['matched_front_tensors']:
                key = native_key(keys, item['component'], item['name'])
                value = handle.get_tensor(key)
                actual = front._tensor_descriptor(item['name'], value)
                if any(actual[k] != item[k] for k in actual): raise ValueError('Bound original component differs: '+key)
                self.tensor_checks.append(dict(key=key, **actual))
                if item['component'] == 'vision': visual_state[item['name']] = value
                else: language = value
        if language is None: raise ValueError('Original language embedding missing')
        self.visual.load_state_dict(visual_state, strict=True)
        self.visual.to(self.device).eval().requires_grad_(False)
        self.embedding = nn.Embedding.from_pretrained(language, freeze=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        self.stem = None
        self.loaded_transformer_layers = 0

    def load_spatial_stem(self):
        import torch
        from experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.modeling import FrameStem
        if sha(STEM) != STEM_SHA: raise ValueError('Wrong original 192-channel FrameStem source')
        saved = torch.load(STEM, map_location='cpu', weights_only=False)
        state = saved.get('state_dict', saved.get('model', saved))
        selected = {k[len('frame_stem.'):]: v for k,v in state.items() if k.startswith('frame_stem.')}
        self.stem = FrameStem(192)
        self.stem.load_state_dict(selected, strict=True)
        if sum(p.numel() for p in self.stem.parameters()) != 339312:
            raise ValueError('Original FrameStem parameter count differs')
        self.stem.to(self.device).eval().requires_grad_(False)

    def build(self, rows, target, meter=None):
        import torch
        from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as front
        from experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.data import decode_four_frames
        from stage_meter import StageMeter
        _, _, grid, count = PINS[target]
        if len(rows) != count: raise ValueError('One original physical field required')
        local_meter = meter if meter is not None else StageMeter()
        restore_start = len(local_meter._restore)
        # Call the original producer itself, including its B1 quantization and
        # CPU return. Scoped wrappers change observation only, not producer math.
        if meter is not None:
            meter.wrap(front, 'decode_frames', 'frontend.Qwen_decode65_448', family='frontend_cpu', gpu=False)
            meter.wrap(front, 'quality_tokens_from_images', 'frontend.quality14', family='frontend_cpu', gpu=False)
            meter.wrap(front, 'pool_qwen_front_tokens', 'frontend.native_mean_pool')
            meter.wrap(self.visual.patch_embed, 'forward', 'frontend.native_patch')
            meter.wrap(self.visual, 'fast_pos_embed_interpolate', 'frontend.native_position')
        # Special-method lookup bypasses an instance __call__ assignment. Use an
        # ordinary callable proxy so the actual processor span is always seen.
        def processor(*args, **kwargs):
            with (meter.scope('frontend.processor', 'frontend_cpu', True, False) if meter else nullcontext()):
                return self.processor(*args, **kwargs)
        try:
            vision, quality14 = front._extract_vision_rows(rows=rows, processor=processor, visual=self.visual,
                device=self.device, batch_size=1, frame_count=4, token_grid=grid)
            with (meter.scope('frontend.UTF8_tokenizer', 'frontend_cpu', True, False) if meter else nullcontext()):
                rendered = front.render_prompt(self.tokenizer, target_name=target, prompt=front.TARGET_PROMPTS[target])
                tokens = self.tokenizer([rendered], padding=True, return_tensors='pt', add_special_tokens=False)
            with (meter.scope('frontend.language_embedding', 'frontend_neural') if meter else nullcontext()):
                language = self.embedding(tokens['input_ids'].to(self.device)).detach().cpu().half().contiguous()
            expected = next(r for r in self.binding['matched_prompt_checks'] if r['target'] == target)
            if (tokens['input_ids'].tolist() != expected['input_ids'] or
                    tokens['attention_mask'].tolist() != expected['attention_mask']):
                raise ValueError('Original task prompt/tokenizer differs')
            batch = dict(vision_tokens=vision, quality_tokens=quality14, language_tokens=language,
                         language_mask=tokens['attention_mask'].bool())
            if target == 'spatial':
                if self.stem is None: raise ValueError('Original FrameStem must be loaded outside inference')
                with (meter.scope('frontend.raw100_224_decode', 'frontend_cpu', True, False) if meter else nullcontext()):
                    raw = torch.stack([decode_four_frames(Path(row.video_path), size=224, crop_fraction=1.0) for row in rows])
                with (meter.scope('frontend.FrameStem192', 'frontend_neural') if meter else nullcontext()):
                    quality = self.stem(raw.to(self.device)).detach().cpu().half().contiguous()
                if tuple(quality.shape) != (1,4,196,192): raise ValueError('Wrong original quality stem output')
                batch.update(raw_frames=raw, quality_tokens=quality)
            else:
                batch['vision_tokens'] = vision.unsqueeze(0)
                batch['quality_tokens'] = quality14.unsqueeze(0)
            return batch, dict(rendered_prompt=rendered, input_ids=tokens['input_ids'].tolist(),
                attention_mask=tokens['attention_mask'].tolist(), source_ids=[r.sample_id for r in rows])
        finally:
            # Restore only these temporary producer observations, retaining any
            # body observations installed on the same meter by the caller.
            if meter is not None:
                restore_since(local_meter, restore_start)


def instrument_body(model, meter):
    import torch
    # Maximal unchanged neural subgraphs, with parallel electronic routes kept
    # as actual-but-uncharged. No scope covers an FFT or field layout as NN.
    covered = set()
    for name, module in list(model.named_modules()):
        if not name or any(name == root or name.startswith(root+'.') for root in covered): continue
        root = name.split('.')[0]
        whole = (root in {'vision_adapter','quality_adapter','visual_input_norm','language_adapter',
                         'prompt_to_visual','frame_merger','readout','quality_refiner'}
                 or (root in {'vision_routes','language_routes'} and name.count('.') == 1)
                 or residual_name(name) and name == root
                 or module.__class__.__name__ == 'CcdReadout')
        if isinstance(module, (torch.nn.ModuleList, torch.nn.Identity)): continue
        if whole:
            meter.wrap(module, 'forward', name, charged=not residual_name(name))
            covered.add(name)
    for name, module in list(model.named_modules()):
        if not name or any(name == root or name.startswith(root+'.') for root in covered): continue
        if isinstance(module, (torch.nn.Linear,torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d,
                               torch.nn.LayerNorm,torch.nn.BatchNorm1d,torch.nn.BatchNorm2d)):
            meter.wrap(module, 'forward', name, charged=not residual_name(name)); covered.add(name)
    for owner_name in ('parallel_router','parallel_optics','serial_router','serial_optics'):
        owner = getattr(model, owner_name)
        meter.wrap(owner.propagation, 'forward', owner_name+'.propagation', family='optical_simulation_FFT', charged=False)
    for index, fusion in enumerate(model.fusions):
        meter.wrap(fusion, 'forward', f'fusions.{index}', family='pure_fusion', charged=False)
    return covered


def tensor_record(value):
    import torch
    return dict(shape=list(value.shape), dtype=str(value.dtype),
        sha256=hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def equal_tree(left, right, path=''):
    import torch
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left,right):
            raise ValueError('Original/observed tensor differs: '+path)
    elif isinstance(left, dict):
        if not isinstance(right,dict) or left.keys()!=right.keys(): raise ValueError('Output keys differ: '+path)
        for key in left: equal_tree(left[key],right[key],path+'/'+str(key))
    elif isinstance(left,(list,tuple)):
        if type(left)!=type(right) or len(left)!=len(right): raise ValueError('Output structure differs: '+path)
        for i,(a,b) in enumerate(zip(left,right)): equal_tree(a,b,path+'/'+str(i))
    elif left != right: raise ValueError('Output value differs: '+path)


def contract(output):
    import torch
    from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import load_model, forward
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import read_manifest
    from stage_meter import StageMeter
    if torch.cuda.is_initialized(): raise ValueError('CPU proof cannot inherit CUDA')
    torch.set_num_threads(8)
    output.mkdir(parents=True, exist_ok=False)
    frozen = sources(); checks=[]
    if sha(MANIFEST)!=MANIFEST_SHA: raise ValueError('Wrong original manifest')
    tests = [r for r in read_manifest(MANIFEST) if r.split=='test']
    if len(tests)!=558: raise ValueError('Original full TEST identity required, not reevaluation')
    write(output/'status.json',dict(status='loading_original_components',pid=__import__('os').getpid()))
    live=LiveFront('cpu'); live.load_spatial_stem()
    with torch.inference_mode():
        for target in ('temporal','spatial'):
            run, pin, grid, count=PINS[target]
            model, settings=load_model(target,TASK/'runs/simulation'/run/'best_checkpoint.pt','cpu')
            if target=='spatial' and (settings.quality_branch_enabled or not settings.electronic_quality_residual_enabled
                    or not settings.custom_conv_electronic_enabled or model.frame_stem is not None):
                raise ValueError('Original .6710 auxiliary/residual architecture differs')
            for field_index in ((13,14) if target=='temporal' else (200,557)):
                indices, valid=field_indices(558,count,field_index); rows=[tests[i] for i in indices]
                write(output/'status.json',dict(status='real_original_CPU_contract',target=target,field_index=field_index))
                gold, meta=live.build(rows,target)
                original=forward(model,gold)
                meter=StageMeter(); meter.sample_index=field_index; meter.valid_samples=sum(valid)
                observed, observed_meta=live.build(rows,target,meter)
                equal_tree(gold,observed,'inputs'); equal_tree(meta,observed_meta,'input_metadata')
                covered=instrument_body(model,meter)
                shapes=[]; handles=[]
                for name,module in model.named_modules():
                    if isinstance(module,(torch.nn.Linear,torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)):
                        def hook(owner,args,result,name=name):
                            shapes.append(dict(name=name,type=type(owner).__name__,input_shape=list(args[0].shape),
                                output_shape=list(result.shape),weight_shape=list(owner.weight.shape),
                                charged_serial=not residual_name(name)))
                        handles.append(module.register_forward_hook(hook))
                try: measured=forward(model,observed); equal_tree(original,measured,'full_original_output')
                finally:
                    for handle in handles: handle.remove()
                    meter.close()
                passes=[r for r in meter.rows if r['family']=='optical_simulation_FFT']
                if len(passes)!=6 or not any(r['stage']=='frame_merger' for r in meter.rows):
                    raise ValueError('Original six-pass/full-merger observation missing')
                if any(r['cuda_event_ms'] is not None for r in meter.rows): raise ValueError('CPU proof used CUDA')
                checks.append(dict(target=target,body_sha256=pin,field_index=field_index,test_indices=indices,
                    valid=valid,source_ids=meta['source_ids'],bitwise_equal=True,inputs={k:tensor_record(v) for k,v in gold.items()},
                    prediction=tensor_record(original['prediction']),six_passes=[r['stage'] for r in passes],
                    full_original_output_bitwise_equal=True,selected_NN_stage_instrumentation_tested=True,
                    stage_records=meter.rows,source_ledger=meter.ledger,nn_module_shapes=shapes,
                    covered_neural_modules=sorted(covered),frontend_transformer_layers_executed=0))
            del model
    if sources()!=frozen: raise ValueError('Original sources changed during CPU proof')
    proof=dict(status='passed',no_gpu=True,cuda_initialized=torch.cuda.is_initialized(),checks=checks,sources=frozen,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        original_front_parameter_binding_sha256=BINDING_SHA,manifest_sha256=MANIFEST_SHA,
        formal_timing_or_full_TEST_accuracy_evaluation=False,
        original_cpu_producer_vs_scoped_live_input_and_full_original_forward=True,
        historical_GPU_cache_bitwise_equivalence_not_claimed=True,created_unix=time.time())
    write(output/'contract.json',proof)
    write(output/'status.json',dict(status='complete',contract_sha256=sha(output/'contract.json')))
    print(json.dumps(dict(status='passed',real_fields=len(checks),no_gpu=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    contract(parser.parse_args().output.resolve())
