import ast
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import t12_baseline_live as m


def test_gold_is_unmodified_original_formal_math_not_old_timing_bypass():
    statements=m.original_statements()
    text='\n'.join(ast.unparse(x) for x in statements)
    assert len(statements)==7
    assert 'qwen(**tokens, use_cache=False, return_dict=True)' in text
    assert 'vae.encode(ref.half()).latent_dist.mode()' in text
    assert 'noise_batch(len(ref), device, start, args.seed)' in text
    assert 'vae.decode' in text and 'target' not in text
    assert 'Identity' not in text and 'bypass' not in text


def test_adopted_matched_checkpoint_and_full_layer_contract():
    assert m.BASELINE_SHA=='599805ad3062bebf67acf3b515f0a812fb843506643e18251004f180131b4b52'
    tree=ast.parse(Path(m.__file__).read_text())
    load=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_runtime')
    text=ast.unparse(load)
    assert 'keep_layers=28' in text and "payload['unet']" in text
    assert 'train()' not in text and 'requires_grad_(False)' in text


def test_no_prompt_condition_reuse_or_gt_input_in_online_forward():
    tree=ast.parse(Path(m.__file__).read_text())
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    text=ast.unparse(f)
    assert 'conditions[' not in text and 'target' not in text
    assert 'runtime.qwen(**tokens' in text
    assert 'latent_dist.mode()' in text and '1042' in text


def test_original_full_VAE_and_scheduler_asset_pins_not_current_path_guess():
    records=m.dependency_records();by_name={r['relative_path']:r for r in records}
    assert len(records)==5
    assert by_name['models/sd-turbo-fp16/vae/diffusion_pytorch_model.fp16.safetensors']['sha256']=='3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a'
    assert by_name['models/sd-turbo-fp16/scheduler/scheduler_config.json']['sha256']=='5599066c725fc2747a201fa3165ad7c8c3de3be629ce325451752e8f94c0d1ae'
