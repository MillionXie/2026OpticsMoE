import argparse
from dataclasses import replace
from pathlib import Path
import torch
from torch.utils.data import Subset
from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification import run as original
from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.settings import save_resolved_config
from .settings import load_settings
from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.data import prepare_cifar10
from .backbone import load_backbone
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import seed_everything
from .data import prepare,DATA_ROOT,atomic_json
from .engine import train
from . import model as M
from .protocol import require_authorization,require_gpu_preflight,output_root,RoutingProtocolFailure

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(Path(__file__).with_name('config.yaml')))
    p.add_argument('--architecture',choices=['moe','d2nn'],default='moe')
    p.add_argument('--task',choices=['A','B'],default='A')
    p.add_argument('--variant',choices=['reserved','all','d2nn'],default='reserved')
    p.add_argument('--source',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--prepare-data',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();require_authorization()
    if not a.smoke and not a.prepare_data:require_gpu_preflight()
    if Path(a.config).resolve()!=Path(__file__).with_name('config.yaml').resolve():raise RuntimeError('Use the reviewed Vision-only entry configuration')
    s=load_settings(a.config)
    name=a.architecture+'_A' if a.task=='A' else a.architecture+'_'+a.variant+'_B'
    s.output_dir=output_root(a.smoke)/name
    s.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);seed_everything(42)
    s.router_optimization_seed=42;s.run_final_test=False
    if a.smoke:
        s.max_train_steps_per_epoch=2;s.batch_size=20;s.pk_images_per_sku=2;s.inference_batch_size=20;s.num_workers=0
    save_resolved_config(s)
    bundle=prepare_cifar10(s,persist=True)
    if a.prepare_data:
        prepare(bundle,a.smoke);return
    data_root=Path(str(DATA_ROOT)+('_smoke' if a.smoke else ''))
    if a.task=='B' and not (data_root/'manifest.json').exists():raise RuntimeError('B data preparation incomplete')
    if a.smoke:bundle=replace(bundle,validation=Subset(bundle.validation,list(range(20))))
    loaded=load_backbone(s,original._device(s))
    seed_everything(42)
    r,h=M.build(loaded,s,a.architecture)
    save_resolved_config(s)
    try:
        atomic_json(s.output_dir/'environment.json',original.environment_report())
        # Architecture construction and diagnostic RNG cannot change the sample stream.
        seed_everything(1442)
        train(loaded,r,h,bundle,s,data_root,a.task,a.variant,a.source,a.smoke,a.resume)
    except RoutingProtocolFailure as e:
        atomic_json(s.output_dir/'failure.json',dict(status='routing_failed',exception=repr(e)))
        raise SystemExit(2)
    except BaseException as e:
        atomic_json(s.output_dir/'failure.json',dict(status='failed',exception=repr(e)))
        raise
    finally:r.close()

if __name__=='__main__':main()
