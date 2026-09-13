"""Real eight-image, three-stage update check; not an accuracy experiment."""
import json
from pathlib import Path
import torch

from .settings import load_settings
from .modeling import load_vision_backbone, build_student, initialize_student, optimizer, architecture_report
from .training_support import ModelEMA, TrainTeacherMaps
from .alternating_training import AlternatingSchedule
from .sam_training import train_sam_epoch
from .training import legacy
from .run import _seed
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import prepare_salicon


def main():
    s = load_settings(Path(__file__).parent/'configs/moe_alpha40_alternating_20260914.yaml')
    s.output_dir = Path(__file__).parent/'runs/smoke/alternating_20260914'
    if (s.output_dir/'report.json').exists():
        raise FileExistsError('Do not overwrite previous preflight')
    s.output_dir.mkdir(parents=True, exist_ok=True)
    _seed(42)
    bundle = prepare_salicon(s, persist=False)
    loaded = load_vision_backbone(s, torch.device('cuda:0'))
    s.resolve_architecture(loaded.model)
    model = build_student(loaded, s)
    initialize_student(model,s)
    train_loader, _ = legacy.build_loaders(bundle,s,training=True)
    train_loader.num_workers = 0
    train_loader.persistent_workers = False
    batch = next(iter(train_loader))
    assert len(batch['sample_ids']) == 8
    opt = optimizer(model,s)
    frozen = [(p,p.detach().clone()) for p in model.parameters() if not p.requires_grad]
    ema = ModelEMA(model,s.ema_decay)
    hook = opt.register_step_post_hook(ema.update)
    schedule = AlternatingSchedule(opt,s,ema)
    teacher = TrainTeacherMaps(s,bundle.train_records)
    native_calls = []
    handles = [block.register_forward_hook(lambda *args: native_calls.append(1))
               for block in loaded.model.visual.blocks] if hasattr(loaded.model,'visual') else []
    rows = []
    s.map_kd_weight = s.distillation_initial_weight
    try:
        for epoch in (1,11,26):
            row = schedule.begin_epoch(epoch)
            s.alternating_stage = row['stage']
            model._router_hard_weight = row['hard_balance_weight']
            model.core.set_phase_dropout_active(True)
            metrics = train_sam_epoch(model,[batch],loaded,s,opt,teacher)
            row.update(schedule.end_epoch())
            assert all(torch.equal(p,old) for p,old in frozen)
            for group in opt.param_groups:
                changed = row['epoch_raw_update_rms_' + group['name']] > 0
                assert changed == any(p.requires_grad for p in group['params']), group['name']
            row.update(train_metrics=metrics, original_frozen_parameters_verified=sum(p.numel() for p,_ in frozen))
            rows.append(row)
        assert not native_calls
        report = {'scope': 'three single updates on the SAME eight TRAIN images, NOT test CC',
                  'sample_ids': batch['sample_ids'], 'stages': rows,
                  'architecture': architecture_report(model,s), 'native_transformer_calls': len(native_calls),
                  'native_hook_count': len(handles), 'inference_parameters_added': 0}
        (s.output_dir/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    finally:
        for handle in handles:
            handle.remove()
        hook.remove()
        model.core.set_phase_dropout_active(False)
        model.restore_native()


if __name__ == '__main__':
    main()
