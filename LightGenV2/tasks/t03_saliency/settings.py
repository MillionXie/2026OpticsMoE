"""SALICON task settings layered on the audited T02 optical contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.settings import (
    _nested,
    _read_config,
)
from LightGenV2.tasks.t02_keypoint_detection.settings import (
    load_settings as load_t02_settings,
    save_resolved_config as save_t02_resolved_config,
)


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return (path if path.is_absolute() else base / path).resolve()


def load_settings(path: str | Path) -> Any:
    config = Path(path).expanduser().resolve()
    settings = load_t02_settings(config)
    raw = _read_config(config)
    d = lambda key, default=None: _nested(raw, key, default)
    settings.router_phase_coordinates = str(d('training.router_phase_coordinates','sigmoid'))
    settings.convert_router_phase_on_warmstart = bool(d('training.convert_router_phase_on_warmstart',False))

    # The shared server keeps Hugging Face snapshots beside the repository,
    # while portable configs point at a repository-local cache.  Resolve the
    # former only when the configured cache is absent; this remains offline.
    adjacent_hf_cache = (config.parents[5] / ".cache" / "huggingface" / "hub"
                         if len(config.parents) > 5 else None)
    if (settings.cache_dir is not None and not settings.cache_dir.exists()
            and adjacent_hf_cache is not None and adjacent_hf_cache.is_dir()):
        settings.cache_dir = adjacent_hf_cache.resolve()

    settings.validation_limit = d("dataset.validation_limit")
    if settings.validation_limit is not None:
        settings.validation_limit = int(settings.validation_limit)
    settings.materialize_density_maps = bool(
        d("dataset.materialize_density_maps", True)
    )
    settings.density_sigma_px = float(d("dataset.density_sigma_px", 19.0))
    settings.train_images_url = str(
        d("dataset.train_images_url", "https://s3.amazonaws.com/salicon-dataset/2015r1/train.zip")
    )
    settings.validation_images_url = str(
        d("dataset.validation_images_url", "https://s3.amazonaws.com/salicon-dataset/2015r1/val.zip")
    )
    settings.train_annotations_url = str(
        d("dataset.train_annotations_url", "https://s3.amazonaws.com/salicon-dataset/2015r1/fixations_train2014.json")
    )
    settings.validation_annotations_url = str(
        d("dataset.validation_annotations_url", "https://s3.amazonaws.com/salicon-dataset/2015r1/fixations_val2014.json")
    )
    settings.artifact_cache_dir = _resolve(
        d("artifact_cache_dir", "../../../../cache/qwen3_vl_embedding_2b_salicon_lightgen"),
        config.parent,
    )
    settings.augmentation_enabled = bool(d("augmentation.enabled", True))
    settings.augmentation_mode = str(d("augmentation.mode", "legacy"))
    if settings.augmentation_mode not in {"legacy", "aligned_flip", "aligned_weak"}:
        raise ValueError("Unknown T03 augmentation mode")
    settings.crop_scale_min = float(d("augmentation.crop_scale_min", 0.90))
    settings.horizontal_flip_probability = float(
        d("augmentation.horizontal_flip_probability", 0.5)
    )
    settings.brightness_jitter = float(d("augmentation.brightness_jitter", 0.10))
    settings.contrast_jitter = float(d("augmentation.contrast_jitter", 0.10))
    settings.augmentation_end_epoch = int(d("augmentation.end_epoch", 0))
    settings.augmentation_apply_probability = float(d("augmentation.apply_probability", 1.0))
    if not 0 <= settings.augmentation_apply_probability <= 1:
        raise ValueError("augmentation.apply_probability must be in [0,1]")
    if settings.augmentation_apply_probability != 1 and settings.augmentation_mode != "aligned_weak":
        raise ValueError("Partial augmentation is supported only for aligned_weak")
    if settings.augmentation_mode == "aligned_weak" and (
        not 0.8 <= settings.crop_scale_min <= 1 or not 0 <= settings.brightness_jitter <= .15
        or not 0 <= settings.contrast_jitter <= .15 or settings.augmentation_end_epoch < 0
    ):
        raise ValueError("aligned_weak requires bounded crop/jitter and nonnegative end epoch")
    if not 0 <= settings.horizontal_flip_probability <= 1:
        raise ValueError("Invalid horizontal flip probability")
    if settings.augmentation_mode == "aligned_flip" and (
        settings.crop_scale_min != 1.0 or settings.brightness_jitter != 0.0 or settings.contrast_jitter != 0.0
    ):
        raise ValueError("aligned_flip forbids crop/photometric jitter; teacher maps must remain aligned")
    settings.kl_weight = float(d("loss.kl_weight", 1.0))
    settings.cc_weight = float(d("loss.cc_weight", 0.5))
    settings.sim_weight = float(d("loss.sim_weight", 0.25))
    settings.nss_weight = float(d("loss.nss_weight", 0.1))
    from .pyramid_supervision import validate as validate_pyramid_cc
    settings.pyramid_cc = validate_pyramid_cc(d('loss.pyramid_cc', {}), settings.image_size)
    settings.map_kd_weight = 0.0
    settings.sam_rho = float(d("training.sam_rho", 0.0))
    settings.exact_fusion_backward = bool(d("training.exact_fusion_backward", False))
    if not 0 <= settings.sam_rho <= .1:
        raise ValueError("training.sam_rho must be in [0,.1]")
    settings.map_kd_temperature = 1.0
    settings.distillation_initial_weight = float(d("distillation.initial_weight", 0.0))
    settings.distillation_final_weight = float(d("distillation.final_weight", 0.0))
    settings.distillation_end_epoch = int(d("distillation.end_epoch", 50))
    teacher_only_epochs = d("distillation.teacher_only_epochs", 0)
    if isinstance(teacher_only_epochs, bool) or not isinstance(teacher_only_epochs, int):
        raise ValueError("teacher_only_epochs must be an integer")
    settings.teacher_only_epochs = teacher_only_epochs
    settings.unlabeled_weight = float(d('unlabeled_distillation.weight', 0.0))
    settings.semantic_weight = float(d('semantic_auxiliary.weight', 0.0))
    settings.semantic_mode = str(d('semantic_auxiliary.mode', 'image_presence'))
    settings.semantic_learning_rate = float(d('semantic_auxiliary.learning_rate', .001))
    semantic_path = d('semantic_auxiliary.targets_file')
    settings.semantic_targets = _resolve(semantic_path,config.parent) if semantic_path else None
    settings.semantic_targets_sha256 = d('semantic_auxiliary.targets_sha256')
    settings.unlabeled_image_manifest = d('unlabeled_distillation.image_manifest')
    settings.unlabeled_image_manifest_sha256 = d('unlabeled_distillation.image_manifest_sha256')
    settings.unlabeled_cache = d('unlabeled_distillation.cache_file')
    settings.unlabeled_cache_sha256 = d('unlabeled_distillation.cache_sha256')
    for name in ('unlabeled_image_manifest','unlabeled_cache'):
        value = getattr(settings,name)
        setattr(settings,name,_resolve(value,config.parent) if value else None)
    cache = d("distillation.cache_file")
    settings.distillation_cache = _resolve(cache, config.parent) if cache else None
    settings.distillation_teacher_sha256 = d("distillation.teacher_sha256")
    settings.distillation_loss = str(d("distillation.loss", "kl"))
    if settings.distillation_loss not in ('kl','spatial_cc'):
        raise ValueError('Unknown distillation loss')
    if settings.distillation_loss == 'spatial_cc' and (
        not settings.sam_rho or not settings.distillation_initial_weight):
        raise ValueError('Spatial CC distillation currently requires the task-local SAM training path and active teacher')
    settings.feature_hint_initial_weight = float(d("feature_hint.initial_weight",0.0))
    settings.feature_hint_final_weight = float(d("feature_hint.final_weight",0.0))
    settings.feature_hint_end_epoch = int(d("feature_hint.end_epoch",30))
    settings.feature_hint_learning_rate = float(d("feature_hint.learning_rate",0.0002))
    settings.feature_hint_loss_mode = str(d('feature_hint.loss_mode','cosine'))
    if settings.feature_hint_loss_mode not in {'cosine','spatial_centered_cosine'}:
        raise ValueError('Unknown feature hint loss mode')
    hint_cache = d("feature_hint.cache_file")
    settings.feature_hint_cache = _resolve(hint_cache,config.parent) if hint_cache else None
    if not 0 <= settings.feature_hint_final_weight <= settings.feature_hint_initial_weight < float('inf'):
        raise ValueError("Invalid feature hint weights")
    if settings.feature_hint_initial_weight > 0 and (
        settings.augmentation_enabled or not settings.feature_hint_cache or not settings.distillation_teacher_sha256
        or settings.feature_hint_end_epoch < 2 or not 0 < settings.feature_hint_learning_rate < float('inf')
    ):
        raise ValueError("Feature hints require unaugmented train cache, teacher SHA, and valid schedule")
    settings.ema_decay = float(d("training.ema_decay", 0.0))
    settings.phase_weight_decay = float(d("training.phase_weight_decay", settings.weight_decay))
    if not 0 <= settings.ema_decay < 1 or not 0 <= settings.distillation_final_weight <= settings.distillation_initial_weight:
        raise ValueError("Invalid EMA/KD coefficient")
    if settings.distillation_initial_weight > 0 and (
        (settings.augmentation_enabled and settings.augmentation_mode not in {"aligned_flip", "aligned_weak"}) or settings.distillation_cache is None
        or not settings.distillation_teacher_sha256 or settings.distillation_end_epoch < 2
    ):
        raise ValueError("KD requires aligned inputs (none/aligned_flip/aligned_weak), cache, teacher SHA and valid end epoch")
    settings.teacher_checkpoint = None
    settings.ccd_normalization = str(d("lightgen.ccd_normalization", "historical_log1p"))
    if settings.ccd_normalization not in {"historical_log1p", "mean_only"}:
        raise ValueError("Unknown T03 CCD normalization")
    warmstart = d("training.initialization_checkpoint")
    settings.initialization_checkpoint = _resolve(warmstart, config.parent) if warmstart else None
    settings.initialization_checkpoint_sha256 = d("training.initialization_checkpoint_sha256")
    settings.reset_fusion_on_warmstart = bool(d("training.reset_fusion_on_warmstart", False))
    settings.electronic_spatial_kernel_size = int(d("lightgen.electronic_spatial_kernel_size", 3))
    settings.expand_kernel_on_warmstart = bool(d("training.expand_kernel_on_warmstart", False))
    settings.electronic_grn = bool(d("lightgen.electronic_grn", False))
    settings.initialize_grn_on_warmstart = bool(d("training.initialize_grn_on_warmstart", False))
    settings.electronic_ffn_spatial_dilation = int(d("lightgen.electronic_ffn_spatial_dilation", 0))
    settings.initialize_ffn_on_warmstart = bool(d("training.initialize_ffn_on_warmstart", False))
    settings.ffn_spatial_learning_rate = float(d("training.ffn_spatial_learning_rate", 0.0001))
    settings.electronic_global_rank = int(d("lightgen.electronic_global_rank", 0))
    settings.initialize_global_on_warmstart = bool(d("training.initialize_global_on_warmstart", False))
    settings.electronic_ffn_hidden_width = int(d("lightgen.electronic_ffn_hidden_width", 384))
    settings.widen_ffn_on_warmstart = bool(d("training.widen_ffn_on_warmstart", False))
    settings.electronic_ffn_groups = int(d("lightgen.electronic_ffn_groups", 0))
    settings.expand_ffn_groups_on_warmstart = bool(d("training.expand_ffn_groups_on_warmstart", False))
    if settings.electronic_ffn_groups not in (0,64):
        raise ValueError('FFN groups must be 0(original depthwise) or audited64')
    if settings.expand_ffn_groups_on_warmstart and settings.electronic_ffn_groups != 64:
        raise ValueError('Grouped transfer requires groups64')
    if settings.electronic_ffn_groups == 64:
        if (settings.electronic_ffn_hidden_width != 384 or settings.electronic_ffn_spatial_dilation != 1
                or settings.electronic_global_rank or settings.electronic_grn
                or settings.electronic_spatial_kernel_size != 3 or settings.initialize_ffn_on_warmstart
                or settings.initialize_global_on_warmstart or settings.widen_ffn_on_warmstart
                or settings.reset_fusion_on_warmstart or settings.expand_kernel_on_warmstart
                or settings.initialize_grn_on_warmstart):
            raise ValueError('Isolate grouped64 to the existing384 CFFN1; no other transfer')
        if not settings.initialization_checkpoint or not settings.initialization_checkpoint_sha256:
            raise ValueError('Grouped FFN requires SHA-pinned warmstart')
    if settings.electronic_ffn_hidden_width not in (384,576):
        raise ValueError("Only original384 or widened576 FFN hidden width is audited")
    if settings.widen_ffn_on_warmstart and settings.electronic_ffn_hidden_width != 576:
        raise ValueError("FFN widening transfer requires target576")
    if settings.electronic_ffn_hidden_width == 576:
        if (settings.electronic_ffn_spatial_dilation != 1 or settings.electronic_global_rank
                or settings.electronic_grn or settings.electronic_spatial_kernel_size != 3
                or settings.initialize_ffn_on_warmstart or settings.reset_fusion_on_warmstart
                or settings.initialize_global_on_warmstart or settings.expand_kernel_on_warmstart
                or settings.initialize_grn_on_warmstart):
            raise ValueError("Isolate widening to an existing CFFN1; no other architecture/fusion transfer")
        if not settings.initialization_checkpoint or not settings.initialization_checkpoint_sha256:
            raise ValueError("Widened FFN requires SHA-pinned warmstart")
    if settings.sam_rho and settings.feature_hint_initial_weight:
        raise ValueError("Isolate SAM from training-only feature hints")
    settings.global_spatial_learning_rate = float(d("training.global_spatial_learning_rate", 0.0002))
    if settings.electronic_global_rank not in (0,16,64):
        raise ValueError("Global spatial rank must be 0(off) or audited rank16/64")
    if settings.initialize_global_on_warmstart and not settings.electronic_global_rank:
        raise ValueError("Global transfer requires enabled module")
    if settings.electronic_global_rank:
        if settings.electronic_grn or settings.electronic_spatial_kernel_size != 3:
            raise ValueError("Isolate global spatial mixing from GRN/large kernels")
        if not settings.initialization_checkpoint or settings.electronic_width != 192:
            raise ValueError("Global spatial mixing requires width192 warmstarted SALICON")
        if not 0 < settings.global_spatial_learning_rate < float('inf'):
            raise ValueError("Global spatial learning rate must be finite and positive")
    if settings.electronic_ffn_spatial_dilation not in (0, 1, 2):
        raise ValueError("Spatial FFN dilation must be 0(off), 1, or 2")
    if settings.initialize_ffn_on_warmstart and not settings.electronic_ffn_spatial_dilation:
        raise ValueError("Spatial FFN transfer requires enabled module")
    if settings.electronic_ffn_spatial_dilation:
        if settings.electronic_grn or settings.electronic_spatial_kernel_size != 3:
            raise ValueError("Isolate spatial FFN from GRN/expanded token kernels")
        if not settings.initialization_checkpoint or settings.electronic_width != 192 or settings.electronic_expansion != 2:
            raise ValueError("Spatial FFN requires audited 192->384 warmstarted SALICON")
        if not 0 < settings.ffn_spatial_learning_rate < float('inf'):
            raise ValueError("Spatial FFN learning rate must be finite and positive")
    if settings.initialize_grn_on_warmstart and not settings.electronic_grn:
        raise ValueError("GRN identity transfer requires electronic_grn")
    if settings.electronic_spatial_kernel_size not in {3, 5, 13}:
        raise ValueError("T03 supports audited electronic spatial kernels 3, 5, or 13")
    if settings.expand_kernel_on_warmstart and settings.electronic_spatial_kernel_size not in (5,13):
        raise ValueError("Kernel expansion requires target kernel 5 or 13")
    # Legacy Vision2 overwrites training.* rates with optimization.* defaults.
    # Keep historical profiles reproducible; new profiles explicitly opt in.
    settings.learning_rate_source = str(d("training.learning_rate_source", "legacy_optimization"))
    if settings.learning_rate_source not in {"legacy_optimization", "task_training"}:
        raise ValueError("Unknown learning_rate_source")
    if settings.learning_rate_source == "task_training":
        for name in ("student_learning_rate", "phase_learning_rate", "router_learning_rate"):
            value = d(f"training.{name}")
            if value is None or not 0 < float(value) < float("inf"):
                raise ValueError(f"Explicit positive training.{name} required")
            setattr(settings, name, float(value))
    settings.staged_training = bool(d("training.staged.enabled", False))
    settings.staged_warmup_epochs = int(d("training.staged.warmup_epochs", 10))
    settings.staged_polish_start = int(d("training.staged.polish_start", 71))
    settings.staged_final_hard_balance = float(d("training.staged.final_hard_balance", 0.10))
    settings.staged_freeze_electronic_gradients = bool(d("training.staged.freeze_electronic_gradients", False))
    settings.dense_readout_learning_rate = float(d("training.dense_readout_learning_rate", settings.dense_readout_learning_rate))
    settings.dense_head_learning_rate = float(d("training.dense_head_learning_rate", settings.dense_head_learning_rate))
    settings.gradient_clip_norm = float(d("training.gradient_clip_norm", 1.0))
    settings.test_interval_epochs = int(d("protocol.test_interval_epochs", 5))
    settings.adaptive_plateau_enabled = bool(d("training.adaptive_plateau.enabled", False))
    settings.adaptive_plateau_options = {
        "patience": int(d("training.adaptive_plateau.patience", 3)),
        "min_delta": float(d("training.adaptive_plateau.min_delta", 0.0001)),
        "factor": float(d("training.adaptive_plateau.factor", 0.5)),
        "max_reductions": int(d("training.adaptive_plateau.max_reductions", 2)),
        "min_epoch": int(d("training.adaptive_plateau.min_epoch", 6)),
    }
    if settings.adaptive_plateau_enabled:
        from .plateau import PlateauController
        PlateauController(**settings.adaptive_plateau_options)
        if not settings.staged_training or settings.test_interval_epochs < 1:
            raise ValueError("Adaptive plateau requires staged training and periodic tests")
    settings.router_hard_load_balance_weight = float(
        d("loss.router_hard_load_balance_weight", 0.50)
    )
    settings.router_balance_estimator = d('loss.router_balance_estimator', 'batch')
    if settings.router_balance_estimator not in {'batch', 'cross_sample'}:
        raise ValueError('Unknown router balance estimator')
    if settings.router_balance_estimator == 'cross_sample' and (
        settings.student_batch_size < 2 or settings.router_balance_weight <= 0
        or settings.lightgen_model_variant != 'optical_router_scale_matched_moe'
        or not settings.sam_rho or settings.fusion_alpha_min < .4
    ):
        raise ValueError('Cross-sample probe requires SAM optical Top2 alpha>=.4 and batch>=2')
    settings.segmentation_projection_dim = int(d("saliency_head.projection_dim", 128))
    settings.segmentation_channels = tuple(
        int(value) for value in d("saliency_head.decoder_channels", [96, 64, 32, 16])
    )
    settings.segmentation_groupnorm_groups = int(
        d("saliency_head.groupnorm_groups", 8)
    )
    settings.visualization_sample_count = int(d("visualization.sample_count", 16))
    settings.visualization_optical_sample_count = int(
        d("visualization.optical_sample_count", 4)
    )
    if settings.router_backend != "optical" or settings.top_k != 2:
        raise ValueError("T03 main contract is fixed to optical Router Top-2")
    if settings.lightgen_model_variant == "optical_router_scale_matched_moe":
        if settings.router_hard_load_balance_weight <= 0:
            raise ValueError("T03 optical Router requires a positive hard-load loss")
    if not 0 <= settings.teacher_only_epochs < settings.student_epochs:
        raise ValueError("Teacher-only curriculum must leave at least one GT epoch")
    if settings.teacher_only_epochs and (
        not settings.sam_rho or settings.feature_hint_initial_weight
        or settings.distillation_loss != "spatial_cc"
        or not 0 < settings.distillation_final_weight <= settings.distillation_initial_weight < float('inf')
        or settings.adaptive_plateau_enabled
        or not any(getattr(settings, n) > 0 for n in ('kl_weight','cc_weight','sim_weight','nss_weight'))
    ):
        raise ValueError("Teacher-only curriculum requires SAM spatial-CC KD, positive teacher/GT, no hints or early stop")
    if not 0 <= settings.unlabeled_weight <= 2:
        raise ValueError('Unlabeled distillation weight must be in [0,2]')
    if settings.unlabeled_weight:
        if (not settings.sam_rho or settings.teacher_only_epochs or settings.feature_hint_initial_weight
                or settings.augmentation_enabled or settings.distillation_initial_weight <= 0
                or settings.lightgen_model_variant != 'optical_router_scale_matched_moe'
                or settings.fusion_alpha_min < .4 or settings.kl_weight <= 0 or settings.cc_weight <= 0):
            raise ValueError('Extra-image trial requires GT KL/CC, SAM, existing optical model and alpha>=.4; no hints/augmentation/teacher-only stage')
        if not settings.unlabeled_image_manifest or not settings.unlabeled_cache:
            raise ValueError('Extra-image trial requires image manifest and teacher cache')
        for name in ('unlabeled_image_manifest_sha256','unlabeled_cache_sha256'):
            value = getattr(settings,name)
            if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Extra-image manifest/cache must be SHA256-pinned')
    if not 0 <= settings.semantic_weight <= 2:
        raise ValueError('Semantic auxiliary weight must be in [0,2]')
    if settings.semantic_mode not in ('image_presence','box_coverage'):
        raise ValueError('Unsupported semantic auxiliary mode')
    if settings.semantic_weight:
        if settings.unlabeled_weight <= 0 or not settings.semantic_targets:
            raise ValueError('Semantic auxiliary requires the audited extra-image stream')
        value = settings.semantic_targets_sha256
        if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Semantic targets must be SHA256-pinned')
        if not 0 < settings.semantic_learning_rate <= .01 or settings.electronic_width != 192:
            raise ValueError('Invalid semantic auxiliary learning rate/feature width')
    if settings.router_phase_coordinates not in ('sigmoid','radians'):
        raise ValueError('Unsupported router phase coordinates')
    if settings.router_phase_coordinates=='radians':
        if settings.lightgen_model_variant!='optical_router_scale_matched_moe' or settings.phase_parameterization!='sigmoid':
            raise ValueError('Router radians trial requires optical MoE and unchanged sigmoid feature phases')
        if settings.phase_weight_decay!=0:
            raise ValueError('Direct periodic phase requires zero phase weight decay')
    if settings.convert_router_phase_on_warmstart:
        if settings.router_phase_coordinates!='radians' or not settings.initialization_checkpoint:
            raise ValueError('Router conversion requires radians warmstart')
        if any(getattr(settings,k,False) for k in ('initialize_ffn_on_warmstart','initialize_global_on_warmstart',
                'initialize_grn_on_warmstart','expand_kernel_on_warmstart','widen_ffn_on_warmstart',
                'expand_ffn_groups_on_warmstart','reset_fusion_on_warmstart')):
            raise ValueError('Do not combine router coordinate transfer with other warmstart transformations')
    from .feature_pretraining import configure as configure_feature_pretraining
    configure_feature_pretraining(settings, d('feature_pretraining', {}), config.parent)
    from .relational_distillation import configure as configure_relational
    configure_relational(settings, d('relational_distillation', {}), config.parent)
    from .masked_distillation import configure as configure_masked
    configure_masked(settings, d('masked_distillation', {}), config.parent)
    from .first_stage_supervision import configure as configure_first_stage
    configure_first_stage(settings, d('first_stage_supervision', {}))
    from .fixed_crop_training import configure as configure_fixed_crop
    configure_fixed_crop(settings, d('fixed_crop_distillation', {}), config.parent)
    from .hard_example_cc import configure as configure_hard_cc
    configure_hard_cc(settings, d('hard_example_cc', {}))
    from .teacher_reliability import configure as configure_teacher_reliability
    configure_teacher_reliability(settings, d('teacher_reliability', {}))
    settings.asam = dict(d('training.asam', {}) or {})
    if settings.asam:
        import math
        if (set(settings.asam) != {'rho', 'eta'}
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in settings.asam.values())
                or not .05 <= settings.asam['rho'] <= .5 or not .001 <= settings.asam['eta'] <= .1):
            raise ValueError('ASAM requires bounded finite rho and eta')
        if (settings.sam_rho <= 0 or settings.distillation_loss != 'spatial_cc'
                or settings.teacher_only_epochs or settings.augmentation_enabled
                or settings.fixed_crop_distillation or settings.hard_example_cc or settings.teacher_reliability
                or settings.first_stage_supervision or settings.masked_distillation or settings.relational_distillation
                or settings.feature_pretraining or settings.feature_hint_initial_weight
                or settings.unlabeled_weight or settings.semantic_weight
                or settings.fusion_alpha_min < .4 or settings.top_k != 2):
            raise ValueError('ASAM trial requires isolated GT+CC-KD optical training')
    if settings.pyramid_cc and (
        settings.distillation_loss != 'spatial_cc' or not settings.sam_rho
        or settings.teacher_only_epochs or settings.augmentation_enabled or settings.asam
        or settings.feature_hint_initial_weight or settings.unlabeled_weight or settings.semantic_weight
        or settings.hard_example_cc or settings.teacher_reliability
        or settings.fusion_alpha_min < .4 or settings.top_k != 2
    ):
        raise ValueError('Pyramid CC requires isolated SAM GT+CC-KD training with fixed optical contract')
    settings.gsam_coefficient = float(d('training.gsam_coefficient', 0.))
    if not 0 <= settings.gsam_coefficient <= .2:
        raise ValueError('GSAM coefficient must be finite and in [0,.2]')
    if settings.gsam_coefficient and (
        settings.sam_rho <= 0 or settings.asam or settings.pyramid_cc
        or settings.distillation_loss != 'spatial_cc' or settings.teacher_only_epochs
        or settings.augmentation_enabled or settings.fixed_crop_distillation
        or settings.hard_example_cc or settings.teacher_reliability
        or settings.first_stage_supervision or settings.masked_distillation or settings.relational_distillation
        or settings.feature_pretraining or settings.feature_hint_initial_weight
        or settings.unlabeled_weight or settings.semantic_weight
        or settings.fusion_alpha_min < .4 or settings.top_k != 2
        or settings.router_backend != 'optical'
    ):
        raise ValueError('GSAM requires isolated optical SAM GT+CC-KD training')
    settings.noise_consistency_weight = float(d('training.noise_consistency_weight', 0.))
    if not 0 <= settings.noise_consistency_weight <= 5:
        raise ValueError('Noise consistency weight must be finite and in [0,5]')
    if settings.noise_consistency_weight and (
        settings.sam_rho <= 0 or settings.asam or settings.gsam_coefficient or settings.pyramid_cc
        or settings.distillation_loss != 'spatial_cc' or settings.teacher_only_epochs
        or settings.augmentation_enabled or settings.fixed_crop_distillation
        or settings.hard_example_cc or settings.teacher_reliability
        or settings.first_stage_supervision or settings.masked_distillation or settings.relational_distillation
        or settings.feature_pretraining or settings.feature_hint_initial_weight
        or settings.unlabeled_weight or settings.semantic_weight
        or settings.fusion_alpha_min < .4 or settings.top_k != 2 or settings.router_backend != 'optical'
    ):
        raise ValueError('Noise consistency requires isolated optical SAM GT+CC-KD training')
    from .mixup_training import validate as validate_mixup
    settings.mixup = validate_mixup(d('training.mixup', {}))
    settings.mixup_active = False  # Enabled explicitly by the training epoch loop, never eval.
    if settings.mixup and (
        settings.sam_rho <= 0 or settings.asam or settings.gsam_coefficient or settings.noise_consistency_weight
        or settings.pyramid_cc or settings.distillation_loss != 'spatial_cc' or settings.teacher_only_epochs
        or settings.augmentation_enabled or settings.fixed_crop_distillation
        or settings.hard_example_cc or settings.teacher_reliability
        or settings.first_stage_supervision or settings.masked_distillation or settings.relational_distillation
        or settings.feature_pretraining or settings.feature_hint_initial_weight
        or settings.unlabeled_weight or settings.semantic_weight
        or settings.fusion_alpha_min < .4 or settings.top_k != 2 or settings.router_backend != 'optical'
        or settings.mixup['end_epoch'] >= settings.student_epochs
    ):
        raise ValueError('MixUp requires isolated optical SAM GT+CC-KD and a final unmixed stage')
    return settings


def save_resolved_config(settings: Any) -> None:
    save_t02_resolved_config(settings)
    import yaml
    path = settings.output_dir / "resolved_config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values.setdefault('training',{}).update(router_phase_coordinates=settings.router_phase_coordinates,
        convert_router_phase_on_warmstart=settings.convert_router_phase_on_warmstart)
    values['feature_pretraining'] = settings.feature_pretraining
    values['relational_distillation'] = settings.relational_distillation
    values['masked_distillation'] = settings.masked_distillation
    values['first_stage_supervision'] = settings.first_stage_supervision
    values['fixed_crop_distillation'] = settings.fixed_crop_distillation
    values['hard_example_cc'] = settings.hard_example_cc
    values['teacher_reliability'] = settings.teacher_reliability
    values.setdefault('loss', {})['pyramid_cc'] = settings.pyramid_cc
    values['loss']['router_balance_estimator'] = settings.router_balance_estimator
    # The shared T02 serializer writes pose-specific PCK/NME prose. T03's
    # actual trainer compares test_metrics['cc'] strictly; describe that here.
    values.setdefault("protocol", {}).update(
        checkpoint_selection="maximum public-test CC; ties retain the earlier selected epoch",
        primary_metric="CC",
        test_interval_epochs=settings.test_interval_epochs,
        test_at_epoch_one=True,
        test_at_final_epoch=True,
        test_evaluated_during_training=True,
        test_used_for_checkpoint_selection=True,
    )
    values["lightgen"].update(task="t03_saliency", ccd_normalization=settings.ccd_normalization,
                            electronic_grn=settings.electronic_grn,
                            electronic_ffn_spatial_dilation=settings.electronic_ffn_spatial_dilation,
                            electronic_global_rank=settings.electronic_global_rank,
                            electronic_ffn_hidden_width=settings.electronic_ffn_hidden_width,
                            electronic_ffn_groups=settings.electronic_ffn_groups,
                            electronic_spatial_kernel_size=settings.electronic_spatial_kernel_size)
    values.setdefault("augmentation", {}).update(enabled=settings.augmentation_enabled, mode=settings.augmentation_mode,
        crop_scale_min=settings.crop_scale_min, horizontal_flip_probability=settings.horizontal_flip_probability,
        brightness_jitter=settings.brightness_jitter, contrast_jitter=settings.contrast_jitter,
        end_epoch=settings.augmentation_end_epoch, apply_probability=settings.augmentation_apply_probability)
    values.setdefault("training", {}).update(
        learning_rate_source=settings.learning_rate_source,
        ema_decay=settings.ema_decay,
        phase_weight_decay=settings.phase_weight_decay,
        sam_rho=settings.sam_rho,
        asam=settings.asam,
        gsam_coefficient=settings.gsam_coefficient,
        noise_consistency_weight=settings.noise_consistency_weight,
        mixup=settings.mixup,
        initialization_checkpoint=str(settings.initialization_checkpoint) if settings.initialization_checkpoint else None,
        initialization_checkpoint_sha256=settings.initialization_checkpoint_sha256,
        reset_fusion_on_warmstart=settings.reset_fusion_on_warmstart,
        expand_kernel_on_warmstart=settings.expand_kernel_on_warmstart,
        initialize_grn_on_warmstart=settings.initialize_grn_on_warmstart,
        initialize_ffn_on_warmstart=settings.initialize_ffn_on_warmstart,
        ffn_spatial_learning_rate=settings.ffn_spatial_learning_rate,
        initialize_global_on_warmstart=settings.initialize_global_on_warmstart,
        widen_ffn_on_warmstart=settings.widen_ffn_on_warmstart,
        expand_ffn_groups_on_warmstart=settings.expand_ffn_groups_on_warmstart,
        exact_fusion_backward=settings.exact_fusion_backward,
        global_spatial_learning_rate=settings.global_spatial_learning_rate,
        adaptive_plateau={"enabled": settings.adaptive_plateau_enabled,
                          **settings.adaptive_plateau_options},
        staged={"enabled": settings.staged_training, "warmup_epochs": settings.staged_warmup_epochs,
                "polish_start": settings.staged_polish_start, "final_hard_balance": settings.staged_final_hard_balance,
                "freeze_electronic_gradients": settings.staged_freeze_electronic_gradients},
    )
    values["effective_optimizer_learning_rates"] = {
        name: getattr(settings, name) for name in (
            "student_learning_rate", "phase_learning_rate", "router_learning_rate",
            "dense_readout_learning_rate", "dense_head_learning_rate")
    }
    values["distillation"] = {"initial_weight": settings.distillation_initial_weight,
        "teacher_only_epochs": settings.teacher_only_epochs,
        "loss": settings.distillation_loss,
        "final_weight": settings.distillation_final_weight,
        "end_epoch": settings.distillation_end_epoch,
        "cache_file": str(settings.distillation_cache) if settings.distillation_cache else None,
        "teacher_sha256": settings.distillation_teacher_sha256}
    values['feature_hint'] = {'initial_weight':settings.feature_hint_initial_weight,
        'loss_mode':settings.feature_hint_loss_mode,
        'final_weight':settings.feature_hint_final_weight,'end_epoch':settings.feature_hint_end_epoch,
        'learning_rate':settings.feature_hint_learning_rate,
        'cache_file':str(settings.feature_hint_cache) if settings.feature_hint_cache else None,
        'inference_parameters_added':0,'training_only_projection_parameters':36864 if settings.feature_hint_initial_weight else 0}
    values['unlabeled_distillation'] = {
        'weight': settings.unlabeled_weight,
        'image_manifest': str(settings.unlabeled_image_manifest) if settings.unlabeled_image_manifest else None,
        'image_manifest_sha256': settings.unlabeled_image_manifest_sha256,
        'cache_file': str(settings.unlabeled_cache) if settings.unlabeled_cache else None,
        'cache_sha256': settings.unlabeled_cache_sha256,
        'inference_parameters_added': 0,
        'ground_truth_supervision_retained': True,
    }
    values['semantic_auxiliary'] = {
        'mode': settings.semantic_mode,
        'weight': settings.semantic_weight, 'learning_rate': settings.semantic_learning_rate,
        'targets_file': str(settings.semantic_targets) if settings.semantic_targets else None,
        'targets_sha256': settings.semantic_targets_sha256,
        'training_only_parameters': 15440 if settings.semantic_weight else 0,
        'inference_parameters_added': 0,
        'additional_human_semantic_supervision': bool(settings.semantic_weight),
        'additional_human_box_supervision': bool(settings.semantic_weight and settings.semantic_mode=='box_coverage'),
    }
    path.write_text(yaml.safe_dump(values, allow_unicode=True, sort_keys=False), encoding="utf-8")


__all__ = ["load_settings", "save_resolved_config"]
