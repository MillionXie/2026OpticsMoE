from __future__ import annotations

import random
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .. import train
from ..model import QwenStemProgressiveOpticalImageNetBackbone
from .test_progressive_model import common_config, fake_stem


class TinyGrowthModel(nn.Module):
    def __init__(self, *, orphan: bool = False) -> None:
        super().__init__()
        self.new_phase = nn.Parameter(torch.tensor(0.10))
        self.carried_phase = nn.Parameter(torch.tensor(0.20))
        self.new_electronic = nn.Parameter(torch.tensor(0.30))
        self.carried_electronic = nn.Parameter(torch.tensor(0.40))
        self.head = nn.Linear(12, 6)
        if orphan:
            self.orphan = nn.Parameter(torch.tensor(1.0))
        self.feedback_method = "bp_current"
        self.feedback_seed: int | None = None
        self.full_depth = True

    def forward(self, images: torch.Tensor, *, ablation: str = "normal") -> torch.Tensor:
        del ablation
        value = images.flatten(1)
        scale = (
            self.new_phase
            + self.carried_phase
            + self.new_electronic
            + self.carried_electronic
        )
        return self.head(value * scale)

    def new_phase_parameters(self):
        yield self.new_phase

    def carried_phase_parameters(self):
        yield self.carried_phase

    def new_electronic_parameters(self):
        yield self.new_electronic

    def carried_electronic_parameters(self):
        yield self.carried_electronic

    def head_parameters(self):
        yield from self.head.parameters()

    def configure_feedback(self, method: str, *, random_seed: int = 0) -> None:
        self.feedback_method = method
        self.feedback_seed = int(random_seed) if method == "fa_random" else None

    def feedback_manifest(self) -> dict[str, object]:
        active = (
            f"random-{self.feedback_seed}"
            if self.feedback_method == "fa_random"
            else f"current-{float(self.new_phase.detach()):.8f}"
            if self.feedback_method == "bp_current"
            else "source-fixed"
        )
        return {
            "format": "tiny-feedback-v1",
            "method": self.feedback_method,
            "depth": 2,
            "connector_count": 2,
            "random_base_seed": self.feedback_seed,
            "feedback_phase_sequence_sha256": active,
            "source": {"phase_sequence_sha256": "source-sha"},
            "connections": [
                {
                    "connector_index_zero_based": index,
                    "axis": "token" if index == 0 else "channel",
                    "frozen": self.feedback_method != "bp_current",
                    "actual_stage_feedback_mode": (
                        "bp"
                        if self.feedback_method == "bp_current"
                        else "fa_pretrained"
                        if self.feedback_method == "fa_source"
                        else "fa_random"
                    ),
                    "source_phase_sha256": f"source-{index}",
                    "feedback_phase_sha256": f"{active}-{index}",
                    "propagation_transfer_sha256": f"transfer-{index}",
                    "random_substream_seed": (
                        None
                        if self.feedback_seed is None
                        else self.feedback_seed + index
                    ),
                }
                for index in range(2)
            ],
        }

    def depth_alpha_report(self) -> dict[str, object]:
        value = 1.0 if self.full_depth else 0.5
        return {
            "minimum": value,
            "maximum": value,
            "all_full_depth": self.full_depth,
        }


class FakeContext:
    def __init__(self, *, world_size: int = 1) -> None:
        self.device = torch.device("cpu")
        self.rank = 0
        self.local_rank = 0
        self.world_size = world_size
        self.is_main = False


class FakeDDP(nn.Module):
    _p13_test_ddp_wrapper = True

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.no_sync_calls = 0

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        yield


class CountingScheduler:
    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> None:
        self.calls += 1


class TinyStem(nn.Module):
    checkpoint_sha256 = "tiny-stem-sha256"


class TinyFormalGrowthModel(TinyGrowthModel):
    def __init__(self, stem_checkpoint: Path, model_config: dict[str, object]) -> None:
        del stem_checkpoint
        super().__init__()
        self.stem = TinyStem()
        self.model_config = dict(model_config)
        self.num_stages = int(model_config["num_stages"])
        self.register_buffer(
            "p13_progressive_architecture_signature",
            torch.tensor([13, 1, 2, self.num_stages], dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer("depth_alpha", torch.tensor(0.0), persistent=True)
        self.migration_manifest: dict[str, object] | None = None

    def apply_depth_ramp(self, epoch: int) -> float:
        value = 0.0 if int(epoch) <= 0 else 1.0
        self.depth_alpha.fill_(value)
        return value

    def depth_alpha_report(self) -> dict[str, object]:
        value = float(self.depth_alpha)
        return {
            "new_stage_count": 1,
            "minimum": value,
            "maximum": value,
            "mean": value,
            "all_full_depth": value == 1.0,
            "all_exact_bypass": value == 0.0,
        }

    def phase_snapshot(self) -> torch.Tensor:
        return torch.stack(
            (self.carried_phase.detach().cpu(), self.new_phase.detach().cpu())
        )

    def phase_motion(self, initial: torch.Tensor) -> dict[str, object]:
        displacement = (self.phase_snapshot() - initial).abs()
        return {
            "mean_absolute_rad": float(displacement.mean()),
            "per_stage_mean_absolute_rad": [float(value) for value in displacement],
        }

    def parameter_report(self) -> dict[str, object]:
        return {
            "architecture": "tiny-formal-growth",
            "num_stages": int(self.model_config["num_stages"]),
            "optical_fraction_of_backbone_trainable": 0.75,
            "minimum_optical_gate": 0.60,
            "depth_alpha": self.depth_alpha_report(),
            "migration_manifest": self.migration_manifest,
        }

    def optical_gates(self) -> list[float]:
        return [0.60, 0.60]

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("head.")
        }


class MainFakeContext(FakeContext):
    def __init__(self) -> None:
        super().__init__(world_size=1)
        self.is_main = True

    def barrier(self) -> None:
        return None


class TinySampler:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(int(epoch))


def tiny_config(*, method: str = "bp_current", seed: int | None = None):
    return {
        "optimizer": {
            "new_phase_learning_rate": 0.1,
            "carried_phase_learning_rate": 0.05,
            "new_electronic_learning_rate": 0.01,
            "carried_electronic_learning_rate": 0.005,
            "head_learning_rate": 0.02,
            "weight_decay": 0.001,
            "phase_gradient_clip_norm": 2.0,
            "electronic_gradient_clip_norm": 5.0,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "gradient_accumulation_steps": 3,
            "max_train_batches": 5,
            "use_amp": False,
            "amp_dtype": "float16",
            "log_interval_batches": 100,
        },
        "loss": {"label_smoothing": 0.0, "batch_mix_probability": 0.0},
        "feedback": {"method": method, "random_seed": seed},
    }


def test_optimizer_groups_are_disjoint_and_exhaustive() -> None:
    model = TinyGrowthModel()
    optimizer, schema = train.build_growth_optimizer(model, tiny_config())

    assert [group["name"] for group in optimizer.param_groups] == [
        "new_phase",
        "carried_phase",
        "new_electronic",
        "carried_electronic",
        "head",
    ]
    assert sum(item["parameter_elements"] for item in schema) == sum(
        parameter.numel() for parameter in model.parameters()
    )
    with pytest.raises(RuntimeError, match="do not exactly partition"):
        train.build_growth_optimizer(TinyGrowthModel(orphan=True), tiny_config())


def test_real_16stage_model_partitions_carried_and_new_parameters(tmp_path: Path) -> None:
    model_config = common_config()
    model_config.update({"num_stages": 16, "new_stage_alpha_init": 0.0})
    model = QwenStemProgressiveOpticalImageNetBackbone(
        fake_stem(tmp_path / "stem.pt"), model_config
    )
    optimizer, schema = train.build_growth_optimizer(model, tiny_config())
    by_name = {item["name"]: item for item in schema}

    assert by_name["new_phase"]["parameter_tensors"] == 8
    assert by_name["carried_phase"]["parameter_tensors"] == 8
    assert by_name["new_electronic"]["parameter_tensors"] == 8
    assert by_name["carried_electronic"]["parameter_elements"] > 900_000
    assert by_name["head"]["parameter_elements"] > 0
    assert sum(item["parameter_elements"] for item in schema) == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert {group["name"] for group in optimizer.param_groups} == set(by_name)

    feedback_config = tiny_config(method="fa_source")
    active = train.configure_feedback_strict(model, feedback_config)
    saved = train.feedback_checkpoint_state(model, feedback_config)
    model.configure_feedback("bp_current")
    model.feedback_source_provenance = {"capture": "simulated_load_hook_reset"}
    restored = train.configure_feedback_strict(
        model, feedback_config, saved=saved
    )
    assert model.feedback_source_provenance == saved["manifest"]["source"][
        "provenance"
    ]
    assert saved["manifest_sha256"] == restored["manifest_sha256"]
    assert active["runtime_contract_sha256"] == restored[
        "runtime_contract_sha256"
    ]
    model.slots[3].stage.set_feedback("bp")
    with pytest.raises(RuntimeError, match="silently changed feedback mode"):
        train.assert_feedback_runtime(model, restored)


def test_accumulation_uses_no_sync_and_scheduler_only_on_updates() -> None:
    core = TinyGrowthModel()
    model = FakeDDP(core)
    config = tiny_config()
    optimizer, _ = train.build_growth_optimizer(core, config)
    scheduler = CountingScheduler()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(19)
    loader = [
        {
            "image": torch.randn(2, 3, 2, 2, generator=generator),
            "label": torch.tensor([index % 6, (index + 1) % 6]),
        }
        for index in range(5)
    ]

    metrics, gradients, global_step = train.train_epoch(
        model,
        loader,
        optimizer,
        scheduler,
        scaler,
        config,
        FakeContext(world_size=2),
        epoch=1,
        global_optimizer_step=7,
    )

    assert train.optimizer_updates_per_epoch(5, 5, 3) == 2
    assert metrics["optimizer_updates"] == 2
    assert metrics["micro_batches"] == 5
    assert scheduler.calls == 2
    assert global_step == 9
    assert model.no_sync_calls == 3
    assert gradients is not None
    assert gradients["new_phase"]["all_present"] is True


def test_lambda_scheduler_counter_is_optimizer_update_counter() -> None:
    model = TinyGrowthModel()
    config = tiny_config()
    optimizer, _ = train.build_growth_optimizer(model, config)
    scheduler = train.build_update_scheduler(optimizer, config, updates_per_epoch=2)
    train.assert_scheduler_update_alignment(scheduler, 0)
    optimizer.step()
    scheduler.step()
    train.assert_scheduler_update_alignment(scheduler, 1)
    with pytest.raises(RuntimeError, match="Scheduler/update mismatch"):
        train.assert_scheduler_update_alignment(scheduler, 2)


def test_feedback_resume_reconfigures_and_hash_checks() -> None:
    model = TinyGrowthModel()
    config = tiny_config(method="fa_random", seed=71)
    active = train.configure_feedback_strict(model, config)
    saved = train.feedback_checkpoint_state(model, config)
    assert active["exact_resume_sha256"] == saved["exact_resume_sha256"]

    model.configure_feedback("bp_current")
    restored = train.configure_feedback_strict(model, config, saved=saved)
    assert model.feedback_method == "fa_random"
    assert restored["random_seed"] == 71
    train.assert_feedback_runtime(model, restored)

    wrong_config = tiny_config(method="fa_random", seed=72)
    with pytest.raises(RuntimeError, match="differs from the checkpoint"):
        train.configure_feedback_strict(model, wrong_config, saved=saved)
    tampered = dict(saved)
    tampered["exact_resume_sha256"] = "tampered"
    with pytest.raises(RuntimeError, match="does not hash-match"):
        train.configure_feedback_strict(model, config, saved=tampered)
    tampered_manifest = dict(saved)
    tampered_manifest["manifest"] = {
        **saved["manifest"],
        "connector_count": 999,
    }
    with pytest.raises(RuntimeError, match="manifest hash is inconsistent"):
        train.configure_feedback_strict(model, config, saved=tampered_manifest)

    provenance = train._validated_parent_feedback_provenance({"feedback": saved})
    assert provenance["method"] == "fa_random"
    assert provenance["manifest_sha256"] == saved["manifest_sha256"]
    bad_parent = {"feedback": {**saved, "manifest_sha256": "0" * 64}}
    with pytest.raises(RuntimeError, match="manifest SHA-256 is invalid"):
        train._validated_parent_feedback_provenance(bad_parent)


def test_rng_round_trip_restores_python_numpy_and_torch() -> None:
    context = FakeContext()
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    state = train.capture_rng_state(context)
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    random.random()
    np.random.rand()
    torch.rand(3)

    train.restore_rng_state(state, context)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)


def test_loader_restart_does_not_shift_split_resume_mix_rng() -> None:
    context = FakeContext()
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    checkpoint_rng = train.capture_rng_state(context)
    images = torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2)
    labels = torch.arange(4)
    config = tiny_config()
    config["loss"] = {
        "label_smoothing": 0.0,
        "batch_mix_probability": 1.0,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,
    }

    # An uninterrupted persistent-worker epoch reset does not draw a new base
    # seed from the main RNG. This is the exact next batch-mixing trajectory.
    expected = train.mix_batch(images, labels, config)

    # A resumed process constructs a new iterator. Its worker base seed must be
    # drawn from the dedicated loader generator, leaving all three training RNGs
    # at the checkpoint state before mixup/cutmix is sampled.
    train.restore_rng_state(checkpoint_rng, context)
    loader = DataLoader(TensorDataset(torch.arange(2)), batch_size=1, num_workers=0)
    seed = train.dataloader_generator_seed(2026, split="train", rank=0)
    generator = train.attach_dataloader_generator(loader, seed)
    next(iter(loader))
    actual = train.mix_batch(images, labels, config)

    assert generator.initial_seed() == seed
    assert actual[3:] == expected[3:]
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)
    assert train.dataloader_generator_seed(
        2026, split="validation", rank=0
    ) != seed
    assert train.dataloader_generator_seed(2026, split="train", rank=1) != seed


def test_implementation_manifest_hashes_dirty_files_and_runtime(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("value = 1\n", encoding="utf-8")
    second.write_text("value = 2\n", encoding="utf-8")
    manifest = train.training_implementation_manifest(
        repository_root=tmp_path,
        relative_paths=("first.py", "second.py"),
    )
    assert manifest["format"] == train.IMPLEMENTATION_MANIFEST_FORMAT
    assert [item["path"] for item in manifest["files"]] == [
        "first.py",
        "second.py",
    ]
    assert manifest["runtime"]["torch"] == str(torch.__version__)
    train.assert_implementation_manifest_matches(manifest, manifest)

    second.write_text("value = 3\n", encoding="utf-8")
    changed = train.training_implementation_manifest(
        repository_root=tmp_path,
        relative_paths=("first.py", "second.py"),
    )
    with pytest.raises(RuntimeError, match="implementation/runtime differs"):
        train.assert_implementation_manifest_matches(manifest, changed)


def test_fresh_run_rejects_precheckpoint_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "launch.pid").write_text("123\n", encoding="utf-8")
    assert train.fresh_run_artifacts(output) == []
    (output / "manifest.json").write_text("{}\n", encoding="utf-8")
    assert train.fresh_run_artifacts(output) == [output / "manifest.json"]
    (output / "manifest.json").unlink()
    (output / "metrics").mkdir()
    (output / "metrics" / "initial_baseline.json").write_text(
        "{}\n", encoding="utf-8"
    )
    assert train.fresh_run_artifacts(output) == [
        output / "metrics" / "initial_baseline.json"
    ]


def test_launchers_use_lifetime_lock_and_segmented_resume_logs() -> None:
    experiment = Path(__file__).resolve().parents[1]
    common = (experiment / "commands" / "_training_common.sh").read_text(
        encoding="utf-8"
    )
    assert "flock -n 9" in common
    assert "training_run_has_artifacts" in common
    assert "segmented_log_path" in common
    assert "specified more than once" in common

    for name in (
        "06_launch_growth16_fa_source_20e.sh",
        "09_launch_p11_matched_continue_20e.sh",
        "13_launch_progressive_growth.sh",
    ):
        script = (experiment / "commands" / name).read_text(encoding="utf-8")
        assert script.index("acquire_launch_lock") < script.index(
            "training_mode_argument"
        )
        assert "ln -sfn" in script
        assert '>> "${LOG}" 2>&1' in script
        assert 'mv "${pid_tmp}" "${PID_FILE}"' in script


def test_full_depth_gate_is_exact_and_cli_forces_fresh_resume_split() -> None:
    model = TinyGrowthModel()
    assert train.is_full_depth(model) is True
    model.full_depth = False
    assert train.is_full_depth(model) is False
    assert train.checkpoint_roles_for_epoch(
        improved_any=True,
        improved_full=False,
        full_depth=False,
    ) == ["last", "best_any"]
    assert train.checkpoint_roles_for_epoch(
        improved_any=False,
        improved_full=True,
        full_depth=True,
    ) == ["last", "best_full_depth"]
    with pytest.raises(RuntimeError, match="cannot be selected"):
        train.checkpoint_roles_for_epoch(
            improved_any=True,
            improved_full=True,
            full_depth=False,
        )

    assert train.parse_args(["--config", "x.yaml", "--fresh"]).fresh is True
    assert train.parse_args(["--config", "x.yaml", "--resume"]).resume is True
    with pytest.raises(SystemExit):
        train.parse_args(["--config", "x.yaml"])
    with pytest.raises(SystemExit):
        train.parse_args(["--config", "x.yaml", "--fresh", "--resume"])


def test_p11_to_16_initializer_requires_two_epoch88_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locked_identities = (
        train.OFFICIAL_P11_BACKBONE_SHA256,
        train.OFFICIAL_P11_TRAINING_SHA256,
        train.OFFICIAL_P11_CONFIG_DIGEST,
    )
    assert all(len(value) == 64 for value in locked_identities)
    assert all(int(value, 16) >= 0 for value in locked_identities)
    with pytest.raises(RuntimeError, match="not a SHA-256"):
        train._locked_sha256(
            {"identity": "0" * 64},
            "identity",
            "0" * 65,
        )

    backbone = tmp_path / "backbone.pt"
    training = tmp_path / "best.pt"
    torch.save(
        {"best_epoch": 88, "config_digest": train.OFFICIAL_P11_CONFIG_DIGEST},
        backbone,
    )
    torch.save(
        {"epoch": 88, "config_digest": train.OFFICIAL_P11_CONFIG_DIGEST},
        training,
    )
    called = {}

    def fake_migrate(model, backbone_checkpoint, training_checkpoint):
        called["values"] = (model, Path(backbone_checkpoint), Path(training_checkpoint))
        return {
            "strict": True,
            "source_checkpoint_sha256": train.OFFICIAL_P11_BACKBONE_SHA256,
            "source_training_checkpoint_sha256": train.OFFICIAL_P11_TRAINING_SHA256,
        }

    monkeypatch.setattr(
        train.migration_api,
        "migrate_strict_p11_training_checkpoint",
        fake_migrate,
    )
    monkeypatch.setattr(
        train,
        "sha256_file",
        lambda path: (
            train.OFFICIAL_P11_BACKBONE_SHA256
            if Path(path).resolve() == backbone.resolve()
            else train.OFFICIAL_P11_TRAINING_SHA256
        ),
    )
    model = TinyGrowthModel()
    config = {
        "model": {"num_stages": 16},
        "initialization": {
            "mode": "p11_to_16",
            "p11_backbone_checkpoint": str(backbone),
            "p11_training_checkpoint": str(training),
            "expected_p11_best_epoch": 88,
            "expected_p11_backbone_sha256": train.OFFICIAL_P11_BACKBONE_SHA256,
            "expected_p11_training_sha256": train.OFFICIAL_P11_TRAINING_SHA256,
            "expected_p11_config_digest": train.OFFICIAL_P11_CONFIG_DIGEST,
        },
    }
    manifest = train.initialize_p13_fresh(model, config)
    assert manifest["source_depth"] == 8
    assert manifest["target_depth"] == 16
    assert manifest["migration"]["strict"] is True
    assert called["values"][1:] == (backbone.resolve(), training.resolve())

    wrong_identity = {
        **config,
        "initialization": {
            **config["initialization"],
            "expected_p11_backbone_sha256": "0" * 64,
        },
    }
    with pytest.raises(RuntimeError, match="locked official identity"):
        train.initialize_p13_fresh(model, wrong_identity)
    with pytest.raises(RuntimeError, match="cannot reconstruct"):
        train._official_p11_config_guard({"mixer_dropout": 0.0})

    torch.save(
        {"epoch": 87, "config_digest": train.OFFICIAL_P11_CONFIG_DIGEST},
        training,
    )
    with pytest.raises(RuntimeError, match="epoch-88"):
        train.initialize_p13_fresh(model, config)


@pytest.mark.parametrize(
    "name",
    [
        "gpu_smoke_full_image.yaml",
        "gpu_smoke_full_image_4rank_gb192.yaml",
        "growth16_fa_source_20e_gb192.yaml",
        "p11_epoch88_matched_continue_20e_gb192.yaml",
    ],
)
def test_formal_configs_pin_canonical_p11_source_shas(name: str) -> None:
    config_path = Path(train.__file__).with_name("configs") / name
    initialization = train.load_config(config_path)["initialization"]
    assert (
        initialization["expected_p11_backbone_sha256"]
        == train.OFFICIAL_P11_BACKBONE_SHA256
    )
    assert (
        initialization["expected_p11_training_sha256"]
        == train.OFFICIAL_P11_TRAINING_SHA256
    )


def test_four_rank_full_image_smoke_matches_formal_global_batch_contract() -> None:
    experiment = Path(train.__file__).resolve().parent
    config = train.load_config(
        experiment / "configs" / "gpu_smoke_full_image_4rank_gb192.yaml"
    )
    training = config["training"]

    assert config["output_dir"].endswith(
        "p13_growth16_full_image_4rank_gb192_gpu_smoke"
    )
    assert config["model"]["canvas_size"] == 224
    assert training["epochs"] == 1
    assert training["batch_size"] == 24
    assert training["gradient_accumulation_steps"] == 2
    assert training["expected_world_size"] == 4
    assert training["expected_effective_global_batch"] == 192
    assert training["max_train_batches"] == 2
    assert training["max_validation_batches"] == 1


def test_four_rank_full_image_smoke_launcher_is_foreground_and_guarded() -> None:
    experiment = Path(train.__file__).resolve().parent
    script = (
        experiment
        / "commands"
        / "05b_gpu_smoke_full_image_4rank_gb192.sh"
    ).read_text(encoding="utf-8")

    assert "PHYSICAL_GPU_INDICES" in script
    assert '"${#indices[@]}" -ne 4' in script
    assert 'visible_gpu_uuids "${PHYSICAL_GPU_INDICES}"' in script
    assert "gpu_smoke_full_image_4rank_gb192.yaml" in script
    assert script.index("acquire_launch_lock") < script.index(
        "training_mode_argument"
    )
    assert 'exec "${TORCHRUN_BIN}" --standalone --nproc_per_node=4' in script
    assert "nohup" not in script


def test_migration_provenance_restore_rejects_wrong_target() -> None:
    model = TinyFormalGrowthModel(Path("unused.pt"), {"num_stages": 16})
    manifest = {
        "source_depth": 8,
        "target_depth": 16,
        "migration": {
            "source_depth": 8,
            "target_num_stages": 16,
            "target_architecture_signature": [13, 1, 2, 16],
        },
    }
    restored = train.restore_migration_provenance(
        model, manifest, {"num_stages": 16}
    )
    assert model.migration_manifest == restored

    wrong_depth = {
        **manifest,
        "migration": {
            **manifest["migration"],
            "target_num_stages": 32,
        },
    }
    with pytest.raises(RuntimeError, match="target depth"):
        train.restore_migration_provenance(
            model, wrong_depth, {"num_stages": 16}
        )

    wrong_signature = {
        **manifest,
        "migration": {
            **manifest["migration"],
            "target_architecture_signature": [13, 1, 2, 32],
        },
    }
    with pytest.raises(RuntimeError, match="target signature"):
        train.restore_migration_provenance(
            model, wrong_signature, {"num_stages": 16}
        )


def test_end_to_end_fresh_then_same_depth_resume_checkpoint_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(811)
    train_loader = [
        {
            "image": torch.randn(2, 3, 2, 2, generator=generator),
            "label": torch.tensor([0, 1]),
        },
        {
            "image": torch.randn(2, 3, 2, 2, generator=generator),
            "label": torch.tensor([2, 3]),
        },
    ]
    validation_loader = [
        {
            "image": torch.randn(2, 3, 2, 2, generator=generator),
            "label": torch.tensor([0, 1]),
        }
    ]

    def fake_loaders(config, context):
        del config, context
        return (
            SimpleNamespace(digest="tiny-dataset"),
            train_loader,
            validation_loader,
            TinySampler(),
            TinySampler(),
            list(range(4)),
            list(range(2)),
        )

    monkeypatch.setattr(train, "_dataset_loaders", fake_loaders)
    output = tmp_path / "run"
    config = tiny_config(method="fa_source")
    config.update(
        {
            "_config_path": str(tmp_path / "config.yaml"),
            "_config_digest": "tiny-config-digest",
            "output_dir": str(output),
            "imagenet_config": str(tmp_path / "imagenet.json"),
            "stem_checkpoint": str(tmp_path / "stem.pt"),
            "model": {
                "num_stages": 16,
                "minimum_optical_parameter_fraction": 0.50,
            },
            "initialization": {"mode": "tiny-fresh"},
        }
    )
    config["training"].update(
        {
            "epochs": 1,
            "batch_size": 2,
            "validation_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "expected_world_size": 1,
            "expected_effective_global_batch": 4,
            "max_train_batches": 2,
            "max_validation_batches": 1,
            "checkpoint_interval_epochs": 1,
            "run_final_ablations": False,
        }
    )

    def initialize_tiny(model, values):
        del model, values
        return {
            "mode": "tiny-fresh",
            "source_depth": 8,
            "target_depth": 16,
            "migration": {
                "source_depth": 8,
                "target_num_stages": 16,
                "target_architecture_signature": [13, 1, 2, 16],
            },
        }

    context = MainFakeContext()
    train.run_training(
        config,
        context,
        resume=False,
        model_class=TinyFormalGrowthModel,
        fresh_initializer=initialize_tiny,
    )

    checkpoint_dir = output / "checkpoints"
    assert (checkpoint_dir / "last.pt").is_file()
    assert (checkpoint_dir / "best_any.pt").is_file()
    assert (checkpoint_dir / "best_full_depth.pt").is_file()
    assert (checkpoint_dir / "backbone_full_depth.pt").is_file()
    last = torch.load(checkpoint_dir / "last.pt", weights_only=False)
    assert last["format"] == train.TRAINING_CHECKPOINT_FORMAT
    assert last["checkpoint_role"] == "last"
    assert last["epoch"] == 1
    assert last["global_optimizer_step"] == 1
    assert last["depth_alpha"]["all_full_depth"] is True
    assert last["feedback"]["method"] == "fa_source"
    assert last["model_report"]["migration_manifest"] == last[
        "migration_manifest"
    ]
    assert last["implementation_manifest"]["aggregate_sha256"]
    assert len(last["rng_states"]) == 1
    run_manifest = __import__("json").loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["implementation_manifest"] == last[
        "implementation_manifest"
    ]
    export = torch.load(
        checkpoint_dir / "backbone_full_depth.pt", weights_only=False
    )
    assert export["implementation_manifest"] == last["implementation_manifest"]

    tampered_last = dict(last)
    tampered_last["implementation_manifest"] = {
        **last["implementation_manifest"],
        "aggregate_sha256": "0" * 64,
    }
    torch.save(tampered_last, checkpoint_dir / "last.pt")
    with pytest.raises(RuntimeError, match="implementation/runtime differs"):
        train.run_training(
            config,
            MainFakeContext(),
            resume=True,
            model_class=TinyFormalGrowthModel,
            fresh_initializer=lambda *_: pytest.fail("resume called fresh migration"),
        )
    torch.save(last, checkpoint_dir / "last.pt")

    # Same-depth resume must load last.pt, reconstruct FA and pass its exact
    # connector hash before reaching the alpha-one export path again.
    train.run_training(
        config,
        MainFakeContext(),
        resume=True,
        model_class=TinyFormalGrowthModel,
        fresh_initializer=lambda *_: pytest.fail("resume called fresh migration"),
    )
    result = __import__("json").loads(
        (output / "result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "complete"
    assert result["depth_alpha"]["all_full_depth"] is True
    assert result["model"]["migration_manifest"] == last["migration_manifest"]
    assert result["implementation_manifest"] == last["implementation_manifest"]
