"""Pinned, pretrained MiniLM sentence features with no trainable text encoder.

The legacy task input remains losslessly represented by UTF-8 byte tokens.
Those bytes are decoded before the pretrained tokenizer is applied; neither
byte inputs nor wordpiece inputs are silently truncated.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .frozen_resnet18 import encoder_state_sha256, file_sha256


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Official HF model API, checked 2026-10-23. Never resolve a moving main branch.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILE_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
MAX_WORDPIECES = 256
SNAPSHOT_FILES = (
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt", "sentence_bert_config.json",
    "1_Pooling/config.json", "modules.json",
)


def decode_byte_tokens(tokens: torch.Tensor) -> list[str]:
    """Strict inverse of tokenize_text: PAD0, BOS1, EOS2, raw byte plus three."""
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] < 2:
        raise ValueError("Expected nonempty [B,L] UTF-8 byte tokens")
    if tokens.dtype not in {torch.int32, torch.int64}:
        raise ValueError("Byte tokens must have an integer dtype")
    texts = []
    for row in tokens.detach().cpu().tolist():
        end = row.index(0) if 0 in row else len(row)
        if any(value != 0 for value in row[end:]):
            raise ValueError("Byte token padding must be right-contiguous")
        sequence = row[:end]
        if len(sequence) < 2 or sequence[0] != 1 or sequence[-1] != 2:
            raise ValueError("Byte token sequence requires BOS/EOS")
        if any(not 3 <= value <= 258 for value in sequence[1:-1]):
            raise ValueError("Invalid raw UTF-8 byte token or interior BOS/EOS")
        try:
            texts.append(bytes(value - 3 for value in sequence[1:-1]).decode("utf-8", errors="strict"))
        except UnicodeDecodeError as error:
            raise ValueError("Invalid UTF-8 byte token sequence") from error
    return texts


def _snapshot_identity(snapshot: Path) -> dict:
    if snapshot.name != MODEL_REVISION:
        raise RuntimeError("MiniLM snapshot path does not identify the pinned commit")
    missing = [name for name in SNAPSHOT_FILES if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError(f"Pinned MiniLM snapshot is incomplete: {missing}")
    file_identities = {name: file_sha256(snapshot / name) for name in SNAPSHOT_FILES}
    if file_identities["model.safetensors"] != MODEL_FILE_SHA256:
        raise RuntimeError("Pinned MiniLM model.safetensors failed its complete SHA256 identity")
    sentence_config = json.loads((snapshot / "sentence_bert_config.json").read_text(encoding="utf-8"))
    if sentence_config.get("max_seq_length") != MAX_WORDPIECES:
        raise RuntimeError("Pinned MiniLM sequence-length contract changed")
    pooling = json.loads((snapshot / "1_Pooling/config.json").read_text(encoding="utf-8"))
    if (pooling.get("word_embedding_dimension") != 384
            or pooling.get("pooling_mode_mean_tokens") is not True
            or any(value for name, value in pooling.items()
                   if name.startswith("pooling_mode_") and name != "pooling_mode_mean_tokens")):
        raise RuntimeError("Pinned MiniLM mean-pooling contract changed")
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_path": str(snapshot.resolve()),
        "snapshot_files_sha256": file_identities,
        "model_file_full_sha256_verified": True,
        "max_wordpieces_including_special_tokens": MAX_WORDPIECES,
        "pooling": "attention-mask weighted mean followed by L2 normalization",
        "byte_input_decoding": "strict UTF-8 inverse; full text; no truncation",
    }


class FrozenTextEncoder(nn.Module):
    """Frozen 384-dimensional sentence features for existing byte-token inputs."""

    output_dim = 384

    def __init__(self):
        super().__init__()
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError
        from transformers import AutoModel, AutoTokenizer

        snapshot_options = dict(repo_id=MODEL_ID, revision=MODEL_REVISION,
                                allow_patterns=list(SNAPSHOT_FILES))
        try:
            snapshot = Path(snapshot_download(**snapshot_options, local_files_only=True))
        except LocalEntryNotFoundError:
            snapshot = Path(snapshot_download(**snapshot_options, local_files_only=False))
        identity = _snapshot_identity(snapshot)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, use_fast=True)
        self.transformer, loading = AutoModel.from_pretrained(
            str(snapshot), local_files_only=True, use_safetensors=True,
            add_pooling_layer=False, output_loading_info=True)
        # The classifier-style BERT pooler is unused by sentence mean pooling.
        # It is explicitly absent from this architecture, never newly initialized.
        omitted_pooler = {"pooler.dense.weight", "pooler.dense.bias"}
        unexpected = set(loading.get("unexpected_keys", []))
        if (loading.get("missing_keys") or loading.get("mismatched_keys")
                or loading.get("error_msgs") or unexpected - omitted_pooler):
            raise RuntimeError(f"MiniLM weights were not loaded completely: {loading}")
        if getattr(self.transformer.config, "hidden_size", None) != self.output_dim:
            raise RuntimeError("Pinned MiniLM hidden dimension changed")
        if getattr(self.transformer, "pooler", None) is not None:
            raise RuntimeError("MiniLM mean-pooling encoder unexpectedly has a BERT pooler")
        self.requires_grad_(False)
        self.train(False)
        self.pretrained_provenance = {
            **identity,
            "encoder_state_sha256": encoder_state_sha256(self),
            "encoder_parameters": sum(value.numel() for value in self.parameters()),
            "encoder_trainable_parameters": 0,
            "parameters_frozen": True,
            "all_encoder_modules_eval": True,
            "loading_missing_keys": [],
            "loading_mismatched_keys": [],
            "removed_unused_checkpoint_pooler_keys": sorted(unexpected),
        }

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        texts = decode_byte_tokens(tokens)
        encoded = self.tokenizer(texts, padding=True, truncation=False,
                                 add_special_tokens=True, return_tensors="pt")
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise RuntimeError("MiniLM tokenizer did not provide input IDs and attention mask")
        if encoded["input_ids"].ndim != 2 or encoded["input_ids"].shape[0] != len(texts):
            raise RuntimeError("MiniLM tokenizer returned an invalid batch")
        if encoded["input_ids"].shape[1] > MAX_WORDPIECES:
            raise ValueError("Text exceeds MiniLM 256-wordpiece limit; refusing to truncate")
        if encoded["attention_mask"].shape != encoded["input_ids"].shape:
            raise RuntimeError("MiniLM token IDs and attention mask do not align")
        device = next(self.transformer.parameters()).device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        result = self.transformer(**encoded)
        features = result.last_hidden_state
        if features.shape != (*encoded["input_ids"].shape, self.output_dim):
            raise RuntimeError("MiniLM returned an invalid token-feature shape")
        mask = encoded["attention_mask"].unsqueeze(-1).to(features.dtype)
        mean = (features * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        return F.normalize(mean, p=2, dim=1)
