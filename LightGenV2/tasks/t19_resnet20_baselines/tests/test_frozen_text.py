"""Offline identity, lossless text decoding and decoder-only training checks."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from LightGenV2.tasks.t19_resnet20_baselines import frozen_text as text
from LightGenV2.tasks.t19_resnet20_baselines.frozen_resnet18 import assert_encoder_unchanged
from LightGenV2.tasks.t19_resnet20_baselines.models import tokenize_text


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(300, 384)
        self.dropout = nn.Dropout(0.5)
        self.config = SimpleNamespace(hidden_size=384)
        self.pooler = None
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


class RecordingTokenizer:
    def __init__(self):
        self.texts = None
        self.options = None
        self.force_length = None

    def __call__(self, texts, **options):
        self.texts, self.options = texts, options
        lengths = [self.force_length or len(value) + 2 for value in texts]
        ids = torch.zeros(len(texts), max(lengths), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, length in enumerate(lengths):
            ids[row, :length] = torch.arange(1, length + 1) % 299
            mask[row, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture
def offline_text(tmp_path, monkeypatch):
    import huggingface_hub
    import transformers

    snapshot = tmp_path / text.MODEL_REVISION
    snapshot.mkdir()
    for name in text.SNAPSHOT_FILES:
        path = snapshot / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (snapshot / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": 256}))
    (snapshot / "1_Pooling/config.json").write_text(json.dumps({
        "word_embedding_dimension": 384, "pooling_mode_mean_tokens": True,
        "pooling_mode_cls_token": False, "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False}))
    tokenizer = RecordingTokenizer()
    transformer = TinyTransformer()
    calls = {}
    loading = {"missing_keys": [], "unexpected_keys": ["pooler.dense.weight", "pooler.dense.bias"],
               "mismatched_keys": [], "error_msgs": []}

    def download(**options):
        calls["download"] = options
        return str(snapshot)

    def load_tokenizer(path, **options):
        calls["tokenizer"] = (path, options)
        return tokenizer

    def load_model(path, **options):
        calls["model"] = (path, options)
        return transformer, loading

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(text, "MODEL_FILE_SHA256", text.file_sha256(snapshot / "model.safetensors"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    return SimpleNamespace(snapshot=snapshot, tokenizer=tokenizer, transformer=transformer,
                           calls=calls, loading=loading)


def test_lossless_byte_decode_ascii_chinese_and_reject_malformed():
    values = ["  Add BLUE café — 保留全文。\n", "x", ""]
    assert text.decode_byte_tokens(tokenize_text(values)) == values
    for bad in [torch.tensor([[1, 2, 0, 5]]), torch.tensor([[1, 258, 2]]),
                torch.tensor([[1, 1, 2]]), torch.tensor([[1, 7, 0]])]:
        with pytest.raises(ValueError):
            text.decode_byte_tokens(bad)
    with pytest.raises(ValueError, match="integer"):
        text.decode_byte_tokens(torch.tensor([[1., 2.]]))


def test_commit_pinned_model_tokenizer_and_all_files_recorded(offline_text):
    encoder = text.FrozenTextEncoder()
    assert offline_text.calls["download"]["revision"] == text.MODEL_REVISION
    assert offline_text.calls["download"]["repo_id"] == text.MODEL_ID
    assert offline_text.calls["download"]["local_files_only"] is True
    assert offline_text.calls["model"][0] == offline_text.calls["tokenizer"][0]
    assert offline_text.calls["model"][1]["add_pooling_layer"] is False
    assert offline_text.calls["model"][1]["output_loading_info"] is True
    assert offline_text.calls["tokenizer"][1]["local_files_only"] is True
    assert set(encoder.pretrained_provenance["snapshot_files_sha256"]) == set(text.SNAPSHOT_FILES)
    assert encoder.pretrained_provenance["encoder_trainable_parameters"] == 0
    assert encoder.pretrained_provenance["removed_unused_checkpoint_pooler_keys"] == [
        "pooler.dense.bias", "pooler.dense.weight"]


def test_full_text_masked_mean_normalized_and_no_truncation(offline_text):
    encoder = text.FrozenTextEncoder()
    values = ["hello 世界 café ", "x"]
    result = encoder(tokenize_text(values))
    assert offline_text.tokenizer.texts == values
    assert offline_text.tokenizer.options == dict(padding=True, truncation=False,
                                                add_special_tokens=True, return_tensors="pt")
    assert result.shape == (2, 384)
    torch.testing.assert_close(result.norm(dim=1), torch.ones(2))
    assert not result.requires_grad
    # Padding must not change the shorter sentence's sentence representation.
    alone = encoder(tokenize_text(["x"]))
    torch.testing.assert_close(result[1], alone[0])
    offline_text.tokenizer.force_length = 257
    before_calls = offline_text.transformer.calls
    with pytest.raises(ValueError, match="refusing to truncate"):
        encoder(tokenize_text(["safe bytes, too many tokenizer wordpieces"]))
    assert offline_text.transformer.calls == before_calls


def test_task_head_updates_but_pretrained_text_state_and_dropout_remain_frozen(offline_text):
    encoder = text.FrozenTextEncoder()
    expected = encoder.pretrained_provenance["encoder_state_sha256"]
    head = nn.Linear(384, 5)
    model = nn.Sequential(encoder, head)
    before = head.weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    model.train()
    assert not encoder.training and not encoder.transformer.dropout.training
    optimizer.zero_grad(set_to_none=True)
    model(tokenize_text(["instruction", "another instruction"])).square().mean().backward()
    optimizer.step()
    assert not torch.equal(head.weight, before)
    assert assert_encoder_unchanged(encoder, expected)["encoder_unchanged"]
    assert all(value.grad is None for value in encoder.parameters())


@pytest.mark.parametrize("key,value", [
    ("missing_keys", ["embeddings.word_embeddings.weight"]),
    ("unexpected_keys", ["unrecognized.weight"]),
    ("mismatched_keys", [("embedding", [1], [2])]),
    ("error_msgs", ["failed load"]),
])
def test_refuse_partial_random_or_unexpected_weight_load(offline_text, key, value):
    offline_text.loading[key] = value
    with pytest.raises(RuntimeError, match="not loaded completely"):
        text.FrozenTextEncoder()


def test_snapshot_requires_complete_files_and_original_pooling(offline_text):
    (offline_text.snapshot / "sentence_bert_config.json").write_text('{"max_seq_length":512}')
    with pytest.raises(RuntimeError, match="sequence-length"):
        text.FrozenTextEncoder()
