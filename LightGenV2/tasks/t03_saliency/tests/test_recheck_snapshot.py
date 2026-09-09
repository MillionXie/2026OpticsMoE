import hashlib

import torch

from LightGenV2.tasks.t03_saliency.recheck_aligned import load_hashed_checkpoint


def test_digest_binds_loaded_bytes_not_later_best(tmp_path, monkeypatch):
    path = tmp_path / 'best_checkpoint.pt'
    torch.save({'epoch': 5, 'value': torch.tensor([.25])}, path)
    original = path.read_bytes()
    original_load = torch.load

    def load_then_replace(stream, **kwargs):
        payload = original_load(stream, **kwargs)
        torch.save({'epoch': 10, 'value': torch.tensor([.75])}, path)
        return payload

    monkeypatch.setattr(torch, 'load', load_then_replace)
    payload, digest = load_hashed_checkpoint(path)
    assert payload['epoch'] == 5
    torch.testing.assert_close(payload['value'], torch.tensor([.25]))
    assert digest == hashlib.sha256(original).hexdigest()
    assert digest != hashlib.sha256(path.read_bytes()).hexdigest()
