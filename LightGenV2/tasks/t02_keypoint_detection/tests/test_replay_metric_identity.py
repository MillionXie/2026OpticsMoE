import json
from types import SimpleNamespace

import pytest

from LightGenV2.tasks.t02_keypoint_detection import lab_replay_metrics as entry


@pytest.mark.parametrize('status,samples,digest', [('incomplete', 1000, 'expected'), ('complete', 999, 'expected'), ('complete', 1000, 'wrong')])
def test_reject_identity_before_dataset(tmp_path, monkeypatch, status, samples, digest):
    indices = tmp_path / 'argmax_indices.json'
    indices.write_text('[]')
    indices.with_name('report.json').write_text(json.dumps({'status': status, 'samples': samples, 'checkpoint_sha256': digest}))
    monkeypatch.setattr(entry, 'load_settings', lambda _: SimpleNamespace())
    monkeypatch.setattr(entry, 'prepare_lsp', lambda *a, **kw: pytest.fail('Must reject before reading dataset'))
    monkeypatch.setattr('sys.argv', ['metrics', '--indices', str(indices), '--output', str(tmp_path/'output'), '--expected-checkpoint-sha256', 'expected'])
    with pytest.raises(ValueError, match='identity mismatch'):
        entry.main()
