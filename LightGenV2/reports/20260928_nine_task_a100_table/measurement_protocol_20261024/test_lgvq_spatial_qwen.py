"""No Torch/GPU: spatial targets, original forward reuse and raw reference pins."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('spatial_measure',HERE/'measure_lgvq_spatial_qwen.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_original_reference_is_exact_full558_not_subset():
    r=module.validate_reference(lambda p:hashlib.sha256(p.read_bytes()).hexdigest())
    assert r['test_videos']==558 and r['checkpoint_sha256']==module.PIN
    assert r['performance']['srcc']==pytest.approx(.6907725734336255)


def test_spatial_collect_preserves_cold_first_call_gaps_and_targets():
    records=[];evidence=[];progress=[];clocks=iter([1.,2.,3.,5.,8.,9.,10.,12.])
    first=iter([3.5,10.5]);rows=[dict(sample_id=str(i),spatial=20+i,temporal=99) for i in range(2)]
    module.collect(rows,lambda r:({'cpu':'real'},[[3,12,21,30]]),lambda x:x,
        lambda x:([11.],500.,490.),lambda:next(first),lambda:None,lambda:None,
        lambda r,c:evidence.append(r['sample_id']),progress.append,records,lambda:next(clocks))
    assert [r['target_mos'] for r in records]==[20,21]
    assert records[0]['preprocess_start']==1 and records[-1]['call_return']==12
    assert records[1]['preprocess_start']-records[0]['call_return']==3
    assert evidence==['0','1'] and progress==[1]


def test_native_network_and_full_vocab_are_not_replaced():
    tree=ast.parse((HERE/'measure_lgvq_spatial_qwen.py').read_text())
    names=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert 'original.load_runtime' in names and 'original.prepare_batch' in names and 'original.forward_batch' in names
    assert not any(n.startswith('model.') and n.endswith('forward') for n in names)
    assert "core.render_prompt(processor, 'spatial')" in ast.unparse(tree)
