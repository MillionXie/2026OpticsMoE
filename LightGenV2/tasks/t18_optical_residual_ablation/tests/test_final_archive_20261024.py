import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archive_verifier',HERE/'verify_caltech20_archive_20261024.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(root,monkeypatch):
    code=b'original_model_source\n';code_sha=hashlib.sha256(code).hexdigest()
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:code)
    c=dict(schema='Caltech20_final_archive_20261024_v1',independent_generalization_claim=False,all_experiments_TEST_informed_development=True,rows=[],user_requested_rounded_table={},supporting_lineage=[],backup_receipts=[])
    for depth,rho,sim,exp in((2,0,299,280),(4,0,304,282),(6,0,308,284),(2,.3,337,333),(4,.3,346,341),(6,.3,354,348)):
        row=dict(depth=depth,rho=rho);c['rows'].append(row);table=c['user_requested_rounded_table'].setdefault(str(depth),{})
        for role,correct in(('simulation',sim),('experiment',exp)):
            name=f'{depth}_{rho}_{role}';best=root/(name+'.pt');last=root/(name+'_last.pt');best.write_bytes(name.encode());last.write_bytes((name+'last').encode());labels=np.arange(411)%20;pred=labels.copy();pred[correct:]=(pred[correct:]+1)%20;prob=np.eye(20)[pred];p=root/(name+'.npz');np.savez_compressed(p,ids=np.array([str(i)for i in range(411)]),labels=labels,probabilities=prob)
            display=f'{100*correct/411:.2f}';table[('nores'if rho==0 else'rho03')+('_sim'if role=='simulation'else'_exp')]=display
            row[role]=dict(correct=correct,samples=411,display_percent=display,selected_PT=best.name,selected_PT_SHA256=module.sha(best),last_PT=last.name,last_PT_SHA256=module.sha(last),predictions=p.name,predictions_SHA256=module.sha(p),source_git='historical_commit',source='model.py',source_SHA256=code_sha,proofs=[],teacher_seen_TEST=role=='experiment'and rho==.3 and depth in(4,6),official_TEST_accuracy=None,valid_for_formal_test_table=False)
    p=root/'manifest.json';p.write_text(json.dumps(c));return p,c


def test_exact_twelve_cells_and_original_TEST_id_identity(tmp_path,monkeypatch):
    p,c=fixture(tmp_path,monkeypatch);r=module.verify(tmp_path,p)
    assert r['passed']and len(r['cells'])==12 and r['same_TEST411_ID_order_and_labels']and not r['independent_generalization_claim']


def test_corrupted_weight_is_never_accepted(tmp_path,monkeypatch):
    p,c=fixture(tmp_path,monkeypatch);(tmp_path/c['rows'][0]['experiment']['selected_PT']).write_bytes(b'different')
    with pytest.raises(AssertionError):module.verify(tmp_path,p)


def test_teacher_TEST_lineage_cannot_be_removed(tmp_path,monkeypatch):
    p,c=fixture(tmp_path,monkeypatch);next(r for r in c['rows']if r['depth']==4 and r['rho']==.3)['experiment']['teacher_seen_TEST']=False;p.write_text(json.dumps(c))
    with pytest.raises(AssertionError):module.verify(tmp_path,p)
