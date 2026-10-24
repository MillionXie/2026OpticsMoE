"""Read-only identities/saved-predictions audit of the final 12-cell archive.

Does not refit, capture or claim a fresh model performance evaluation. Existing
full native CPU proofs are retained separately. TEST-adaptation lineage remains
explicit even when saved student gradients contain TRAIN inputs only.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def verify(root,manifest):
    c=read(manifest);assert c['schema']=='Caltech20_final_archive_20261024_v1'
    assert c['independent_generalization_claim']is False and c['all_experiments_TEST_informed_development']
    expected={(2,0.):(299,280),(4,0.):(304,282),(6,0.):(308,284),(2,.3):(337,333),(4,.3):(346,341),(6,.3):(354,348)}
    assert len(c['rows'])==6;identity=None;checked=[];seen=set();commits=set()
    for row in c['rows']:
        key=(row['depth'],row['rho']);assert key in expected and key not in seen;seen.add(key)
        for i,role in enumerate(('simulation','experiment')):
            a=row[role];correct=expected[key][i];assert a['correct']==correct and a['samples']==411
            display=f'{100*correct/411:.2f}';assert a['display_percent']==display
            column=('nores'if row['rho']==0 else'rho03')+('_sim'if role=='simulation'else'_exp')
            assert c['user_requested_rounded_table'][str(row['depth'])][column]==display
            assert sha(root/a['selected_PT'])==a['selected_PT_SHA256']and sha(root/a['last_PT'])==a['last_PT_SHA256']
            assert sha(root/a['predictions'])==a['predictions_SHA256']
            for proof in a['proofs']:assert sha(root/proof['path'])==proof['SHA256']
            with np.load(root/a['predictions'],allow_pickle=False)as z:
                ids=z['ids'];labels=z['labels'];prob=z['probabilities']
                assert prob.shape==(411,20)and len(ids)==len(set(ids))==411 and np.isfinite(prob).all()
                assert np.allclose(prob.sum(1),1,atol=3e-6)and int((prob.argmax(1)==labels).sum())==correct
                if identity is None:identity=(ids.copy(),labels.copy())
                else:assert np.array_equal(ids,identity[0])and np.array_equal(labels,identity[1])
            blob=subprocess.check_output(['git','show',a['source_git']+':'+a['source']],cwd=root)
            assert hashlib.sha256(blob).hexdigest()==a['source_SHA256'];commits.add(a['source_git'])
            if role=='experiment'and row['rho']==.3 and row['depth']in(4,6):
                assert a['teacher_seen_TEST']and a['official_TEST_accuracy']is None and not a['valid_for_formal_test_table']
            checked.append(dict(depth=key[0],rho=key[1],role=role,correct=correct,samples=411,accuracy=correct/411,PT_SHA256=a['selected_PT_SHA256'],saved_predictions_verified=True,source_blob_verified=True))
    assert set(expected)==seen
    for item in c['supporting_lineage']:assert sha(root/item['path'])==item['SHA256']
    for receipt in c['backup_receipts']:
        path=root/receipt['path'];assert sha(path)==receipt['SHA256'];r=read(path)
        assert r.get('passed',r.get('all_local_remote_SHA_matched',False))
        for f in r['files']:
            p=root/f['path'];assert p.stat().st_size==f['bytes']and sha(p)==f['SHA256'],f['path']
    return dict(passed=True,manifest_SHA256=sha(manifest),verifier_source_SHA256=sha(Path(__file__)),cells=checked,unique_source_commits=len(commits),same_TEST411_ID_order_and_labels=True,all_selected_bestlast_SHA_verified=True,all_backup_files_rehashed=True,retained_native_CPU_proofs_verified_by_SHA=True,model_not_rerun=True,no_training=True,SDK_used=False,GPU_used=False,independent_generalization_claim=False,teacher_TEST_information_disclosed=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--manifest',type=Path,default=Path(__file__).parent/'reports/caltech20_final_20261024/final_manifest.json');p.add_argument('--out',type=Path);a=p.parse_args();r=verify(a.root,a.manifest)
    if a.out:
        assert not a.out.exists();a.out.write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps(r))
