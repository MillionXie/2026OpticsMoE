import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
run=root/'runs/layerwise_physical2400'
contract=json.loads((run/'contract.json').read_text())
report=json.loads((run/'report.json').read_text())
ids=contract['ids'];assert len(ids)==2400 and len(set(ids))==2400
summary={}
for stage in report['ccd_counts']:
    folder=run/'ccd'/stage
    pngs={p.stem for p in folder.glob('*.png') if not p.stem.endswith(('_warmup0','_warmup1'))}
    assert pngs==set(ids),(stage,len(pngs))
    phase=hashlib.sha256((run/'phase'/(stage+'.bmp')).read_bytes()).hexdigest()
    sat=[];p99=[]
    for sid in ids:
        r=json.loads((folder/(sid+'.json')).read_text())
        assert r['sample_id']==sid and r['stage']==stage and r['phase_sha256']==phase
        assert r['exposure']['exposure_us']==400 and r['exposure']['gain']=='Gain_X4' and r['wait_ms']==240
        assert r['canonical_orientation']=='flip_v' and r['no_photometric_normalization']
        sat.append(r['saturation_fraction']);p99.append(r['p99'])
    summary[stage]={'count':len(ids),'phase_sha256':phase,'max_saturation_fraction':max(sat),'min_p99':min(p99),'mean_p99':sum(p99)/len(p99)}
tune=json.loads((root/'runs/readout50_trainonly_aligned/report.json').read_text())
assert tune['source_sha256']==report['checkpoint_sha256']
assert tune['baseline_replay_max_error']<1e-5 and tune['protected_unchanged']
assert not tune['test_used_for_training_or_selection']
assert not set(tune['fit_ids'])&set(tune['validation_ids'])
assert not (set(tune['fit_ids'])|set(tune['validation_ids']))&set(tune['test_ids'])
assert len(tune['fit_ids'])==1200 and len(tune['validation_ids'])==400 and len(tune['test_ids'])==800
best=root/'runs/readout50_trainonly_aligned/best.pt'
assert hashlib.sha256(best.read_bytes()).hexdigest()==tune['best_sha256']
audit={'status':'complete','samples':2400,'ccd_count':14400,'stages':summary,'physical_to_physical':report['physical_to_physical'],'train_only_adaptation':True,'best_sha256':tune['best_sha256'],'raw_ccd_retained_at':str(run)}
(root/'runs/sixhour_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit))
