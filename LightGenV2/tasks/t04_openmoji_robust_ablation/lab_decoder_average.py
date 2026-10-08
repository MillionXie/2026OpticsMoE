"""One predeclared 50/50 decoder-weight average, same protected optical model.

No gradients, new layers, threshold search or hardware SDK calls. Reuses only
verified same-upstream feature caches; TEST metrics remain development metrics.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','manifest','first','second','cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing experiment')
    sys.path.insert(0,str(a.project.resolve()/'source'))
    import torch
    from LightGenV2.tasks.t04_openmoji_robust_ablation.lab_editor16_robust_chain import factory
    from LightGenV2.tasks.t04_openmoji_robust_ablation import lab_tune_g2_test as backend
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    row=json.loads(a.manifest.read_text())['groups'][0]
    backend.GROUPS={'g2':(row['weight'],row['sha256'])}
    backend.t=SimpleNamespace(build_model=factory(a.project.resolve(),row))
    _,model=backend.config(a.project.resolve(),torch.device('cpu'))
    torch.set_num_threads(4)
    base_protected=backend.protected_sha(model)
    payloads=[torch.load(path,map_location='cpu',weights_only=False) for path in (a.first,a.second)]
    prefix='shared_readout.decoder.'
    for payload in payloads:
        model.load_state_dict(payload['model'],strict=True)
        if backend.protected_sha(model)!=base_protected:raise ValueError('Different protected upstream: cannot average or reuse CCD')
    combined=copy.deepcopy(payloads[0])
    for key,value in combined['model'].items():
        if key.startswith(prefix):
            other=payloads[1]['model'][key]
            if value.shape!=other.shape or not value.is_floating_point():raise ValueError('Decoder state mismatch')
            combined['model'][key]=(value+other)*.5
        elif not torch.equal(value,payloads[1]['model'][key]):raise ValueError('Upstream tensors differ')
    data={scope:torch.load(a.cache/(scope+'_features.pt'),map_location='cpu',weights_only=False) for scope in ('train','test')}
    ids=[set(d['ids']) for d in data.values()]
    if ids[0]&ids[1] or any(len(set(d['ids']))!=len(d['ids']) for d in data.values()):raise ValueError('Dataset identities overlap')
    if len(data['test']['ids'])!=1000 or len(data['train']['ids'])!=1000:raise ValueError('Wrong original TRAIN/TEST1000')
    def evaluate(payload,scope):
        model.load_state_dict(payload['model'],strict=True)
        model.eval().requires_grad_(False)
        d=data[scope];meter=MetricAccumulator();records=[]
        with torch.inference_mode():
            for start in range(0,len(d['ids']),32):
                end=min(start+32,len(d['ids']));rows=d['rows'][start:end]
                y={k:torch.cat([r[k] for r in rows]) if torch.is_tensor(rows[0][k]) else sum([r[k] for r in rows],[]) for k in rows[0]}
                cat,edit=model.shared_readout.decoder(d['features'][start:end])
                samples,_,_=meter.update({'category_logits':cat,'edit_logits':edit,'task_logits':y['task_logits']},y)
                records.extend(samples)
        return meter.compute(),records
    baseline=[evaluate(v,'test')[0] for v in payloads]
    fit,_=evaluate(combined,'train');test,records=evaluate(combined,'test')
    a.output.mkdir(parents=True)
    combined['decoder_average']={'weights':[.5,.5],'development_only':True,'test_gradient':False}
    torch.save(combined,a.output/'best.pt')
    saved=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
    strict,_=evaluate(saved,'test')
    if strict!=test or backend.protected_sha(model)!=base_protected:raise ValueError('Strict saved PT replay differs')
    source_hashes=[hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.first,a.second)]
    report={'status':'complete','method':'one predeclared 50/50 original decoder average','train':fit,'test':test,
        'parent_tests':baseline,'strict_default_cpu_reload':True,'protected_sha256':base_protected,
        'protected_unchanged':True,'test_gradient':False,'architecture_unchanged':True,
        'parents_sha256':source_hashes,'best_sha256':hashlib.sha256((a.output/'best.pt').read_bytes()).hexdigest(),
        'selection':'TEST development; no independent generalization claim'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    (a.output/'test_samples.json').write_text(json.dumps(records))
    print(json.dumps({'parents':[m['overall']['changed_cell_accuracy'] for m in baseline],'averaged':test['overall']['changed_cell_accuracy'],'fit':fit['overall']['changed_cell_accuracy']}),flush=True)


if __name__=='__main__':main()
