"""CPU evaluation of an already locked PT, with explicit VAL/TEST identity."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from .static_four import build_static_four,CONTRACT
from .diagnose_frontend_transfer import checked_features
from .train_unified_static import load_arrays
from .train_sensor_neutral import infer
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,tensor_sha


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('run','data','frontend','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--split',choices=('validation','test'),default='validation');a=p.parse_args();torch.set_num_threads(2)
    receipt=json.loads((a.run/'completion.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text());path=a.run/'best_checkpoint.pt';digest=sha(path)
    assert receipt['contract']==CONTRACT and digest==receipt['best_sha256'];cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['contract']==CONTRACT and cp['frontend_sha256']==receipt['frontend_SHA256']
    m=build_static_four(cp['architecture']);m.load_state_dict(cp['model'],strict=True);m.eval().requires_grad_(False);assert tensor_sha(m.display_gains)==protocol['fixed_gain_sha256'];before={n:tensor_sha(v) for n,v in m.state_dict().items()}
    data=load_arrays(a.data,a.split);front=checked_features(data,a.frontend,cp['frontend_sha256']);a.out.mkdir(parents=True,exist_ok=False);started=time.time();result={n:[] for n in ('ids','labels','domains','predictions','probabilities')};scores={}
    for start in range(0,len(data['labels']),400):
        end=min(start+400,len(data['labels']));part={n:v[start:end] for n,v in data.items()};r=infer(m,part,front[start:end],torch.device('cpu'))
        for name in result:result[name].extend(r[name])
        for domain,row in r['scores'].items():
            acc=scores.setdefault(domain,dict(n=0,correct=0,loss=0.,confusion_matrix=np.zeros((10,10),dtype=np.int64)));acc['n']+=row['n'];acc['correct']+=row['correct'];acc['loss']+=row['loss']*row['n'];acc['confusion_matrix']+=np.asarray(row['confusion_matrix'])
        put(a.out/'progress.json',dict(split=a.split,n_completed=end,n_expected=len(data['labels']),updated=time.time()))
    for row in scores.values():row['accuracy']=row['correct']/row['n'];row['loss']/=row['n'];row['confusion_matrix']=row['confusion_matrix'].tolist()
    result['scores']=scores;assert sha(path)==digest and all(tensor_sha(v)==before[n] for n,v in m.state_dict().items())
    reference=a.run/('selected_validation.json' if a.split=='validation' else 'selected_test.json');old=json.loads(reference.read_text()) if reference.exists() else None
    put(a.out/'report.json',dict(evaluation_split=a.split,checkpoint_SHA256=digest,frontend_SHA256=cp['frontend_sha256'],strict_CPU=True,selected_before_evaluation=True,model_reselected=False,parameter_updates=False,SDK=False,full_result=result,accuracy_delta_from_training_record={d:row['accuracy']-old['scores'][d]['accuracy'] for d,row in scores.items()} if old else None,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seconds=time.time()-started))


if __name__=='__main__':main()
