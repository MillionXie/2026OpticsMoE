"""Evaluate locked validation-selected SONYC checkpoints on test once."""
import argparse,json
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,balanced_accuracy_score
from LightGenV2.tasks.t09_multimodal_matching.prepare import tokens
from LightGenV2.tasks.t09_multimodal_matching.model import OpticalOEO,TextEncoder
from LightGenV2.tasks.t09_multimodal_matching.run import evaluate,state_sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--run-dir',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
 d=json.loads((a.data/'manifest.json').read_text()); vocab=json.loads((a.data/'vocab.json').read_text()); rows=json.loads((a.data/'test_questions.json').read_text()); images=torch.tensor(np.load(a.data/'test_images.npz')['images'],device='cuda'); ids=torch.zeros(len(rows),32,dtype=torch.long,device='cuda')
 for i,r in enumerate(rows): ids[i,:len(tokens(r['question']))]=torch.tensor([vocab.get(w,1) for w in tokens(r['question'])],device='cuda')
 data={'images':images,'ids':ids,'index':torch.tensor([r['image_local'] for r in rows],device='cuda'),'labels':torch.tensor([r['label'] for r in rows],device='cuda'),'rows':rows}; out={}
 for arch in ['moe','d2nn']:
  run=a.run_dir if a.run_dir else next(a.runs.glob(f'*_{arch}_s17_20260918')); root=run/'fixed'/arch; ck=root/'best_checkpoint.pt'; cp=torch.load(ck,weights_only=False); front=TextEncoder(len(vocab),'fixed').cuda(); front.load_state_dict(torch.load(run/'fixed'/'frontend.pt',weights_only=False)['state']); front.eval(); model=OpticalOEO(arch,17,input_layout=cp.get('input_layout','left_right'),oeo_activation=cp.get('oeo_activation','centered_leaky_relu')).cuda(); model.load_state_dict(cp['model']); model.eval(); score,pred=evaluate(model,front,data,64); y=np.array([r['label'] for r in rows]); s=pred[:,1]; score.update(macro_auprc=float(average_precision_score(y,s)),auroc=float(roc_auc_score(y,s)),macro_f1=float(f1_score(y,s>0.5)),balanced_accuracy=float(balanced_accuracy_score(y,s>0.5))); out[arch]=score; np.savez_compressed(a.out/f'{arch}_test_predictions.npz',probabilities=pred,labels=y)
  del model,front,cp
 (a.out/'results.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
