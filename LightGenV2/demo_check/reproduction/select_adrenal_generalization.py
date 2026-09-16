"""Freeze candidates using validation only, before the evaluation entry opens test."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--confirmation-runs',type=Path,nargs='*',default=[]);p.add_argument('--reference-runs',type=Path,nargs='*',default=[]);a=p.parse_args();assert not a.out.exists()
    summaries=[];all_candidates={};locks={}
    for root in a.runs+a.confirmation_runs:
        results=read(root/'validation_results.json');metadata=read(root/'metadata.json');assert not (root/'test_results.json').exists(),'Do not select after viewing new tests'
        assert read(root/'status.json')['state']=='training_complete_test_not_read'
        locks[root.name]=dict(test_lock_sha256=sha(root/'test_lock.json'),metadata_sha256=sha(root/'metadata.json'),validation_results_sha256=sha(root/'validation_results.json'))
        if root in a.confirmation_runs:continue
        assert len(results)==6 and {x['seed'] for x in results}=={17}
        scores=[x['metrics']['val']['balanced_nll'] for x in results]
        summaries.append(dict(run=root.name,mean_validation_balanced_nll=sum(scores)/6,mean_validation_auroc=sum(x['metrics']['val']['auroc'] for x in results)/6,profile=metadata['config']))
        for x in results:
            # Do not count the deterministic router-only D2NN repetition as a new candidate.
            if x['variant'].startswith('d2nn') and 'router_temperature' in metadata['config']:continue
            all_candidates.setdefault(x['variant'],[]).append(dict(run=root.name,epoch=x['selected_epoch'],balanced_nll=x['metrics']['val']['balanced_nll'],auroc=x['metrics']['val']['auroc'],checkpoint_sha256=x['checkpoint_sha256']))
    selected=min(summaries,key=lambda x:(x['mean_validation_balanced_nll'],x['run']))
    obj=dict(time=datetime.now(timezone.utc).isoformat(),selection_rule='One shared configuration: lowest mean validation balanced NLL across both architectures and all three depths; no gap/test-based selection',selected_shared_configuration=selected['run'],candidate_configurations=summaries,architecture_specific_validation_winners={k:min(v,key=lambda x:(x['balanced_nll'],x['run'])) for k,v in all_candidates.items()},runs=locks,confirmation_runs=[p.name for p in a.confirmation_runs],test_status='Official test previously seen in earlier work; new test results not read during this optimization',selector_sha256=sha(Path(__file__)))
    obj['historical_validation_only_references']={}
    for root in a.reference_runs:
        assert not (root/'test_results.json').exists()
        obj['historical_validation_only_references'][root.name]=dict(metadata_sha256=sha(root/'metadata.json'),validation_results_sha256=sha(root/'validation_results.json'),test_lock_sha256=sha(root/'test_lock.json'),scope='Historical locked validation-only candidate; not retrained or used to select the new shared configuration')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(obj,indent=2),encoding='utf-8');print(json.dumps(obj,indent=2))

if __name__=='__main__':main()
