"""Use the unchanged original cost/audit body with an explicit native-GPU proof.

Only the proof-platform validator and identity label are adapted. No inference,
timing, energy, precision, ordering, sample count or stage code is replaced.
Legacy cpu_contract fields remain path/SHA aliases, explicitly NOT CPU claims.
"""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from gpu_numerical_equivalence_20261025 import KIND,t12_proof,lgvq_proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=('t12','lgvq'),required=True)
    p.add_argument('--audit',action='store_true');p.add_argument('--folder',type=Path);p.add_argument('--audit-output',type=Path)
    args,remaining=p.parse_known_args()
    if args.task=='t12':
        import measure_t12_cost as worker
        validator=t12_proof
    else:
        import measure_lgvq_cost as worker
        validator=lgvq_proof
    worker.proof_contract=validator
    if args.audit:
        if args.task=='t12':import audit_t12_cost as auditor
        else:import audit_lgvq_cost as auditor
        auditor.proof_contract=validator
        if args.folder is None or args.audit_output is None or args.audit_output.exists():raise ValueError('New explicit independent audit output required')
        result=auditor.audit(args.folder)
        for key in ('CPU_contract_sha256','cpu_contract_sha256'):
            if key in result:result['numerical_contract_sha256']=result.pop(key)
        result['numerical_proof_kind']=KIND;result['validation_platform']='same native A100 GPU, not CPU'
        worker.write(args.audit_output,result);return
    save=worker.write
    def labeled_write(path,value):
        if Path(path).name=='identity.json':
            value.update(numerical_proof_kind=KIND,validation_platform='same native A100 GPU, not CPU',
                numerical_contract=value['cpu_contract'],numerical_contract_sha256=value['cpu_contract_sha256'],
                legacy_cpu_contract_fields_are_path_SHA_aliases_not_CPU_execution_claims=True)
        save(path,value)
    worker.write=labeled_write
    sys.argv=[str(Path(worker.__file__)),*remaining];worker.main()


if __name__=='__main__':main()
