"""Prepare a fresh failed-layer retake from an independently passed parent.

Artifact namespaces only: no engineering copy, changed physical graph or lower
gate. Old failed-layer CCDs remain in their original namespace, never linked in.
"""
import argparse
from pathlib import Path

ATTEMPT='response_20261020'


def prepare(a):
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_strict_retake_20261020 import check_score,link_file
    torch.set_num_threads(2)
    assert not m.LOCK.exists()
    old=a.sequence/f'L{a.depth}_strict8pp_20261020'
    prefix={4:3,6:1}[a.depth];stage={4:'2',6:'0'}[a.depth]
    name=('1' if a.depth==4 else 'router')+'_spatial_response_retrain_20261020'
    prior=old/'fits'/name;r=m.read(prior/'report.json');p=m.read(prior/'protocol.json')
    audit=m.read(prior/'independent_audit.json')
    digest=check_score(r,a.depth,prefix)
    assert not r['selected_e0'],'No changed response-trained candidate: no identical-PT reshoot'
    assert audit['passed'] and audit['report_SHA256']==m.sha(prior/'report.json')
    assert audit['protocol_SHA256']==m.sha(prior/'protocol.json') and audit['selected_correct']==r['selected_correct']
    data,dm=m.data_for(a.sequence);hardware=m.session_hardware_digest(a.sequence)
    fields,identities=m.fields_for(old,prefix,data,dm,hardware)
    assert p['prefix']==prefix and p['prefix_SHA256']==identities and p['data_SHA256']==dm['sha256']
    previous=(['router']+[str(i) for i in range(a.depth)])[prefix-1]
    chain=m.validate_fit_ancestry(old,name,digest,prefix,m.read(old/'capture'/previous/'report.json')['PT_SHA256'],identities,dm['sha256'])
    session=m.session_for(a);assert not session.exists();session.mkdir();links=[]
    for physical in ['router']+[str(i) for i in range(prefix-1)]:
        source=old/'capture'/physical
        assert m.read(source/'independent_capture_audit.json')['passed']
        for file in source.rglob('*'):
            if file.is_file():link_file(file,session/'capture'/physical/file.relative_to(source),links)
    for node in chain:
        source=old/'fits'/node['fit']
        assert m.read(source/'independent_audit.json')['passed']
        for file in source.rglob('*'):
            if file.is_file():link_file(file,session/'fits'/node['fit']/file.relative_to(source),links)
    for name in ('initial.pt','source_reference.ts','lab_gate.json'):
        link_file(old/name,session/name,links)
    _,newidentities=m.fields_for(session,prefix,data,dm,hardware)
    assert identities==newidentities and not (session/'capture'/stage).exists()
    m.save(session/'reuse_manifest.json',dict(source_session=str(old),links=links,passed_prefix=prefix,
        preserved_old_failed_stage=str(old/'capture'/stage),failed_stage_CCD_reused=False,
        data_SHA256=dm['sha256'],hardware_SHA256=hardware,prefix_SHA256=identities,
        parent_fit=prior.name,parent_PT_SHA256=digest,parent_audit_SHA256=m.sha(prior/'independent_audit.json'),
        source_git=m.git_head(),source_SHA256=m.sha(__file__),engineering_copy=False,immutable_artifact_links=True))
    m.save(session/'prepared.json',dict(passed=True,reuse_manifest_SHA256=m.sha(session/'reuse_manifest.json'),
        first_retake=stage,source_git=m.git_head(),no_automatic_next_layer=True))
    print(m.json.dumps(dict(prepared=True,depth=a.depth,first_retake=stage,parent_fit=prior.name,
        parent_PT_SHA256=digest,native_correct=r['selected_correct'],failed_layer_CCD_reused=False,SDK_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--depth',type=int,choices=(4,6),required=True)
    a=p.parse_args();a.attempt=ATTEMPT;prepare(a)
