"""Explicit metric cohort admission; a new48-row protocol is not old DEV128."""


def expanded_identity(report,records=None):
    p=report.get('expanded_DEV_protocol')or{}
    if p.get('schema')!='scene_visible_anchor_DEV_preflight_v1':
        return 128,('historical_expanded',p['manifest_sha256'])
    sha=p.get('request_sha256','')
    if(p.get('prompt_profile')!='two_sentence_visible_anchor_v5' or p.get('contexts')!=48
       or p.get('reference_views')!=16 or p.get('TEST_opened')is not False
       or not p.get('balanced_counterfactual_training_data') or len(sha)!=64
       or any(c not in'0123456789abcdef'for c in sha)):
        raise ValueError('Only the completed canonical R2 full48 protocol is supported')
    if records is not None:
        if len(records)!=48 or len({r['reference_id']for r in records})!=16:
            raise ValueError('All sixteen references and forty-eight requests required')
        if {c:sum(r['category']==c for r in records)for c in ['lamp','pillow']}!={'lamp':24,'pillow':24}:
            raise ValueError('Both full categories required')
        for rid in {r['reference_id']for r in records}:
            rows=[r for r in records if r['reference_id']==rid]
            if(len(rows)!=3 or {r['condition_id']for r in rows}!={'0','1','2'}
               or len({r['input_sha256']for r in rows})!=1 or len({r['seed']for r in rows})!=1):
                raise ValueError('Preserve the same-RGB same-seed three-condition grid')
    return 48,('visible_anchor_v5',sha)
