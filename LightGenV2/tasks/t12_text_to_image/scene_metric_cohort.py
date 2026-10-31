"""Explicit metric cohort admission; a new48-row protocol is not old DEV128."""


def expanded_identity(report,records=None):
    p=report.get('expanded_DEV_protocol')or{}
    trained=p.get('schema')=='scene_paper_scene_trained_DEV_v1'
    paper=trained or p.get('schema')=='scene_paper_scene_DEV_preflight_v1'
    if not paper and p.get('schema')!='scene_visible_anchor_DEV_preflight_v1':
        return 128,('historical_expanded',p['manifest_sha256'])
    sha=p.get('request_sha256','')
    profile='two_sentence_product_scene_v7'if paper else'two_sentence_visible_anchor_v5'
    valid_lineage=(p.get('prior_model_on_new_prompts_not_retrained_model')is True
                   and p.get('model_trained_on_revised_dataset')is False and p.get('old_results_not_relabelled')is True)if paper else p.get('balanced_counterfactual_training_data')
    if trained:
        pair=p.get('paired_dataset_sha256','')
        valid_lineage=(p.get('model_trained_on_revised_dataset')is True and p.get('old_results_not_relabelled')is True and p.get('new_targets_reviewed')is True and len(pair)==64 and all(c in'0123456789abcdef'for c in pair))
    if(p.get('prompt_profile')!=profile or p.get('contexts')!=48
       or p.get('reference_views')!=16 or p.get('TEST_opened')is not False
       or not valid_lineage or len(sha)!=64
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
    # Training histories differ across peers; comparison still requires identical
    # complete public RGB/prompt/seed grids, checked by the scorer before metrics.
    return 48,('paper_scene_v7_same_requests'if paper else'visible_anchor_v5',sha)
