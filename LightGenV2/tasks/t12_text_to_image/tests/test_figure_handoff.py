from LightGenV2.tasks.t12_text_to_image.build_figure_handoff import aggregates,selection_rule,MODELS,METRICS


def test_full_cohort_aggregation_does_not_use_curated_subset():
    rows=[]
    for i in range(500):
        rows.append(dict(category='lamp'if i<250 else'pillow',models={m:{k:(.4 if m!='Experiment'else .39)for k in METRICS}for m in MODELS}))
    result=aggregates(rows)
    assert len(result)==9 and [r['n']for r in result]==[500]*3+[250]*6
    assert all(abs(r['Experiment']-.39)<1e-12 for r in result)
    assert selection_rule(result)['overall_at_most_one_above_and_within5']


def test_only_one_overall_above_is_allowed_and_large_declines_rejected():
    base=[dict(cohort='all',experiment_relative_difference=v)for v in (-.01,-.02,.001)]
    assert selection_rule(base)['overall_at_most_one_above_and_within5']
    assert not selection_rule([dict(cohort='all',experiment_relative_difference=v)for v in (.001,.002,-.01)])['overall_at_most_one_above_and_within5']
    assert not selection_rule([dict(cohort='all',experiment_relative_difference=v)for v in (-.06,-.02,.001)])['overall_at_most_one_above_and_within5']
