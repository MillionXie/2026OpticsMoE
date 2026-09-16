import pytest
from LightGenV2.tasks.t04_semantic_interaction.lab_adaptation import (
    stratified_split, trainable_readout_name, selection_key)


def test_split_is_balanced_disjoint_and_deterministic():
    rows=[dict(sample_id=f'{task}_{i}',task=task) for task in ('add','replace','move','remove') for i in range(250)]
    train,held=stratified_split(rows)
    assert (train,held)==stratified_split(rows)
    assert len(train)==800 and len(held)==200 and not set(train)&set(held)
    assert set(train)|set(held)==set(range(1000))
    for task in ('add','replace','move','remove'):
        assert sum(rows[i]['task']==task for i in train)==200
        assert sum(rows[i]['task']==task for i in held)==50
    rows[1]['sample_id']=rows[0]['sample_id']
    with pytest.raises(ValueError):stratified_split(rows)


def test_upstream_language_summary_must_not_train():
    for name in ('position_readout.weight','language_pool.1.weight','task_head.1.weight'):
        assert not trainable_readout_name(name)
    for name in ('post_film.weight','coordinate_projection.bias','editor.0.weight','decoder.edit_head.bias'):
        assert trainable_readout_name(name)


def test_selection_prioritizes_scene_not_cherry_picked_changed_accuracy():
    def m(scene,changed):return {'overall':dict(scene_exact_match=scene,changed_cell_accuracy=changed,edit_grid_iou=.8,object_f1=.9)}
    assert selection_key(m(.7,.8))>selection_key(m(.6,.99))


def test_training_changes_only_downstream_head_and_cached_metric_pipeline():
    try:
        import torch
    except (ImportError,OSError) as exc:
        pytest.skip(str(exc))
    from LightGenV2.tasks.t04_semantic_interaction.shared_readout import SharedGridReadout
    from LightGenV2.tasks.t04_semantic_interaction.lab_adaptation import state_digest,evaluate_head,predict_head
    head=SharedGridReadout()
    for name,p in head.named_parameters():p.requires_grad_(trainable_readout_name(name))
    frozen=[k for k in head.state_dict() if not trainable_readout_name(k)]
    before=state_digest(head,frozen); trained=state_digest(head)
    x=torch.randn(2,192,14,14); c=torch.randn(2,192)
    o=predict_head(head,x,c)
    (o['category_logits'].square().mean()+o['edit_logits'].square().mean()).backward()
    opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=.001)
    opt.step()
    assert state_digest(head,frozen)==before and state_digest(head)!=trained
    labels=[dict(source_grid=torch.zeros(1,6,6,dtype=torch.long),target_grid=torch.zeros(1,6,6,dtype=torch.long),
                 edit_grid=torch.ones(1,6,6,dtype=torch.bool),preserve_grid=torch.ones(1,6,6,dtype=torch.bool),
                 task_index=torch.tensor([0]),task=['add'],instruction=['test'],sample_id=[str(i)]) for i in range(2)]
    metrics,rows=evaluate_head(head,dict(spatial=x,condition=c,labels=labels),[0],[1],'cpu',2)
    assert metrics['all1000']['overall']['samples']==2
    assert metrics['adaptation800']['overall']['samples']==metrics['holdout200']['overall']['samples']==1
    assert len(rows)==2
