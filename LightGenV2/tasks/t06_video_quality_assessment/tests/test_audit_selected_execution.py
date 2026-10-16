from LightGenV2.tasks.t06_video_quality_assessment.audit_selected_execution import ranks,pearson,work


def test_tied_ranks_and_correlations():
    assert ranks([3,1,1,2])==[4,1.5,1.5,3]
    assert pearson(ranks([1,2,3]),ranks([1,2,3]))==1
    assert pearson([1,2,3],[3,2,1])==-1


def test_independent_attention_work_not_context_join():
    one={'op':'scaled_dot_product_attention','tensors':[{'shape':[1,4,32,16],'dtype':'bf16'}]*3}
    grouped={'op':'scaled_dot_product_attention','tensors':[{'shape':[3,4,32,16],'dtype':'bf16'}]*3}
    joined={'op':'scaled_dot_product_attention','tensors':[{'shape':[1,4,96,16],'dtype':'bf16'}]*3}
    assert work([one]*3)==work([grouped])
    assert work([one]*3)!=work([joined])
