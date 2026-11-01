import numpy as np
from LightGenV2.tasks.t17_router_classification.analyze_f_error_curriculum import error_summary,curriculum_weights


def test_full_modality_error_counts_and_pairs_are_reconstructible():
    labels=np.array([0,1,0,1]);pred=np.array([0,0,1,1]);scores=np.eye(2)[pred]*.8+.1
    result=error_summary(scores,labels,np.array([0,0,1,1]),['a','b'])
    assert result['RGB']['errors']==result['SAR']['errors']==1
    assert result['RGB']['largest_confusions'][0]==dict(truth='b',prediction='a',count=1)
    assert result['SAR']['largest_confusions'][0]==dict(truth='a',prediction='b',count=1)
    assert np.array_equal(curriculum_weights(pred,labels),[1,3,3,1])
