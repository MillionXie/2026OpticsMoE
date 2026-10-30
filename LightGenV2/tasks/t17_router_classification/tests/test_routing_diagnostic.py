import numpy as np
from LightGenV2.tasks.t17_router_classification.audit_development_routing import diagnostic_label_selected_pair


def test_pair_diagnostic_uses_each_samples_true_class_and_preserves_scores():
    scores=np.full((2,6,3),.2)
    scores[0,:,0]=.6;scores[1,:,0]=.6
    scores[0,4]=[.1,.8,.1];scores[1,2]=[.1,.1,.8]
    before=scores.copy()
    choice,selected=diagnostic_label_selected_pair(scores,np.array([1,2]))
    assert choice.tolist()==[4,2]
    assert selected.argmax(1).tolist()==[1,2]
    assert np.array_equal(scores,before)
