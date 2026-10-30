import numpy as np
from LightGenV2.tasks.t17_router_classification.test_exposed_data import metrics_from_scores
from LightGenV2.tasks.t17_router_classification.development_teacher_data import development_training_membership


def test_development_union_excludes_validation_pixels_without_editing_evaluation():
    data={'train':{'hashes':np.array(['a','b']),'ids':np.array(['train_a','train_b'])},
          'test':{'hashes':np.array(['b','c','d']),'ids':np.array(['test_b','test_c','test_d'])},
          'val':{'hashes':np.array(['d']),'ids':np.array(['val_d'])}}
    kept,removed=development_training_membership(data)
    assert kept.tolist()==[0,1,3]
    assert {r['reason'] for r in removed}=={'validation_pixel_duplicate','duplicate_training_pixel'}
    assert data['test']['ids'].tolist()==['test_b','test_c','test_d']


def test_development_metrics_count_all_original_samples_and_classes():
    p=np.array([[.8,.1,.1],[.6,.3,.1],[.1,.2,.7]])
    m=metrics_from_scores(np.array([0,1,2]),p)
    assert m['n']==3 and m['confusion_matrix']==[[1,0,0],[1,0,0],[0,0,1]]
    assert m['accuracy']==2/3 and m['balanced_accuracy']==2/3
