import torch
from LightGenV2.tasks.t08_abo_image_text_retrieval.text_to_image_baseline import metrics


def test_multiple_positive_metrics():
    report,order,first=metrics(torch.tensor([[4.,3.,2.,1.],[1.,2.,3.,4.]]),torch.tensor([0,1]),torch.tensor([0,1,0,1]))
    assert report['hit_at_1']==1
    assert report['recall_at_1']==.5
    assert abs(report['map']-5/6)<1e-7
    assert first.tolist()==[1.,1.]
