import json

from LightGenV2.tasks.t19_resnet20_baselines.queue_remaining import AttachedWorker


def test_completed_orphan_is_collected_from_receipt_without_restarting_it(tmp_path):
    result=tmp_path/'result.json'
    result.write_text(json.dumps({'status':'complete'}))
    job={'pid':987654321,'name':'a','gpu_uuid':'GPU-a'}
    worker=AttachedWorker(job,result)
    assert worker.poll()==0 and worker.wait(timeout=1)==0


def test_missing_worker_without_success_receipt_is_failure(tmp_path):
    job={'pid':987654321,'name':'a','gpu_uuid':'GPU-a'}
    assert AttachedWorker(job,tmp_path/'missing.json').poll()==1
