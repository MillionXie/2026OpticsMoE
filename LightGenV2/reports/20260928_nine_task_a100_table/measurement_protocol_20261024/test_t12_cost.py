import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from measure_t12_cost import original_cost,proof_contract,original,check_platform,GPU,MAX_QUEUE_SECONDS,sha
from run_t12_cost import point_plan,predecessor_ready,input_proof


def test_missing_only_plan_does_not_recreate_completed_points():
    assert point_plan(['ours_repeat03','qwen_repeat02'])==[('ours_repeat03','ours',3),('qwen_repeat02','qwen',2)]
    for names in ([],['ours_repeat01','ours_repeat01'],['ours_repeat04'],['other_repeat01']):
        with pytest.raises(ValueError):point_plan(names)


def test_baseline_continuous_host_proxy_is_added_once():
    calls=[dict(input_start=i*.1,cpu_result_return=i*.1+.05,internal_gap_ms=0 if i==0 else 50) for i in range(200)]
    result=original_cost('qwen',calls,[],100.)
    assert result['seconds200']==pytest.approx(19.95)
    assert result['joules200']==pytest.approx(338.2*19.95+100.)
    assert result['charged_electronic_ms_total'] is None


def test_ours_retains_all199_gaps_but_not_excluded_parallel_residuals():
    calls=[dict(input_start=i*.1,cpu_result_return=i*.1+.05,internal_gap_ms=0 if i==0 else 50) for i in range(200)]
    stages=[dict(synchronized_wall_ms=2,charged=True),dict(synchronized_wall_ms=900,charged=False)]
    result=original_cost('ours',calls,stages,100.)
    assert result['charged_electronic_ms_total']==9952
    assert result['optical_ms_total']==pytest.approx(200*6*1.0447)
    assert result['joules200']==pytest.approx((104.230*9952+121.746*200*6*1.0447)/1000)
    with pytest.raises(ValueError):original_cost('ours',calls[:199],stages,0.)


def test_pending_and_stopped_cpu_jobs_never_authorize_gpu():
    assert predecessor_ready(dict(status='waiting_previous_queue'),{},None) is False
    with pytest.raises(RuntimeError):predecessor_ready(dict(status='stopped'),{},None)
    with pytest.raises(ValueError):predecessor_ready(dict(status='complete'),{},None)


@pytest.mark.parametrize('system',('ours','qwen'))
def test_actual_original_proof_and_no_target_required_before_loading(tmp_path,system):
    key='body_sha256' if system=='ours' else 'baseline_sha256'
    proof=dict(status='passed',no_gpu=True,sources={},checks=[dict(test_index=i,bitwise_equal=True,target_passed_to_model=False) for i in (200,2303)])
    proof[key]=original.BODY_SHA if system=='ours' else original.BASELINE_SHA
    if system=='qwen':proof['qwen_report']=dict(language_layers_retained=28,vision_tower_used=False,language_model_head_used=False)
    path=tmp_path/'contract.json';path.write_text(json.dumps(proof))
    assert proof_contract(path,system,check_sources=False)['status']=='passed'
    proof['checks'][0]['target_passed_to_model']=True;path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):proof_contract(path,system,check_sources=False)


def test_worker_keeps_original_noise_dtype_and_zero_warmup_contract():
    text=(HERE/'measure_t12_cost.py').read_text()
    assert 'baseline.forward(runtime,reference,tokens,index)' in text
    assert 'original.online_embeddings(processor,embedding,prompt,device)' in text
    assert 'torch.Generator().manual_seed(1042+index)' in text
    assert 'explicit_warmup_forwards=0' in text
    assert "board['energy_j']" in text and "board['gpu_energy_j']" not in text
    assert 'dataset[' not in text and 'PromptEmbeddingLookup' not in text


def test_exact_platform_and_one_shared_finite_deadline():
    assert check_platform(f'{GPU}, NVIDIA A100-PCIE-40GB, 250, 40960')['power_limit_w']==250
    for row in (f'{GPU}, NVIDIA A100-SXM4-80GB, 400, 81920',f'{GPU}, NVIDIA A100-PCIE-40GB, 200, 40960'):
        with pytest.raises(ValueError):check_platform(row)
    assert MAX_QUEUE_SECONDS==7200
    assert '<=MAX_QUEUE_SECONDS' in (HERE/'run_t12_cost.py').read_text()


def test_admission_only_reads_existing_independent_input_proof(tmp_path):
    manifest=tmp_path/'input_manifest.json';manifest.write_text('{}')
    value=dict(status='passed',samples=200,input_manifest_sha256=sha(manifest),no_model_or_GPU_forward=True)
    receipt=tmp_path/'server_independent_input_audit01.json';receipt.write_text(json.dumps(value))
    assert input_proof(tmp_path,manifest)==(receipt,sha(receipt))
    value['samples']=199;receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):input_proof(tmp_path,manifest)
    assert 'audit_inputs(args.inputs)' not in (HERE/'run_t12_cost.py').read_text()
