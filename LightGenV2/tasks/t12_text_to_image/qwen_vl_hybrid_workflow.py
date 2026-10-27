"""One-GPU, failure-stopping continuation AFTER the electronic baseline.

No process is killed, no physical optical path is operated, and no original
run/checkpoint is overwritten. All stage children must exit before the next.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def gpu_pids(uuid):
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']).decode().splitlines()
    return [int(row.split(',')[1]) for row in lines if row.startswith(uuid)]


def verify_feature_count(report, audit):
    """Reuse only a complete cache covering the explicitly recorded cohort."""
    expected = audit['pairs']['train'] + audit['pairs']['val']
    if not report['complete'] or report['count'] != expected:
        raise ValueError('Reused teacher cache does not cover the data cohort')
    return expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'data', 'qwen', 'sd', 'config', 'smoke-config', 'tail-config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--parent-workflow-pid', type=int, required=True)
    p.add_argument('--reuse-features', type=Path, help='Verified completed teacher feature cache; never recompute it')
    p.add_argument('--initial-hybrid', type=Path, help='Preserved trained macro best, including an interrupted recovery')
    p.add_argument('--input-manifest', type=Path, help='Raw RGB/text-only development inputs for matched diagnostic timing')
    a = p.parse_args()
    if (a.output/'execution.json').exists():
        raise ValueError('Do not start a duplicate workflow')
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.config.read_text()); gpu = cfg['gpu_uuid']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu or cfg['alpha_minimum'] != .6:
        raise ValueError('GPU or alpha authorization differs')
    stages, children, started = [], [], time.time()
    teacher_features = a.reuse_features or a.output/'teacher_cache'
    write(a.output/'execution.json', {'pid': os.getpid(), 'argv': sys.argv, 'started_unix': started,
          'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'gpu_uuid': gpu, 'alpha_minimum': .6,
          'wait_for_completed_electronic_baseline': True, 'parent_workflow_pid': a.parent_workflow_pid})
    module = 'LightGenV2.tasks.t12_text_to_image.'
    def run(label, tail, gpu_job=True):
        if gpu_job and gpu_pids(gpu):
            raise RuntimeError('Authorized GPU occupied; do not stop or overlap another process')
        cmd = [sys.executable, '-m', module+tail[0], *tail[1:]]
        with (a.output/(label+'.log')).open('ab') as log:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children.append(child.pid)
            write(a.output/'progress.json', {'stage': label, 'pid': child.pid, 'own_children': children, 'command': cmd})
            rc = child.wait()
        stages.append({'stage': label, 'pid': child.pid, 'returncode': rc, 'command': cmd})
        write(a.output/'stages.json', stages)
        if rc:
            raise RuntimeError(label+' failed; inspect its log')
    def hybrid_args(stage, config, out, initial=None):
        cmd = ['qwen_vl_hybrid_run', '--stage', stage, '--config', str(config), '--data', str(a.data),
             '--parent', str(a.parent), '--qwen', str(a.qwen), '--sd', str(a.sd), '--output', str(out),
             '--features', str(teacher_features)]
        if initial:
            cmd += ['--initial', str(initial)]
        return cmd
    def timing(label, hybrid=None, tail=None):
        cmd = ['qwen_vl_latency', '--checkpoint', str(a.parent/'best_checkpoint.pt'),
              '--manifest', str(a.input_manifest or a.parent/'raw_inputs.json'), '--qwen', str(a.qwen), '--sd', str(a.sd),
              '--gpu-uuid', gpu, '--rounds', '3', '--memory-fraction', '.85', '--output', str(a.output/label)]
        if hybrid:
            cmd += ['--hybrid-checkpoint', str(hybrid/'best_checkpoint.pt')]
        if tail:
            cmd += ['--tail-checkpoint', str(tail/'best_checkpoint.pt')]
        run(label, cmd)
    try:
        while not (a.parent/'workflow_report.json').exists():
            failure = a.parent/'workflow_failure.json'
            if failure.exists():
                raise RuntimeError('Electronic baseline workflow failed; diagnose before continuation')
            proc = Path(f'/proc/{a.parent_workflow_pid}/cmdline')
            if not proc.exists() or 'qwen_vl_' not in proc.read_bytes().decode(errors='replace'):
                raise RuntimeError('Electronic workflow exited without a completion report')
            write(a.output/'progress.json', {'stage': 'waiting_electronic_baseline', 'gpu_used_by_continuation': False,
                   'observed_unix': time.time(), 'parent_progress': json.loads((a.parent/'progress.json').read_text()) if (a.parent/'progress.json').exists() else None})
            time.sleep(30)
        done = json.loads((a.parent/'workflow_report.json').read_text())
        if not done['complete']:
            raise RuntimeError('Baseline report is incomplete')
        parent_report = json.loads((a.parent/'report.json').read_text())
        if not parent_report['complete'] or parent_report['best_step'] <= 0:
            raise RuntimeError('No completed selected electronic checkpoint')
        # Completion is written just before the parent's release receipt; wait
        # for its GPU children rather than racing the final raw-input process.
        for _ in range(12):
            if not gpu_pids(gpu):
                break
            time.sleep(5)
        if shutil.disk_usage(a.output).free < 20*2**30:
            raise RuntimeError('Need 20 GiB for feature caches and retained best/last')
        timing('electronic_latency')
        if a.reuse_features:
            feature_report = json.loads((teacher_features/'report.json').read_text())
            verify_feature_count(feature_report, json.loads((a.data/'data_audit.json').read_text()))
            write(a.output/'reuse_identity.json', {'teacher_features': str(teacher_features),
                'teacher_features_sha256': feature_report['sha256'], 'initial_hybrid': str(a.initial_hybrid) if a.initial_hybrid else None})
        else:
            run('teacher_cache', hybrid_args('cache', a.config, teacher_features))
        smoke_features = a.initial_hybrid or a.output/'smoke_features'
        if not a.initial_hybrid:
            run('smoke_features', hybrid_args('distill', a.smoke_config, smoke_features))
        run('smoke_joint', hybrid_args('joint', a.smoke_config, a.output/'smoke_joint', smoke_features))
        for folder in ([a.output/'smoke_joint'] if a.initial_hybrid else [smoke_features, a.output/'smoke_joint']):
            report = json.loads((folder/'report.json').read_text())
            if not report['complete'] or report['best_step'] <= 0:
                raise RuntimeError('Real-backward smoke incomplete')
        features, joint, tail = (a.output/x for x in ('feature_distillation', 'hybrid_recovery', 'tail_quality'))
        if not a.initial_hybrid:
            run('feature_distillation', hybrid_args('distill', a.config, features))
        run('hybrid_recovery', hybrid_args('joint', a.config, joint, a.initial_hybrid or features))
        run('hybrid_review', ['qwen_vl_hybrid_review', '--run', str(joint)], gpu_job=False)
        timing('hybrid_latency', joint)
        tail_smoke_config = a.output/'tail_smoke_config.json'
        tail_cfg = json.loads(a.tail_config.read_text()); tail_cfg.update(smoke=True, steps=2, batch_size=2, warmup_steps=0, eval_every=2)
        write(tail_smoke_config, tail_cfg)
        def tail_cmd(config, out):
            return ['qwen_vl_tail_run', '--config', str(config), '--data', str(a.data), '--parent', str(a.parent),
                  '--hybrid', str(joint), '--qwen', str(a.qwen), '--sd', str(a.sd), '--output', str(out)]
        run('tail_smoke', tail_cmd(tail_smoke_config, a.output/'tail_smoke'))
        smoke = json.loads((a.output/'tail_smoke/report.json').read_text())
        if not smoke['complete'] or smoke['best_step'] <= 0:
            raise RuntimeError('Tail real-backward smoke incomplete')
        run('tail_quality', tail_cmd(a.tail_config, tail))
        run('tail_review', ['qwen_vl_hybrid_review', '--run', str(tail)], gpu_job=False)
        run('tail_raw_prompt_review', ['qwen_vl_scene_review', '--parent', str(a.parent),
            '--data', str(a.data), '--qwen', str(a.qwen), '--sd', str(a.sd),
            '--hybrid', str(joint), '--tail', str(tail), '--gpu-uuid', gpu,
            '--output', str(a.output/'tail_raw_prompt_review'),
            '--label', 'LG postfree + compressed tail candidate; semantic acceptance pending'])
        timing('tail_latency', joint, tail)
        baseline_time = json.loads((a.output/'electronic_latency/report.json').read_text())
        tail_time = json.loads((a.output/'tail_latency/report.json').read_text())
        optical = tail_time['optical_timing_model']['user_scope_serial_cuda_span_substitution_ms']['mean']
        baseline = baseline_time['first_tf_to_gpu_rgb_stream_ms']['mean']
        tail_report = json.loads((tail/'report.json').read_text())
        write(a.output/'report.json', {'complete': True, 'baseline_checkpoint_sha256': parent_report['checkpoint_sha256'],
              'tail_quality_gate': tail_report['quality_gate'], 'user_scope_cuda_substitution_estimated_speedup': baseline/optical,
              'speedup_is_physical_measurement': False, 'final_end_to_end_dag_audit_pending': True,
              'quality_accepted': False, 'stages': stages, 'alpha_minimum': .60,
              'evaluation_label': 'development_test', 'independent_test_claim': False,
              'note': 'Pipeline completion is not the 5%/2x acceptance. Review task images and per-mode gates; if needed add E/G capacity or further tail compression, then do final timing audit.'})
        write(a.output/'progress.json', {'stage': 'complete_pending_quality_review', 'finished_unix': time.time()})
    except Exception:
        write(a.output/'failure.json', {'traceback': traceback.format_exc(), 'stages': stages})
        write(a.output/'progress.json', {'stage': 'failed', 'finished_unix': time.time()})
        raise
    finally:
        active = gpu_pids(gpu)
        own = [pid for pid in active if pid in children]
        write(a.output/'gpu_release_receipt.json', {'own_children': children, 'own_remaining_gpu_pids': own,
              'own_gpu_released': not own, 'other_processes_untouched': True, 'finished_unix': time.time()})


if __name__ == '__main__':
    main()
