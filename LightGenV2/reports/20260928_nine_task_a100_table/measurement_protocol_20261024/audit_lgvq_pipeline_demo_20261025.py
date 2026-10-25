"""Local read-only review of sealed LG Temporal demo; no Torch/model/network."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
GPU = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-5):
        raise AssertionError((a, b))


def union_seconds(intervals):
    merged = []
    for a, b in sorted(intervals):
        assert a <= b
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b-a for a, b in merged)


def overlap_seconds(intervals, a, b):
    return union_seconds([(max(x, a), min(y, b)) for x, y in intervals if min(y, b) > max(x, a)])


def critical_parts(run):
    cursor = run['actual_software_E2E_start']
    parts = defaultdict(float)
    for field in run['records']:
        points = [cursor, field['cpu_ready'], field['front_enqueued'], field['gpu_done'], field['cpu_result_return']]
        assert all(b >= a for a, b in zip(points, points[1:]))
        for name, a, b in zip(('wait_input_and_schedule', 'front_host_to_enqueued', 'body_and_sync', 'CPU_result'), points, points[1:]):
            parts[name] += b-a
        cursor = points[-1]
    close(cursor, run['actual_software_E2E_end'])
    close(sum(parts.values()), run['actual_software_E2E_seconds'])
    return dict(parts)


def audit(root, baseline):
    receipt = read(root/'backup_receipt_complete01.json')
    for item in receipt['verified_files']:
        path = root/item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['path']
    suite, status, identity, summary = [read(root/n) for n in ('suite_status.json', 'status.json', 'identity.json', 'summary.json')]
    assert suite['status'] == status['status'] == summary['status'] == 'complete'
    assert suite['mps_released'] and summary['Qwen_not_executed'] and summary['demo_only']
    assert identity['videos'] == 40 and identity['fields'] == 3 and identity['executed_slots'] == 48
    assert identity['gpu_uuid'] == GPU and identity['mps_percent'] == 100
    # Git blob bytes, not a CRLF Windows checkout, are the published Linux source identity.
    for path, digest in identity['sources'].items():
        blob = subprocess.check_output(['git', 'show', identity['source_commit']+':'+path], cwd=REPO)
        assert hashlib.sha256(blob).hexdigest() == digest, path
    reference = REPO/identity['input_reference'].split('/2026OpticsMoE/', 1)[1]
    assert sha(reference) == identity['input_reference_sha256']
    original_inputs = read(reference)['input_files']
    files = read(root/'input_files.json')
    assert len(files) == 40 and len({r['sample_id'] for r in files}) == 40
    for row in files:
        assert row['sha256'] == original_inputs[row['sample_id']]['sha256']
    checks = read(root/'consistency.json')
    assert len(checks) == 1 and checks[0]['batch_videos'] == 16 and checks[0]['passed']
    assert [c['field'] for c in checks[0]['checks']] == [0, 2]
    for c in checks[0]['checks']:
        assert c['full_field_output_equal'] and c['max_output_abs'] == 0
        for value in c['input_checks'].values():
            assert value['equal'] and value['original_dtype'] == value['candidate_dtype'] and value['max_abs'] == 0
        assert c['input_checks']['vision_tokens']['shape'] == [1,16,4,49,1024]
    load = read(root/'loading_process_audit.json')
    assert load and all(not a['foreign_gpu_processes'] and not a['foreign_mps_clients'] for a in load)
    process = read(root/'process_audit.json')
    assert process
    worker = suite['worker_pid']
    for obs in process:
        platform = obs['platform']
        assert platform['uuid'] == GPU and platform['name'] == 'NVIDIA A100-PCIE-40GB' and platform['power_limit_w'] == 250
        assert set(obs['clients']) <= {worker}
        assert all(int(row[0]) in set(obs['servers']) | {worker} and row[1].strip() == GPU for row in obs['processes'])
    base = read(baseline)
    points = [p for p in base['points'] if p['system'] == 'Qwen' and p['cap'] == 50]
    assert len(points) == 3 and sum(p['valid'] for p in points) == 600
    baseline16 = sum(p['original_ms'] for p in points)*16/600
    close(baseline16, summary['Qwen50_frozen_E2E16_ms'])
    output_reference = None
    groups = defaultdict(list)
    reviewed = []
    for brief in summary['runs']:
        run = read(root/(brief['label']+'.json'))
        assert {k:v for k,v in run.items() if k != 'records'} == brief
        assert len(run['records']) == 3 and run['videos'] == 40 and run['executed_slots'] == 48
        valid_indices, pid_intervals, stage_sums = [], defaultdict(list), defaultdict(float)
        future_overlap, cpu_work = 0., defaultdict(float)
        for i, field in enumerate(run['records']):
            expected = list(range(i*16, min(i*16+16, 40)))
            if i == 2:
                expected += list(range(8))
            assert field['field'] == i and field['indices'] == expected
            assert field['valid'] == ([True]*16 if i < 2 else [True]*8+[False]*8)
            assert field['sample_ids'] == [files[j]['sample_id'] for j in expected]
            valid_indices += [j for j, v in zip(expected, field['valid']) if v]
            assert len(field['CPU_preparation']) == 16
            for task, sample in zip(field['CPU_preparation'], field['sample_ids']):
                assert task['sample_id'] == sample
                t = [task[k] for k in ('start', 'decode_end', 'processor_end', 'end')]
                assert run['actual_software_E2E_start'] <= t[0] <= t[1] <= t[2] <= t[3] <= field['cpu_ready']
                for name, a, b in zip(('decode', 'processor', 'quality14'), t, t[1:]):
                    cpu_work[name] += b-a
                pid_intervals[task['pid']].append((t[0], t[-1]))
            assert sum(s['family'] == 'optical_simulation_FFT' for s in field['stages']) == 6
            for stage in field['stages']:
                assert stage['field'] == i and stage['cuda_event_ms'] >= 0
                assert field['cpu_ready'] <= stage['enqueue_start'] <= stage['enqueue_end'] <= field['gpu_done']
                stage_sums[stage['stage']] += stage['cuda_event_ms']
            if i:
                previous = run['records'][i-1]
                future_overlap += overlap_seconds([(t['start'], t['end']) for t in field['CPU_preparation']], previous['cpu_ready'], previous['gpu_done'])
        assert valid_indices == list(range(40)) and len(pid_intervals) <= run['workers']
        for intervals in pid_intervals.values():
            ordered = sorted(intervals)
            assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:]))
        if run['pipeline']:
            assert future_overlap > 0, 'No observed next-field CPU overlap'
        else:
            close(future_overlap, 0)
        with np.load(root/(run['label']+'.npz'), allow_pickle=False) as data:
            array, valid = data['prediction'], data['valid']
            assert array.shape == (3,16) and np.isfinite(array).all()
            assert valid.tolist() == [True]*40+[False]*8
            if output_reference is None:
                output_reference = array.copy()
            else:
                assert np.array_equal(output_reference, array)
        parts = critical_parts(run)
        duration = run['actual_software_E2E_end']-run['actual_software_E2E_start']
        close(duration, run['actual_software_E2E_seconds'])
        close(duration*400, run['actual_software_E2E16_ms'])
        close(duration*1000/160, run['actual_software_equivalent_frame_ms'])
        elapsed_after_first = run['records'][-1]['cpu_result_return']-run['records'][0]['cpu_result_return']
        close(elapsed_after_first*500, run['post_first_field_cadence16_ms'])
        result = dict(label=run['label'], workers=run['workers'], pipeline=run['pipeline'], valid_videos=40,
            E2E16_ms=duration*400, equivalent_frame_ms=duration*1000/160, Qwen50_reference_ratio=baseline16/(duration*400),
            nonoverlapping_critical_parts16_ms={k:v*400 for k,v in parts.items()},
            next_field_CPU_overlapping_previous_compute_window_ms=future_overlap*1000,
            CPU_work_sum_seconds_NOT_elapsed=dict(cpu_work),
            stage_event_totals_ms_NOT_pure_kernel_or_critical_time=dict(stage_sums),
            actual_16slot_field_cadence_after_first_ms=elapsed_after_first*500,
            valid16_throughput_time_after_first_ms=elapsed_after_first*1000*16/24)
        reviewed.append(result)
        groups[(run['workers'], run['pipeline'])].append(result)
    assert len(reviewed) == 6
    grouped = []
    for (workers, pipeline), rows in groups.items():
        assert len(rows) == 2
        mean16 = sum(r['E2E16_ms'] for r in rows)/2
        grouped.append(dict(workers=workers,pipeline=pipeline,repeats16_ms=[r['E2E16_ms'] for r in rows],
            mean16_ms=mean16,equivalent_frame_ms=mean16/64,Qwen50_reference_ratio=baseline16/mean16,
            mean_critical_parts16_ms={k:sum(r['nonoverlapping_critical_parts16_ms'][k] for r in rows)/2 for k in rows[0]['nonoverlapping_critical_parts16_ms']},
            mean_next_field_overlap_ms=sum(r['next_field_CPU_overlapping_previous_compute_window_ms'] for r in rows)/2))
    return dict(status='passed',review_kind='offline_saved_intervals_and_receipt_not_model_rerun',
        receipt_verified_files=len(receipt['verified_files']),published_source_blobs_verified=len(identity['sources']),
        input_files_verified_against_qualified_reference=40,consistency_evidence_fields=[0,2],
        timed_full_fields=18,timed_valid_video_calls=240,timed_executed_slots=288,outputs_equal_across_runs=True,
        native_front_batch_videos=16,frames_per_field=64,guard_records=len(process),Qwen_not_rerun=True,
        Qwen50_reference16_ms=baseline16,hundredfold_budget16_ms=baseline16/100,
        comparison_caveat='LG resident demo first40/two rounds; Qwen frozen first200/three rounds. Exploratory, not a replacement matched formal measurement.',
        notes=['E2E includes initial input dispatch, original decode/processor/quality, IPC, CPU concatenation/H2D, tokenization/embedding, full original software body and CPU output.',
            'Per-stage CUDA Event spans may include host submission gaps; input_pixels_H2D contains CPU concatenation and transfer, not pure device-copy bandwidth.',
            'Stage sums and fixed optical projection are not physical optical end-to-end or pipeline critical-path measurement.',
            'Only fixed model-geometry position interpolation is reused; per-sample pixels/features are recomputed in all timing runs.',
            'After-first cadence has 32 executed slots but only24 valid videos; valid16 normalization is separate, not request latency.'],
        groups=grouped,runs=reviewed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=audit(a.root,a.baseline)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'independent_demo_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['LGVQ Temporal并行与流水线demo：独立离线核验通过',
        '每场16视频×4帧＝64帧；首40个真实视频，执行3整场48槽，尾8补齐成本全计。',
        '每配置两轮；模型常驻、CPU处理器初始化完成后开始。并非正式首200×3的新测量。',
        '以下时间从首次输入提交到最后CPU结果，按40有效视频折合16视频，含启动和收尾。',
        '完整软件实现包含FFT等模拟操作；不是物理光路端到端新实测。',
        'Qwen50沿用原首200×3完整端到端，未重测。样本规模/常驻执行条件不同，倍率是探索性对照。',
        f"Qwen50参考：{result['Qwen50_reference16_ms']:.6f} ms/16视频；百倍LG预算：{result['hundredfold_budget16_ms']:.6f} ms。",'']
    for g in result['groups']:
        lines += [f"{g['workers']}路CPU准备 / {'与下一场准备重叠' if g['pipeline'] else '各场顺序'}：两轮 {g['repeats16_ms'][0]:.3f}、{g['repeats16_ms'][1]:.3f} ms；均值{g['mean16_ms']:.3f} ms/16视频，等效单帧{g['equivalent_frame_ms']:.3f} ms；Qwen50/LG={g['Qwen50_reference_ratio']:.4f}倍。"]
        names={'wait_input_and_schedule':'等待当前场输入就绪及调度（不能当作CPU累计工作量）','front_host_to_enqueued':'输入就绪至前端提交结束（含CPU拼接、传输、前端提交、实际分词等）','body_and_sync':'前端提交结束至完整网络GPU同步完成','CPU_result':'结果回到CPU'}
        lines += ['  '+names[k]+f"：{v:.3f} ms/16视频" for k,v in g['mean_critical_parts16_ms'].items()]
        lines += [f"  下一场CPU准备与上一场计算窗口重叠：两轮平均{g['mean_next_field_overlap_ms']:.3f} ms（各场并集，不累加多个CPU线程）。",'']
    lines += ['核验：两个真实完整场（包括尾补齐场）原输入/全部16输出逐位一致；六轮输出全相同。',
        '前端确实一次处理16视频64帧，没有偷换成单视频后再除16。位置缓冲只依赖固定模型几何，不缓存样本。',
        '瓶颈排查：CPU工作量总和与墙钟等待必须分开；CPU输入拼接及传输作用域也不是纯GPU核时间。',
        '此结果没有达到百倍，不能用删除解码、仅GPU段、稳态节拍或固定光周期来冒称端到端百倍。']
    (a.output/'LGVQ_Temporal_流水线demo核验说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sources=[a.root/r['path'] for r in read(a.root/'backup_receipt_complete01.json')['verified_files']]
    sources += [a.root/'backup_receipt_complete01.json',a.root/'external_controller.log',a.baseline,
        a.output/'independent_demo_audit.json',a.output/'LGVQ_Temporal_流水线demo核验说明.txt']
    manifest=dict(status='sealed',files=[dict(path=str(f.resolve()),bytes=f.stat().st_size,sha256=sha(f)) for f in sources])
    (a.output/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],groups=result['groups'],output=str(a.output),manifest_sha256=sha(a.output/'delivery_manifest.json')),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
