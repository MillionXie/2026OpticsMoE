"""Five-model delivery with independently verified full-request timing rounds.

This presentation entry does not train, generate, select a checkpoint or contact
hardware. Ideal CCD fixed-wait timing remains an emulation with unknown device
transport cost, and a passed proxy mean never overrides a rejected image review.
"""
import argparse
import html
import json
import math
from pathlib import Path
import statistics
from .scene_compression_delivery import build as base_build,verified_models,request_signature
from .white_reference_scene_probe import digest,write


def checked_timing(spec,model):
    mode=spec['mode']
    if mode not in ('software','ideal_CCD_fixed_wait') or len(spec['rounds'])!=3:
        raise ValueError('Three independent complete-request rounds required')
    values=[];pids=[];signature=None;references=set();evidence=[]
    for folder in map(Path,spec['rounds']):
        report=json.loads((folder/'report.json').read_text())
        timing=json.loads((folder/'software_latency.json').read_text())
        execution=json.loads((folder/'execution.json').read_text())
        if (not report['complete'] or report.get('scope')!='DEV_software_latency'
            or report['checkpoint_sha256']!=model['report']['checkpoint_sha256']
            or report['sampling_steps']!=model['report']['sampling_steps']
            or report.get('instrumented_timing_not_formal_software_wall')
            or timing.get('contains_profiler_overhead') or timing.get('not_formal_software_wall')
            or timing['n']!=8 or len(timing['requests_ms'])!=8
            or timing['batch_size']!=1 or timing['extra_warmup'] is not False
            or not timing['model_load_excluded'] or not timing['task_LoRA_fused']
            or not timing['output_PNG_identity_vs_completed_quality_run']):
            raise ValueError('Timing must use the displayed checkpoint and uninstrumented B1 graph')
        if timing['first_request_ms']!=timing['requests_ms'][0]:
            raise ValueError('First request must be retained')
        if mode=='ideal_CCD_fixed_wait':
            proof=json.loads((folder/'CCD_replay_latency.json').read_text())
            if (timing.get('contains_GPU_FFT_optical_simulation')is not False
                or not timing.get('optical_wait_emulation') or timing.get('audit_run_not_formal_timing')
                or not proof['complete'] or proof['FFT_propagation_calls']!=0
                or proof['physical_CCD_or_SDK_used']is not False
                or not proof['electronic_residual_and_fusion_retained']
                or not proof['CPU_payload_conversion_and_device_copies_retained']
                or not math.isclose(proof['fixed_optical_ms'],5.1655361611877,rel_tol=0,abs_tol=1e-9)):
                raise ValueError('CCD replay is an audited fixed-wait emulation, not physical timing')
        rows=json.loads((folder/'images.json').read_text())
        current={r['sample_id']:request_signature(r)for r in rows}
        if len(rows)!=8 or len(current)!=8:raise ValueError('Eight unique timing requests required')
        if signature is None:signature=current
        elif current!=signature:raise ValueError('Timing rounds use different requests')
        for row in rows:
            expected=model['images'][row['sample_id']]
            if request_signature(row)!=request_signature(expected) or row['generated_sha256']!=expected['generated_sha256']:
                raise ValueError('Timed output differs from the displayed quality image')
            if digest(folder/row['generated_PNG'])!=row['generated_sha256']:
                raise ValueError('Timed PNG changed')
            references.add(row['reference_id'])
        numbers=[float(x)for x in timing['requests_ms']]
        if not all(math.isfinite(x)and x>0 for x in numbers):raise ValueError('Invalid request time')
        values.extend(numbers);pids.append(execution['pid'])
        evidence.append(dict(run=str(folder),report_sha256=digest(folder/'report.json'),timing_sha256=digest(folder/'software_latency.json')))
    if len(set(pids))!=3:raise ValueError('Independent process identities required')
    return dict(role=model['role'],label=model['label'],mode=mode,mean_ms=statistics.mean(values),requests_ms=values,
        n=24,independent_processes=3,unique_requests=8,reference_views=len(references),first_requests_retained=True,
        checkpoint_sha256=model['report']['checkpoint_sha256'],physical_measurement=False,
        unknown_extra_device_transport_not_assumed_zero=mode=='ideal_CCD_fixed_wait',evidence=evidence)


def timing_html(rows):
    baseline=next(r for r in rows if r['role']=='qwen'and r['mode']=='software')['mean_ms']
    output=['<h2>同边界完整请求计时</h2><table><tr><th>模型</th><th>计时方式</th><th>均值 ms</th><th>相对电子30步</th><th>四倍目标额外设备预算</th></tr>']
    for row in rows:
        ratio=baseline/row['mean_ms'];replay=row['mode']=='ideal_CCD_fixed_wait'
        row.update(ratio_vs_electronic30=ratio,extra_hardware_H_budget_for4x_ms=baseline/4-row['mean_ms']if replay else None)
        name='无FFT，理想CCD回放＋六固定周期'if replay else'完整软件请求（LG含FFT）'
        budget=f"{row['extra_hardware_H_budget_for4x_ms']:.2f} ms"if replay else'—'
        output.append('<tr><th>'+html.escape(row['label'])+'</th><td>'+name+'</td><td>'+f"{row['mean_ms']:.3f}"+'</td><td>'+f'{ratio:.4f}×'+'</td><td>'+budget+'</td></tr>')
    output.append('</table><p>每模型／方式3个独立进程，各8个B1请求，加载在窗外，保留首请求，无额外预热或最快轮筛选。计时从RGB与文字就绪至CPU RGB；固定8请求的参考视图数分别记录在计时JSON，不是24独立商品。</p><p>理想CCD回放列重新执行真实RGB文字Qwen和所有必要电子计算，保留场数据复制及像素转换；六周期合计5.165536ms。它不是实际CCD画质或实验台整机计时。实际SDK、SLM链路及额外DMA仍为未知H；只有H不超过表中预算时，这一时序模型才仍达到四倍。</p>')
    return ''.join(output)


def build(manifest,output):
    models=verified_models(manifest);indexed={m['role']:m for m in models}
    times=[checked_timing(spec,indexed[spec['role']])for spec in manifest['timing_rounds']]
    if len({(r['role'],r['mode'])for r in times})!=len(times):raise ValueError('Duplicate timing role/mode')
    if not any(r['role']=='qwen'and r['mode']=='software'for r in times):raise ValueError('Electronic reference timing required')
    review_path=Path(manifest['manual_review']);review=json.loads(review_path.read_text())
    light=indexed['light_lg']
    if (not review['complete'] or not review['all128_contexts_reviewed'] or not review['all16_boards_reviewed']
        or review['checkpoint_sha256']!=light['report']['checkpoint_sha256']):
        raise ValueError('Complete manual diagnostic must bind the displayed light model')
    table=timing_html(times)
    status='原图审查通过；仍按下列计时及独立评估范围解释。'if review['approved_as_final_lightweight_model']else'尚未批准为最终轻量版：代理均值不能覆盖原图审查发现的失败。'
    base_build(manifest,output)
    page=output/'index.html';content=page.read_text(encoding='utf8')
    content=content.replace('<h2>整体及分类均值</h2>','<div class="notice"><strong>'+status+'</strong> 本页显示全部128组开发结果，不是新权重独立TEST500或硬件验收。</div><h2>整体及分类均值</h2>',1)
    content=content.replace('<h2>耗时口径</h2>',table+'<h2>耗时口径</h2>',1)
    page.write_text(content,encoding='utf8')
    write(output/'timing.json',dict(complete=True,rows=times,physical_measurement=False))
    write(output/'manual_review.json',review)
    record=json.loads((output/'delivery_manifest.json').read_text());record.update(acceleration_builder_sha256=digest(Path(__file__)),
        manual_review_source_sha256=digest(review_path),approved_as_final_lightweight_model=review['approved_as_final_lightweight_model'],
        timing_rounds_independently_verified=True,new_weights_independent_TEST_performed=False)
    write(output/'delivery_manifest.json',record)
    write(output/'file_manifest.json',{p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))for p in sorted(output.rglob('*'))if p.is_file()and p.name!='file_manifest.json'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(json.loads(a.manifest.read_text()),a.output)


if __name__=='__main__':main()
