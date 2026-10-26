"""Data-only final-table inputs. No network, model, Torch, or CUDA imports.

Existing point audits are read, never regenerated. Partial exports remain
explicitly partial; final exports require 24 lab points and qualified async LG.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

LAB_GPU = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
RENTAL_GPU = 'GPU-ecac2c49-306a-9ec5-1572-8fcd2d232165'
CHARGED = ('CPU_tokenizer_retained_ms', 'entry_dispatch_retained_ms',
           'full_GPU_model_to_score_host_ms', 'intermediate_CPU_output_retained_ms',
           'intervideo_dispatch_retained_ms')
EXPECTED = {(m, c, r) for m in ('qwen', 'glm', 'clip', 'yolo')
            for c in (100, 50) for r in (1, 2, 3)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite numeric value required')
    return value


def close(a, b):
    if not math.isclose(number(a), number(b), rel_tol=1e-9, abs_tol=1e-5):
        raise ValueError('Audit/report/denominator disagreement')


def sealed_files(folder):
    folder = Path(folder).resolve()
    receipt_path = folder / 'backup_receipt_complete01.json'
    if not receipt_path.exists():
        candidates = [folder.parent / name for name in
                      ('backup_receipt_stopped01.json', 'backup_receipt_complete01.json')]
        receipt_path = next((p for p in candidates if p.exists()), None)
        if receipt_path is None:
            raise ValueError('Completed leaf or closed-parent receipt required: ' + str(folder))
    receipt = read(receipt_path)
    receipt_root = receipt_path.parent
    rows = []; seen = set(); receipt_seen = set()
    for row in receipt['verified_files']:
        member = Path(row['path'])
        path = receipt_root / member
        if (member.is_absolute() or member.drive or '..' in member.parts
                or row['path'] in receipt_seen or path.is_symlink()
                or not path.resolve().is_relative_to(receipt_root)):
            raise ValueError('Unsafe or duplicate receipt member')
        receipt_seen.add(row['path'])
        if not path.resolve().is_relative_to(folder):
            continue
        seen.add(path.relative_to(folder).as_posix())
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Original backup SHA/size differs: ' + str(path))
        rows.append(dict(local_path=str(path), remote_path=receipt['source'].rstrip('/') + '/' + member.as_posix(),
                         bytes=row['bytes'], sha256=row['sha256'], role='original_remote_data'))
    for required in ('status.json', 'identity.json', 'cost_report.json'):
        if required not in seen:
            raise ValueError('Incomplete original receipt: ' + required)
    rows.append(dict(local_path=str(receipt_path), remote_path=None, bytes=receipt_path.stat().st_size,
                     sha256=sha(receipt_path), role='local_backup_receipt'))
    return rows


def point(folder, kind='baseline'):
    folder = Path(folder).resolve()
    files = sealed_files(folder)
    cost = read(folder / 'cost_report.json'); identity = read(folder / 'identity.json')
    if read(folder / 'status.json')['status'] != 'complete' or cost['status'] != 'complete':
        raise ValueError('Failed/pending points cannot enter a result table')
    audit_path = folder / ('local_independent_energy_audit01.json' if kind == 'baseline'
                           else 'local_independent_async_audit01.json')
    audit = read(audit_path)
    if audit['status'] != 'passed' or audit['video_calls'] != 2048:
        raise ValueError('Existing independent point audit required')
    required_raw = ({'timing.json', 'telemetry.csv', 'outputs.npz', 'cost_report.json'} if kind == 'baseline'
                    else {'async_raw_timing.json', 'async_event_records.json', 'async_causal_trace.json',
                          'outputs.npz', 'telemetry.csv', 'cost_report.json'})
    sealed_names = {Path(r['local_path']).relative_to(folder).as_posix() for r in files
                    if r['role'] == 'original_remote_data'}
    if not required_raw <= set(audit['raw_sha256']) or not set(audit['raw_sha256']) <= sealed_names:
        raise ValueError('Independent audit/receipt raw closure incomplete')
    for name, digest in audit['raw_sha256'].items():
        if sha(folder / name) != digest:
            raise ValueError('Previously audited raw evidence changed')
    files.append(dict(local_path=str(audit_path), remote_path=None, bytes=audit_path.stat().st_size,
                      sha256=sha(audit_path), role='local_independent_audit'))
    if kind != 'baseline':
        if (identity['GPU'] != LAB_GPU or cost['qualified_for_final_hybrid_table'] is not True
                or not audit['all_2048_scores_bitwise_equal_sealed_reference']
                or not audit['every_original_DAG_node_start_reproduced']
                or audit['physical_hardware_time_or_energy_measured'] is not False
                or cost['physical_hardware_time_measured'] is not False):
            raise ValueError('Qualified measured-software / projected-optics LG required')
        close(cost['time16_ms'], audit['projected_time16_ms'])
        close(cost['energy16_j'], audit['projected_energy16_j'])
        close(cost['common_tokens_per_s'], audit['projected_common_tokens_per_s'])
        close(cost['native_tokens_per_s'], audit['projected_native_tokens_per_s'])
        if not 0 <= number(audit['max_power_sample_age_s']) <= .25:
            raise ValueError('LG original software power-age gate differs')
        raw = read(folder / 'async_raw_timing.json')
        board = raw['raw_board_energy']
        close(board['elapsed_s'] * 1000, cost['actual_async_software_wall_ms'])
        close(board['energy_j'], audit['actual_board_FFT_diagnostic_j'])
        close(board['host_proxy_w'], 0.)
        close(cost['common_effective_tokens'], 913408)
        return dict(path=str(folder), kind='LG_async', platform='A10040GB_PCIE_250W',
                    hardware_power_limit_w=250, diagnostic_host_proxy_w=0.,
                    measured_software_matched_board_average_w=board['energy_j'] / board['elapsed_s'],
                    measured_software_common_tokens_per_s=cost['common_effective_tokens'] / board['elapsed_s'],
                    measured_software_native_tokens_per_s=number(cost['native_effective_tokens']) / board['elapsed_s'],
                    measured_software_time16_ms=cost['actual_async_software_wall_ms'] / 128,
                    measured_software_board_FFT_energy16_j=audit['actual_board_FFT_diagnostic_j'] / 128,
                    fixed_optics_counterfactual_time16_ms=cost['time16_ms'],
                    fixed_optics_component_proxy_energy16_j=cost['energy16_j'],
                    projected_common_tokens_per_s=cost['common_tokens_per_s'],
                    projected_native_tokens_per_s=cost['native_tokens_per_s'],
                    native_token_scope=cost['native_token_scope'],
                    actual_stream_span_overlap_ms=cost['measured_next_frontend_prior_body_overlap_ms'],
                    GPU_resource_span_totals_NOT_additive=cost['actual_GPU_resource_span_totals_ms'],
                    component_energy_proxy=cost['phase_energy_proxy'],
                    physical_hardware_time_or_energy_measured=False,
                    kernel_occupancy_concurrency_proven=False, files=files)
    model = cost['model']; cap = cost['cap']; repeat = cost.get('repeat', cost.get('round'))
    if 'repeat' in cost and 'round' in cost and cost['repeat'] != cost['round']:
        raise ValueError('Conflicting original round identities')
    if identity['gpu_uuid'] == LAB_GPU and model in ('qwen', 'glm', 'clip', 'yolo'):
        platform = 'A10040GB_PCIE_250W'; limit = 250
    elif identity['gpu_uuid'] == RENTAL_GPU and model in ('kimi', 'deepseek'):
        platform = 'A10080GB_SXM4_400W'; limit = 400
    else:
        raise ValueError('Unknown or mixed platform/model identity')
    if (cap not in (50, 100) or repeat not in (1, 2, 3) or identity['power_limit_w'] != limit
            or identity['visible_multiprocessors'] != (54 if cap == 50 else 108)
            or cost['video_calls'] != 2048 or cost['unique_videos'] != 512
            or cost['common_Qwen_effective_tokens'] != 913408 or cost['host_proxy_w'] != 338.2
            or not audit['all_2048_outputs_equal_records'] or not audit['four_cycles_bitwise_equal']
            or not audit['energy_is_same_time_intervals']):
        raise ValueError('Original baseline scope/identity/audit differs')
    elapsed = number(cost['elapsed_s']); energy = number(cost['energy_j'])
    board = number(cost['GPU_energy_j'])
    if board <= 0:raise ValueError('Positive matched-interval board energy required')
    close(energy, board + cost['host_proxy_w'] * elapsed)
    close(cost['observed_timeweighted_GPU_w'], board / elapsed)
    if elapsed <= 0 or energy <= 0 or not 0 <= number(cost['max_power_sample_age_s']) <= .25:
        raise ValueError('Invalid time, energy, or original power-age gate')
    close(cost['time16_ms'], elapsed * 1000 / 128)
    close(cost['energy16_j'], energy / 128)
    close(cost['common_tokens_per_s'], 913408 / elapsed)
    close(audit['time16_ms'], cost['time16_ms']); close(audit['energy16_j'], cost['energy16_j'])
    stages = audit['stage_totals_ms']
    close(sum(number(stages[k]) for k in CHARGED), elapsed * 1000)
    native = cost['native_effective_tokens']
    if model in ('clip', 'yolo'):
        if native is not None or cost['native_tokens_per_s'] is not None:
            raise ValueError('Visual-only native word tokens must be not applicable, never zero')
    else:
        if number(native) <= 0: raise ValueError('Native effective token count missing')
        close(cost['native_tokens_per_s'], native / elapsed)
    return dict(path=str(folder), kind='baseline', platform=platform, model=model, cap=cap, repeat=repeat,
                calls=2048, elapsed_s=elapsed, energy_j=energy, common_effective_tokens=913408,
                hardware_power_limit_w=limit, measured_board_energy_j=board,
                measured_matched_board_average_w=board / elapsed, host_proxy_w=cost['host_proxy_w'],
                native_effective_tokens=native, time16_ms=cost['time16_ms'], energy16_j=cost['energy16_j'],
                stage_totals_ms=stages, first_TF_node_recorded=audit['first_TF_node_recorded'], files=files)


def aggregate(points):
    groups = {}; seen = set()
    for p in points:
        key = (p['platform'], p['model'], p['cap']); identity = key + (p['repeat'],)
        if identity in seen: raise ValueError('Duplicate point identity, including across suite roots')
        seen.add(identity); groups.setdefault(key, []).append(p)
    result = []
    for (platform, model, cap), rows in sorted(groups.items()):
        calls = sum(r['calls'] for r in rows); elapsed = sum(r['elapsed_s'] for r in rows)
        energy = sum(r['energy_j'] for r in rows); common = sum(r['common_effective_tokens'] for r in rows)
        board = sum(r['measured_board_energy_j'] for r in rows)
        if len({(r['hardware_power_limit_w'], r['host_proxy_w']) for r in rows}) != 1:
            raise ValueError('Mixed power cap or host proxy within one configuration')
        natives = [r['native_effective_tokens'] for r in rows]
        if any(x is None for x in natives) and not all(x is None for x in natives):
            raise ValueError('Mixed native token applicability')
        stage_keys = set(rows[0]['stage_totals_ms'])
        if any(set(r['stage_totals_ms']) != stage_keys for r in rows):
            raise ValueError('Mixed phase measurement availability; do not fabricate missing first-TF nodes')
        result.append(dict(platform=platform, model=model, cap=cap, rounds=len(rows), video_calls=calls,
            repeats=sorted(r['repeat'] for r in rows), time16_ms=elapsed * 1000 * 16 / calls,
            equivalent_frame_ms=elapsed * 1000 / (calls * 4), energy16_j=energy * 16 / calls,
            energy_per_frame_j=energy / (calls * 4), pooled_common_tokens_per_s=common / elapsed,
            pooled_native_tokens_per_s=None if natives[0] is None else sum(natives) / elapsed,
            common_effective_tokens=common, native_effective_tokens=None if natives[0] is None else sum(natives),
            elapsed_s=elapsed, energy_j=energy, all_available_rounds_used=True,
            hardware_power_limit_w=rows[0]['hardware_power_limit_w'], host_proxy_w=rows[0]['host_proxy_w'],
            measured_board_energy_j=board, measured_matched_board_average_w=board / elapsed,
            stage_equivalent16_ms={k:sum(r['stage_totals_ms'][k] for r in rows) * 16 / calls for k in sorted(stage_keys)},
            first_TF_node_recorded=all(r['first_TF_node_recorded'] for r in rows)))
    return result


def collect(folders, lg_folder=None, allow_partial=False, physical_scope_ledger=None):
    points = [point(p) for p in folders]
    groups = aggregate(points)
    done = {(p['model'], p['cap'], p['repeat']) for p in points if p['platform']=='A10040GB_PCIE_250W'}
    missing = sorted(EXPECTED - done)
    lg = point(lg_folder, 'LG_async') if lg_folder else None
    if physical_scope_ledger is not None:
        if lg is None:raise ValueError('Measured accepted LG required before physical scope replay')
        from lgvq_phase_ledger_20261026 import build as phase_build
        ledger_path=Path(physical_scope_ledger).resolve();ledger=read(ledger_path)
        if ledger.get('scope_version')!='exclude_software_CPU_submit_v2' or ledger!=phase_build(lg_folder,True):
            raise ValueError('New physical scope must replay from every original measured node')
        keys=('fixed_optics_counterfactual_time16_ms','fixed_optics_component_proxy_energy16_j',
              'projected_common_tokens_per_s','projected_native_tokens_per_s','component_energy_proxy')
        lg['legacy_fixed_optics_v1']={k:lg[k] for k in keys}
        lg.update(fixed_optics_counterfactual_time16_ms=ledger['fixed_optics']['wall_ms16'],
            fixed_optics_component_proxy_energy16_j=ledger['fixed_optics_component_energy']['energy16_j'],
            projected_common_tokens_per_s=ledger['projected_common_tokens_per_s'],
            projected_native_tokens_per_s=ledger['projected_native_tokens_per_s'],
            component_energy_proxy=ledger['fixed_optics_component_energy'],
            physical_scope_version=ledger['scope_version'],physical_scope_ledger_path=str(ledger_path),
            physical_scope_ledger_sha256=sha(ledger_path))
        lg['files'].append(dict(local_path=str(ledger_path),remote_path=None,bytes=ledger_path.stat().st_size,
            sha256=sha(ledger_path),role='local_derived_physical_scope'))
        if ledger.get('reference_round'):
            lg['files'].extend(point(Path(ledger['reference_round']['source_leaf']),'LG_async')['files'])
    if not allow_partial and (missing or lg is None):
        raise ValueError('Final summary requires all 24 lab points and a qualified async LG point')
    files = {}
    for p in points + ([lg] if lg else []):
        for row in p['files']:
            old = files.setdefault(row['local_path'], row)
            if old != row: raise ValueError('Conflicting file identities')
    comparisons = None if lg is None else [dict(platform=g['platform'], model=g['model'], cap=g['cap'],
        platform_relation='same_platform' if g['platform']==lg['platform'] else 'cross_configuration_not_controlled',
        baseline_over_actual_LG_software_time=g['time16_ms']/lg['measured_software_time16_ms'],
        baseline_over_projected_LG_time=g['time16_ms']/lg['fixed_optics_counterfactual_time16_ms'],
        baseline_proxy_over_LG_component_proxy_energy=g['energy16_j']/lg['fixed_optics_component_proxy_energy16_j'],
        physical_optical_speedup_measured=False,energy_proxy_models_differ=True) for g in groups]
    return dict(status='partial_not_final' if missing or lg is None else 'final_qualified_inputs',
        lab_complete_points=len(done), lab_required_points=24, missing_lab_points=missing,
        LG_async_measured=lg is not None, groups=groups, LG=lg, comparisons=comparisons,
        rounds=[{k:v for k,v in p.items() if k!='files'} for p in points],
        raw_file_index=sorted(files.values(), key=lambda x:x['local_path']),
        scope='2048 calls per round = original first 512 TEST videos repeated four times',
        baseline_energy='Same retained intervals: original NVML ZOH + 338.2W host proxy; not wall meter',
        LG_projection_is_not_physical_hardware_measurement=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point', action='append', type=Path, required=True)
    parser.add_argument('--lg-point', type=Path)
    parser.add_argument('--lg-physical-scope',type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Preserve immutable summary; choose a new output path')
    result = collect(args.point, args.lg_point, args.allow_partial,args.lg_physical_scope)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], lab_complete_points=result['lab_complete_points'],
                          groups=len(result['groups']), output=str(args.out.resolve()))))
