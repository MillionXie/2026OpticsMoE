"""Synthetic bookkeeping tests; no model/GPU execution or new qualification."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import collect_energy_summary_20261026 as collector


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def reseal(folder):
    paths = [p for p in folder.iterdir() if p.name not in
             ('backup_receipt_complete01.json', 'local_independent_energy_audit01.json',
              'local_independent_async_audit01.json')]
    write(folder/'backup_receipt_complete01.json', dict(source='/synthetic/'+folder.name,
        verified_files=[dict(path=p.name, bytes=p.stat().st_size, sha256=collector.sha(p)) for p in paths]))


def fixture(folder, model='qwen', repeat=1, elapsed=256.):
    folder.mkdir()
    write(folder/'identity.json', dict(gpu_uuid=collector.LAB_GPU, power_limit_w=250, visible_multiprocessors=54))
    write(folder/'status.json', dict(status='complete'))
    visual = model in ('clip', 'yolo'); energy = elapsed*400
    cost = dict(status='complete',model=model,cap=50,repeat=repeat,video_calls=2048,unique_videos=512,
        common_Qwen_effective_tokens=913408,host_proxy_w=338.2,elapsed_s=elapsed,energy_j=energy,
        max_power_sample_age_s=.1,time16_ms=elapsed*1000/128,energy16_j=energy/128,
        common_tokens_per_s=913408/elapsed,native_effective_tokens=None if visual else 913408,
        native_tokens_per_s=None if visual else 913408/elapsed)
    write(folder/'cost_report.json',cost)
    for name in ('timing.json','telemetry.csv','outputs.npz'): (folder/name).write_text('synthetic')
    write(folder/'local_independent_energy_audit01.json',dict(status='passed',video_calls=2048,
        all_2048_outputs_equal_records=True,four_cycles_bitwise_equal=True,energy_is_same_time_intervals=True,
        time16_ms=cost['time16_ms'],energy16_j=cost['energy16_j'],first_TF_node_recorded=True,
        stage_totals_ms={key:elapsed*1000/len(collector.CHARGED) for key in collector.CHARGED},
        raw_sha256={name:collector.sha(folder/name) for name in
                    ('timing.json','telemetry.csv','outputs.npz','cost_report.json')}))
    reseal(folder)
    return folder


class SummaryTests(unittest.TestCase):
    def test_closed_parent_receipt_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'p');receipt=collector.read(p/'backup_receipt_complete01.json')
            for row in receipt['verified_files']:row['path']='p/'+row['path']
            receipt['source']='/synthetic'
            write(p.parent/'backup_receipt_stopped01.json',receipt)
            (p/'backup_receipt_complete01.json').unlink()
            result=collector.point(p)
            self.assertEqual(result['model'],'qwen')
            self.assertTrue(any(r['remote_path']=='/synthetic/p/timing.json' for r in result['files']))

    def test_LG_software_and_projection_never_conflated(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'p')
            write(p/'identity.json',dict(GPU=collector.LAB_GPU))
            c=collector.read(p/'cost_report.json')
            c.update(qualified_for_final_hybrid_table=True,physical_hardware_time_measured=False,
                actual_async_software_wall_ms=128000.,native_token_scope='one prompt per field',
                measured_next_frontend_prior_body_overlap_ms=10.,actual_GPU_resource_span_totals_ms={'electronics':200.},
                phase_energy_proxy={'controller_union_ms':200.})
            write(p/'cost_report.json',c)
            raw=('async_raw_timing.json','async_event_records.json','async_causal_trace.json',
                 'outputs.npz','telemetry.csv','cost_report.json')
            for name in raw[:3]:write(p/name,dict(synthetic=True))
            write(p/'local_independent_async_audit01.json',dict(status='passed',video_calls=2048,
                all_2048_scores_bitwise_equal_sealed_reference=True,every_original_DAG_node_start_reproduced=True,
                physical_hardware_time_or_energy_measured=False,projected_time16_ms=c['time16_ms'],
                projected_energy16_j=c['energy16_j'],actual_board_FFT_diagnostic_j=128.,max_power_sample_age_s=.05,
                projected_common_tokens_per_s=c['common_tokens_per_s'],projected_native_tokens_per_s=c['native_tokens_per_s'],
                raw_sha256={name:collector.sha(p/name) for name in raw}))
            reseal(p);lg=collector.point(p,'LG_async')
            self.assertEqual(lg['measured_software_time16_ms'],1000.)
            self.assertEqual(lg['fixed_optics_counterfactual_time16_ms'],2000.)
            self.assertEqual(lg['measured_software_board_FFT_energy16_j'],1.)
            self.assertEqual(lg['fixed_optics_component_proxy_energy16_j'],800.)
            self.assertFalse(lg['physical_hardware_time_or_energy_measured'])

    def test_qualified_point_and_explicit_partial(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'p');r=collector.collect([p],allow_partial=True)
            self.assertEqual(r['status'],'partial_not_final');self.assertEqual(r['lab_complete_points'],1)
            self.assertEqual(len(r['missing_lab_points']),23)
            self.assertFalse(r['LG_async_measured']);self.assertIsNone(r['LG'])
            self.assertEqual(len(r['raw_file_index']),8)
            with self.assertRaises(ValueError):collector.collect([p])

    def test_pool_ratio_is_not_average_rate(self):
        with tempfile.TemporaryDirectory() as d:
            a=fixture(Path(d)/'a',repeat=1,elapsed=128.)
            b=fixture(Path(d)/'b',repeat=2,elapsed=512.)
            group=collector.collect([a,b],allow_partial=True)['groups'][0]
            self.assertAlmostEqual(group['pooled_common_tokens_per_s'],2*913408/640)
            self.assertEqual(group['rounds'],2)
            self.assertAlmostEqual(group['time16_ms'],2500.)
            self.assertAlmostEqual(sum(group['stage_equivalent16_ms'].values()),group['time16_ms'])

    def test_duplicate_across_roots_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a=fixture(Path(d)/'a');b=fixture(Path(d)/'b')
            with self.assertRaises(ValueError):collector.collect([a,b],allow_partial=True)

    def test_tamper_and_failed_point_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'p');(p/'telemetry.csv').write_text('changed')
            with self.assertRaises(ValueError):collector.point(p)
            reseal(p)
            with self.assertRaises(ValueError):collector.point(p)  # receipt cannot replace old audit
            write(p/'status.json',dict(status='failed'));reseal(p)
            with self.assertRaises(ValueError):collector.point(p)

    def test_visual_native_token_not_applicable(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'p',model='clip')
            r=collector.collect([p],allow_partial=True)['groups'][0]
            self.assertIsNone(r['pooled_native_tokens_per_s'])
            self.assertIsNone(r['native_effective_tokens'])

    def test_platform_and_power_age_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'p')
            x=collector.read(p/'identity.json');x['power_limit_w']=400;write(p/'identity.json',x);reseal(p)
            with self.assertRaises(ValueError):collector.point(p)
            x['power_limit_w']=250;write(p/'identity.json',x)
            c=collector.read(p/'cost_report.json');c['max_power_sample_age_s']=.251;write(p/'cost_report.json',c)
            a=collector.read(p/'local_independent_energy_audit01.json')
            a['raw_sha256']['cost_report.json']=collector.sha(p/'cost_report.json')
            write(p/'local_independent_energy_audit01.json',a);reseal(p)
            with self.assertRaises(ValueError):collector.point(p)

    def test_missing_phase_or_raw_closure_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'p');a=collector.read(p/'local_independent_energy_audit01.json')
            a['raw_sha256'].pop('timing.json');write(p/'local_independent_energy_audit01.json',a)
            with self.assertRaises(ValueError):collector.point(p)

    def test_final_requires_LG_even_when_all_lab_done(self):
        rows=[]
        for model,cap,repeat in collector.EXPECTED:
            rows.append(dict(platform='A10040GB_PCIE_250W',model=model,cap=cap,repeat=repeat,
                calls=2048,elapsed_s=1.,energy_j=400.,common_effective_tokens=913408,
                native_effective_tokens=None if model in ('clip','yolo') else 913408,
                stage_totals_ms={'stage':1000.},first_TF_node_recorded=True,files=[]))
        with patch.object(collector,'point',side_effect=rows):
            with self.assertRaises(ValueError):collector.collect(list(range(24)))

    def test_nonfinite_and_boolean_numeric_rejected(self):
        for value in (float('nan'),float('inf'),True,None):
            with self.assertRaises(ValueError):collector.number(value)


if __name__ == '__main__':unittest.main()
