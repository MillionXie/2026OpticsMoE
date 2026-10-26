"""Synthetic raw/receipt tests only; not model or GPU measurement evidence."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import audit_lgvq_async_point_20261026 as auditor
from test_lgvq_async_raw_adapter_20261026 import fixture
from lgvq_async_raw_adapter_20261026 import adapt


def create_point(folder):
    folder.mkdir();reference=folder.parent/'synthetic_reference.npz'
    array=np.tile(np.arange(512,dtype=np.float32).reshape(32,16),(4,1))
    np.savez_compressed(reference,prediction=array);np.savez_compressed(folder/'outputs.npz',prediction=array)
    reference_sha=auditor.sha(reference);raw=fixture();raw['reference_sha256']=reference_sha
    raw['perf_monotonic_alignment_bracket']=[0.,0.]
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    samples=[Sample(1793000000.+float(t),float(t),'LG_actual_async_software_including_FFT',100.,50.,4096.)
             for t in np.arange(-.05,raw['host_end']+.1,.05)]
    raw['raw_board_energy']=interval_energy([(0.,raw['host_end'])],samples,host_w=0.)
    fields=tuple(Sample.__dataclass_fields__)
    with (folder/'telemetry.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(fields)
        writer.writerows([getattr(s,key) for key in fields] for s in samples)
    rows=[dict(raw_line='100,50,4096',valid_numeric_row=True,host_monotonic_s=s.host_monotonic_s) for s in samples]
    (folder/'telemetry_stdout.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write=lambda name,data:(folder/name).write_text(json.dumps(data))
    sources={auditor.OBSERVER_RELATIVE:'a'*64}
    write('identity.json',dict(GPU=auditor.GPU,cap=100,video_calls=2048,pid=777,
        published_commit='a'*40,sources=sources))
    write('async_identity.json',dict(gpu_uuid=auditor.GPU,actual_visible_multiprocessors=108,power_limit_w=250,
        checkpoint_sha256=auditor.CHECKPOINT_SHA256,reference_sha256=reference_sha,sources=sources))
    write('status.json',dict(status='complete'))
    write('process_audit.json',[dict(worker_pid=777,clients=[777],foreign_gpu_processes=[],foreign_mps_clients=[],
        human_authorized_nonexclusive=False,platform=dict(uuid=auditor.GPU,name='NVIDIA A100-PCIE-40GB',
                                                         power_limit_w=250,requested_cap=100))])
    write('external_preparation.json',dict(rows=[dict(group=g,CPU_preparation=[{}]*16) for g in range(32)],
        raw_inputs_GPU_resident=True,neural_features_cached=False,excluded_from_GPU_window=True))
    write('input_files.json',[dict(sample_id=str(i),video_path='/synthetic/'+str(i),bytes=1,sha256='a'*64) for i in range(512)])
    write('telemetry_observer_identity.json',dict(gpu_uuid=auditor.GPU,cuda_context_created=False,
        version='independent_process_memory_raw_v1',raw_file_IO_in_acquisition_reader=False,raw_file_written_after_stop=True,
        query=['nvidia-smi','--id='+auditor.GPU,'--loop-ms=50'],observer_source_sha256='a'*64))
    write('async_raw_timing.json',raw)
    write('async_event_records.json',dict(records=raw['records'],host_start=raw['host_start'],host_end=raw['host_end'],
        measured_software_wall_ms=raw['actual_async_software_wall_ms'],origin_host_clock_bracket=raw['origin_host_clock_bracket']))
    trace,report=adapt(raw);write('async_causal_trace.json',trace)
    report.update(raw_timing_sha256=auditor.sha(folder/'async_raw_timing.json'),
        event_records_sha256=auditor.sha(folder/'async_event_records.json'),
        outputs_sha256=auditor.sha(folder/'outputs.npz'),telemetry_sha256=auditor.sha(folder/'telemetry.csv'))
    write('cost_report.json',report)
    refresh_receipt(folder)
    return reference,reference_sha


def refresh_receipt(folder):
    files=[dict(path=p.name,bytes=p.stat().st_size,sha256=auditor.sha(p)) for p in folder.iterdir()
           if p.name!='backup_receipt_complete01.json']
    (folder/'backup_receipt_complete01.json').write_text(json.dumps(dict(source='/synthetic',verified_files=files)))


class AuditTests(unittest.TestCase):
    def test_whole_raw_replay_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'point';reference,digest=create_point(folder);output=folder/'local_audit.json'
            with patch.object(auditor,'REFERENCE_SHA256',digest):
                result=auditor.audit(folder,output,reference,verify_git=False)
                self.assertEqual(result['status'],'passed')
                self.assertTrue(result['all_2048_scores_bitwise_equal_sealed_reference'])
                self.assertFalse(result['physical_hardware_time_or_energy_measured'])
                with self.assertRaises(FileExistsError):auditor.audit(folder,output,reference,verify_git=False)
    def test_receipt_tamper_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'point';reference,digest=create_point(folder)
            (folder/'status.json').write_text('{"status":"complete","tampered":true}')
            with patch.object(auditor,'REFERENCE_SHA256',digest):
                with self.assertRaisesRegex(ValueError,'SHA/size'):auditor.audit(folder,folder/'audit.json',reference,verify_git=False)
    def test_raw_numeric_row_and_foreign_guard_cannot_be_ignored(self):
        for changed in ['power','foreign']:
            with tempfile.TemporaryDirectory() as directory:
                folder=Path(directory)/'point';reference,digest=create_point(folder)
                if changed=='power':
                    path=folder/'telemetry_stdout.jsonl';text=path.read_text().replace('100,50,4096','100,999,4096',1);path.write_text(text)
                else:
                    path=folder/'process_audit.json';rows=json.loads(path.read_text());rows[0]['foreign_gpu_processes']=[['1','gpu','99']];path.write_text(json.dumps(rows))
                refresh_receipt(folder)
                with patch.object(auditor,'REFERENCE_SHA256',digest):
                    with self.assertRaises(ValueError):auditor.audit(folder,folder/'audit.json',reference,verify_git=False)


if __name__=='__main__':unittest.main()
