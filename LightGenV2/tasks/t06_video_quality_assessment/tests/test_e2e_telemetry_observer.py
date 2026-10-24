"""Synthetic CPU contracts: slow disk writes never occur in the acquisition reader."""
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import MemoryRawSampler,observer_command


def test_all_raw_and_numeric_rows_retained_without_disk_IO_in_reader(tmp_path):
    path=tmp_path/'raw.jsonl';sampler=MemoryRawSampler('synthetic',path)
    sampler.process=SimpleNamespace(stdout=StringIO('41.5, 26, 4735\n92.5, 30, 4735\n'))
    sampler.stopping=True;sampler._read()
    assert not path.exists() and sampler.raw_stream is None
    assert len(sampler.samples)==len(sampler.raw_records)==2
    assert [v.watts for v in sampler.samples]==[41.5,92.5]
    assert all(v['valid_numeric_row'] for v in sampler.raw_records)


def test_invalid_NA_row_is_retained_and_raises_not_skipped(tmp_path):
    sampler=MemoryRawSampler('synthetic',tmp_path/'raw.jsonl')
    sampler.process=SimpleNamespace(stdout=StringIO('N/A, 26, 4735\n'))
    sampler.stopping=True;sampler._read()
    assert len(sampler.raw_records)==1 and not sampler.raw_records[0]['valid_numeric_row']
    assert not sampler.samples
    with pytest.raises(RuntimeError):sampler.check()


def test_stop_joins_reader_before_writing_raw_and_never_overwrites(tmp_path,monkeypatch):
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import TelemetrySampler
    path=tmp_path/'raw.jsonl';sampler=MemoryRawSampler('synthetic',path)
    sampler.raw_records=[dict(raw_line='40, 10, 1',host_monotonic_s=1,valid_numeric_row=True)]
    events=[]
    def stop(self,raise_on_error):
        assert not path.exists();events.append('stopped_and_joined');return []
    monkeypatch.setattr(TelemetrySampler,'stop',stop)
    sampler.stop()
    assert events==['stopped_and_joined']
    assert json.loads(path.read_text())['raw_line']=='40, 10, 1'
    with pytest.raises(FileExistsError):MemoryRawSampler('synthetic',path).start()


def test_new_command_does_not_modify_historical_observer():
    command=observer_command('GPU-expected',Path('raw.jsonl'))
    assert command[3].endswith('.e2e_telemetry_observer')
    assert 'GPU-expected' in command
