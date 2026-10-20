import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import strict_tf_window_20261020 as m


def samples():
    return [SimpleNamespace(host_monotonic_s=i/10,watts=100 if i<5 else 200) for i in range(11)]


def test_cost_excludes_all_preblock_intervals_but_charges_actual_wait():
    # Forward1 [.2,.3], internal WAIT [.3,.5], forward2 [.7,.9].
    # [.0,.2] and [.5,.7] represent excluded CPU/GPU frontend work.
    result=m.interval_energy([(.2,.3),(.3,.5),(.7,.9)],samples())
    assert result['elapsed_s']==pytest.approx(.5)
    assert result['energy_j']==pytest.approx((338.2+100)*.3+(338.2+200)*.2)
    assert result['power_samples_used']>=5


def test_unequal_intervals_use_duration_weighted_power_not_sample_mean():
    result=m.interval_energy([(.09,.11)],samples())
    assert result['energy_j']==pytest.approx((338.2+100)*.02)
    assert result['elapsed_s']==pytest.approx(.02)


@pytest.mark.parametrize('intervals',[[(-.1,.1)],[(.2,.1)],[(.2,.4),(.3,.5)],[(1.,1.3)]])
def test_missing_stale_or_overlapping_power_support_is_rejected(intervals):
    with pytest.raises(ValueError):m.interval_energy(intervals,samples())


def test_each_forward_has_its_own_synchronized_boundary_and_raw_record():
    source=Path(m.__file__).read_text();ast.parse(source)
    assert "armed.update(enabled=True,start=None)" in source
    assert "torch.cuda.synchronize();armed['start']=time.monotonic()" in source
    assert 'out,_=compute(inputs);torch.cuda.synchronize()' in source
    assert "if i+1<count and duty<1:" in source
    assert "every_preblock_interval_excluded=True" in source
    assert "cost_intervals_are_not_contiguous_wall_window=True" in source
    assert "rows=[s for s in sampler.samples if wall_start<=s.host_monotonic_s<wall_end]" in source
    assert "('calls.csv',calls),('waits.csv',waits)" in source
    assert 'no_window_selection=True' in source


@pytest.mark.parametrize('duty',[1,.5])
def test_complete_fake_execution_preserves_each_boundary_and_all_three_trials(monkeypatch,tmp_path,duty):
    """CPU-only instrumentation test; these synthetic values are NOT run data."""
    import sys
    import json
    class Clock:
        now=0.
        def monotonic(self):self.now+=.001;return self.now
        def sleep(self,seconds):self.now+=seconds
    clock=Clock();monkeypatch.setattr(m,'time',clock)
    class First:
        hook=None
        def register_forward_pre_hook(self,hook):
            self.hook=hook
            return SimpleNamespace(remove=lambda:setattr(self,'hook',None))
    first=First()
    def sync():clock.now+=.001
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(synchronize=sync)))
    class Sampler:
        def __init__(self,*a,**k):pass
        def start(self):pass
        def set_phase(self,*a):pass
        def check(self):pass
        @property
        def samples(self):
            return [SimpleNamespace(host_monotonic_s=i*.05,host_time_unix=i*.05,phase='',
                watts=100.,utilization_percent=50.,memory_used_mib=1.) for i in range(int(clock.now/.05)+1)]
        def stop(self,**k):return self.samples
    monkeypatch.setattr(m.base.telemetry,'TelemetrySampler',Sampler)
    monkeypatch.setattr(m.base.telemetry,'save_samples',lambda *a:None)
    monkeypatch.setattr(m.base.prior,'ClockSampler',lambda *a:SimpleNamespace(start=lambda:None,stop=lambda *a:None))
    out=SimpleNamespace(detach=lambda:SimpleNamespace(cpu=lambda:SimpleNamespace(tolist=lambda:[1.])))
    def compute(_):
        clock.now+=.004;first.hook();clock.now+=.020
        return out,None
    args=SimpleNamespace(gpu_uuid='synthetic',model='synthetic',precision='native',method='temporal',batch=2,cap=100)
    result=m.measure(args,compute,None,first,tmp_path/'synthetic_cpu_fixture',SimpleNamespace(fast=lambda:None),duty)
    assert len(result['repeats'])==3
    for rep in result['repeats']:
        assert rep['elapsed_s']==pytest.approx(rep['compute_s']+rep['internal_wait_s'])
        assert rep['energy_j']==pytest.approx((338.2+100)*rep['elapsed_s'])
        assert rep['excluded_preblock_s']>0
        assert rep['wall_elapsed_s']>rep['elapsed_s']
        assert rep['video_evaluations']==rep['forward_calls']*2
        assert rep['internal_wait_s']>0 if duty<1 else rep['internal_wait_s']==0
    assert first.hook is None
    assert json.loads((tmp_path/'synthetic_cpu_fixture/report.json').read_text())['timing_protocol']==m.PROTOCOL
