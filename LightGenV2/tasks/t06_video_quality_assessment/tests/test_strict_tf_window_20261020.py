import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from LightGenV2.tasks.t06_video_quality_assessment import strict_tf_window_20261020 as m


def samples():
    return [SimpleNamespace(host_monotonic_s=i/10,watts=100 if i<5 else 200) for i in range(11)]


def test_validation_is_bounded_and_keeps_all_calibration_windows():
    rates=iter([44.,49.5]);history=[];duties=[]
    def observe(duty,attempt):duties.append(duty);return next(rates)
    result=m.validated_duty(.6,observe,history.append)
    assert result==pytest.approx(.6*50/44)
    assert len(history)==2 and history[-1]['passed']
    assert duties[1]>duties[0]
    failed=[]
    with pytest.raises(RuntimeError,match='no formal paced trials'):
        m.validated_duty(.5,lambda *_:40.,failed.append)
    assert len(failed)==3 and not any(r['passed'] for r in failed)


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
    assert "synchronize();armed['start']=time.monotonic()" in source
    assert 'out,_=compute(inputs);synchronize()' in source
    assert 'for device in range(len(gpus)):torch.cuda.synchronize(device)' in source
    assert "if i+1<count and duty<1:" in source
    assert "every_preblock_interval_excluded=True" in source
    assert "cost_intervals_are_not_contiguous_wall_window=True" in source
    assert "if wall_start<=s.host_monotonic_s<wall_end]" in source
    assert "('calls.csv',calls),('waits.csv',waits)" in source
    assert 'no_window_selection=True' in source


def test_dual_gpu_energy_counts_both_boards_and_host_once():
    streams=[samples(),[SimpleNamespace(host_monotonic_s=r.host_monotonic_s,watts=2*r.watts) for r in samples()]]
    result=m.multi_gpu_energy([(.2,.3),(.3,.5),(.7,.9)],streams)
    assert result['elapsed_s']==pytest.approx(.5)
    assert result['energy_j']==pytest.approx(338.2*.5+300*.3+600*.2)


@pytest.mark.parametrize('duty',[1,.5])
@pytest.mark.parametrize('devices',[1,2])
@pytest.mark.parametrize('validation',[False,True])
def test_complete_fake_execution_preserves_each_boundary_and_all_three_trials(monkeypatch,tmp_path,duty,devices,validation):
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
    synchronized=[]
    def sync(*device):clock.now+=.001;synchronized.append(device)
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
    args=SimpleNamespace(gpu_uuid='synthetic',gpu_uuids=tuple('synthetic'+str(i) for i in range(devices)),model='synthetic',precision='native',method='temporal',batch=2,cap=100,validate_pacing=validation)
    result=m.measure(args,compute,None,first,tmp_path/'synthetic_cpu_fixture',SimpleNamespace(fast=lambda:None),duty)
    assert len(result['repeats'])==3
    for rep in result['repeats']:
        assert rep['elapsed_s']==pytest.approx(rep['compute_s']+rep['internal_wait_s'])
        assert rep['energy_j']==pytest.approx((338.2+100*devices)*rep['elapsed_s'])
        assert rep['excluded_preblock_s']>0
        assert rep['wall_elapsed_s']>rep['elapsed_s']
        assert rep['video_evaluations']==rep['forward_calls']*2
        assert rep['internal_wait_s']>0 if duty<1 else rep['internal_wait_s']==0
    assert first.hook is None
    if devices==2:assert (0,) in synchronized and (1,) in synchronized
    assert json.loads((tmp_path/'synthetic_cpu_fixture/report.json').read_text())['timing_protocol']==m.PROTOCOL
    calibration=tmp_path/'synthetic_cpu_fixture/calibration_validation01.json'
    assert calibration.exists()==(validation and duty<1)
    if calibration.exists():
        assert json.loads(calibration.read_text())['not_formal_measurement']
        assert result['pacing_calibration']=='full_window_validation_v1'
