"""Synthetic clock/scheduling fixtures only; not GPU benchmark evidence."""
import copy
import unittest
from lgvq_async_raw_adapter_20261026 import adapt, observations


def fixture():
    c=.2; clocks={'frontend':0.,'body':0.};rows=[];scores={}
    excluded=['optical_simulation']*6+['detector_ROI']*6+['electronic_residual']*4
    kinds=[]
    for kind in excluded:kinds += ['electronics',kind]
    kinds += ['electronics']
    durations=dict(frontend=.3,electronics=.05,optical_simulation=2.,detector_ROI=.1,electronic_residual=.2)
    absolute=lambda x:x/1000
    def cpu(duration):
        nonlocal c
        a=c;c+=duration;return absolute(a),absolute(c)
    def sync(dependency):
        nonlocal c
        a=c;c=max(c,dependency)+.003
        return dict(start=absolute(a),end=absolute(c))
    def command(kind,stream,dependency=0.,duration=None):
        a,b=cpu(.01)
        x=max(clocks[stream],a*1000,dependency)+.005;y=x+(durations[kind] if duration is None else duration)
        clocks[stream]=y
        return dict(kind=kind,label=kind,stream=stream,host_submit_start=a,host_submit_end=b,
                    cuda_start_ms=x,cuda_end_ms=y)
    for field in range(128):
        row=dict(field=field);a,b=cpu(.1)
        row['tokenizer']=dict(field=field,start=a,end=b,input_ids=[[9,8]],attention_mask=[[1,1]])
        if field>=5:
            row['slot']=(field-5)%2;s=sync(scores[field-2])
            row.update(slot_reuse_wait_host_start=s['start'],slot_reuse_wait_host_end=s['end'])
        if field in (3,4):
            row['mode']='capture_real';row['slot']=field-3
            s=sync(max(clocks.values()));s['kind']='device_before_buffers'
            row['synchronization_observations']=[s]
            cpu(.08)  # actual allocation/pinning host gap, not GPU waiting
        else:
            a,b=cpu(.05);row.update(pin_host_start=a,pin_host_end=b)
        h=command('frontend','frontend',duration=.02)
        row.update(H2D_host_submit_start=h['host_submit_start'],H2D_host_submit_end=h['host_submit_end'],
                   original_H2D_cuda_ms=[h['cuda_start_ms'],h['cuda_end_ms']])
        events=[]
        if field in (3,4):
            events.append(command('frontend','frontend',duration=.03))
            s=sync(clocks['frontend']);s['kind']='device_after_transfer';row['synchronization_observations'].append(s)
            a,b=cpu(.04)
            row['graph_construction']=[dict(kind='frontend',capture_host_start=a,capture_host_end=b)]
        front=command('frontend','frontend');events.append(front)
        if field<3:
            row['mode']='eager_real';row['body_host_start']=absolute(c)
            body=[]
            for kind in kinds:
                e=command(kind,'body',front['cuda_end_ms']);body.append(e)
                if kind!='electronics':events.append(e)
            row['body_host_end']=absolute(c)
            row['body_envelope_cuda_ms']=[body[0]['cuda_start_ms'],body[-1]['cuda_end_ms']]
        else:
            if field in (3,4):
                s=sync(front['cuda_end_ms']);s['kind']='frontend_before_body_capture';row['synchronization_observations'].append(s)
                cpu(.03)
                for kind in kinds:
                    a,b=cpu(.02);row['graph_construction'].append(dict(kind=kind,capture_host_start=a,capture_host_end=b))
            for kind in kinds:events.append(command(kind,'body',front['cuda_end_ms']))
            events.append(command('electronics','body',duration=.005))
        row['score_ready_cuda_ms']=clocks['body'];scores[field]=clocks['body'];row['events']=events
        if field<5:
            s=sync(scores[field]);s['kind']='score_ready'
            row.setdefault('synchronization_observations',[]).append(s)
        rows.append(row)
    rows[-1]['final_synchronization_observation']=sync(scores[127])
    return dict(records=rows,video_calls=2048,unique_videos=512,cycles=4,
                all_predictions_bitwise_equal=True,startup_and_both_actual_captures_included=True,
                host_start=0.,host_end=absolute(c),actual_async_software_wall_ms=c,
                origin_host_clock_bracket=[-1e-9,1e-9],reference_sha256='synthetic_only',
                raw_board_energy=dict(max_power_sample_age_s=.05,energy_j=123.))


class RawAdapterTests(unittest.TestCase):
    def test_complete_causal_trace_projection_and_actual_overlap(self):
        raw=fixture();trace,result=adapt(raw)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['fixed_optics_projection']['optical_passes'],768)
        self.assertEqual(sum(n['kind']=='optical_simulation' for n in trace['nodes']),768)
        self.assertGreater(result['measured_next_frontend_prior_body_overlap_ms'],0.)
        self.assertFalse(result['kernel_occupancy_concurrency_proven'])
        self.assertFalse(result['arbitrary_wall_padding'])
        self.assertTrue(result['phase_energy_proxy']['controller_counted_once'])
        self.assertEqual(result['native_effective_tokens'],256)
    def test_missing_real_sync_cannot_be_replaced_by_wall_padding(self):
        raw=fixture();del raw['records'][-1]['final_synchronization_observation']
        with self.assertRaisesRegex(ValueError,'terminal'):adapt(raw)
    def test_impossible_clock_is_rejected_not_retimed(self):
        raw=fixture();raw['origin_host_clock_bracket']=[100.,100.1]
        with self.assertRaisesRegex(ValueError,'clock alignment'):adapt(raw)
    def test_original_workload_and_capture_boundaries_required(self):
        raw=fixture();raw['records'][3]['graph_construction'].pop()
        with self.assertRaisesRegex(ValueError,'capture segments'):adapt(raw)
        raw=fixture();raw['records'].pop()
        with self.assertRaisesRegex(ValueError,'workload'):adapt(raw)
    def test_power_age_gate_not_weakened_for_LG(self):
        for value in [.25001,float('nan')]:
            raw=fixture();raw['raw_board_energy']['max_power_sample_age_s']=value
            with self.assertRaisesRegex(ValueError,'250'):adapt(raw)
    def test_missing_reuse_fence_or_original_segment_is_rejected(self):
        raw=fixture();del raw['records'][5]['slot_reuse_wait_host_start']
        with self.assertRaisesRegex(ValueError,'reuse fence'):adapt(raw)
        raw=fixture();raw['records'][5]['events'].pop()
        with self.assertRaisesRegex(ValueError,'segments'):adapt(raw)


if __name__=='__main__':unittest.main()
