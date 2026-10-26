"""Data-only critical-path contract for the future measured async LG worker.

This is preparation, NOT an asynchronous GPU benchmark or a measured speedup.
An admitted real trace must reproduce its original software wall time before
optical durations can be replaced. Summed-span subtraction is not allowed once
frontend and backbone overlap. Counterfactual physical timing remains labelled
as a projection; no hardware optical execution is claimed.
"""
import math

OPTICAL_PASS_MS=1000/1400+.300+.0307
KINDS={'tokenizer','frontend','electronics','dispatch','sync','optical_simulation',
       'detector_ROI','electronic_residual','capture_electronics','capture_excluded'}

def critical_path(nodes, durations=None):
    """Every stream/resource order and real dispatch gap must be explicit edges."""
    by_id={n['id']:n for n in nodes}
    if len(by_id)!=len(nodes):raise ValueError('Duplicate trace node')
    ends={};starts={};pending=dict(by_id)
    while pending:
        ready=[n for n in pending.values() if all(p in ends for p in n['parents'])]
        if not ready:raise ValueError('Cycle or missing dependency')
        for n in ready:
            measured=n['duration_ms']
            if n['kind'] not in KINDS or not isinstance(measured,(int,float)) or not math.isfinite(measured) or measured<0:
                raise ValueError('Unknown or invalid measured duration')
            duration=durations[n['id']] if durations is not None else measured
            start=max((ends[p] for p in n['parents']),default=0.)
            starts[n['id']]=start;ends[n['id']]=start+duration
            del pending[n['id']]
    return dict(wall_ms=max(ends.values(),default=0.),starts_ms=starts,ends_ms=ends)

def project(trace):
    """Fail closed on missing fields, tails, capture costs or timeline edges."""
    if trace.get('status')!='complete' or trace.get('all_predictions_bitwise_equal') is not True:
        raise ValueError('Complete original scoring evidence required')
    fields=trace['fields']
    if not isinstance(fields,int) or fields<1 or trace['video_calls']!=16*fields:
        raise ValueError('Exactly 16 videos / 64 frames per field required')
    if trace.get('all_startup_and_capture_retained') is not True or trace.get('new_inputs_each_field') is not True:
        raise ValueError('No discarded startup or cached features/predictions')
    nodes=trace['nodes'];original=critical_path(nodes)
    if not math.isclose(original['wall_ms'],trace['measured_software_wall_ms'],rel_tol=1e-7,abs_tol=1e-4):
        raise ValueError('Trace does not reproduce actual wall; missing scheduling/latency edges')
    optical={i:[] for i in range(fields)}
    for n in nodes:
        if n['kind']=='optical_simulation':
            if n.get('field') not in optical:raise ValueError('Optical field missing')
            optical[n['field']].append(n)
    if any(len(v)!=6 for v in optical.values()):raise ValueError('Exactly six original optical passes per field')
    durations={n['id']:(OPTICAL_PASS_MS if n['kind']=='optical_simulation' else
        0. if n['kind'] in {'detector_ROI','electronic_residual','capture_excluded'} else n['duration_ms']) for n in nodes}
    projected=critical_path(nodes,durations)
    return dict(status='COUNTERFACTUAL_FIXED_OPTICS_PROJECTION',actual_async_software_wall_ms=original['wall_ms'],
                projected_wall_ms=projected['wall_ms'],projected_time16_ms=projected['wall_ms']/fields,
                equivalent_frame_ms=projected['wall_ms']/(fields*64),fixed_pass_ms=OPTICAL_PASS_MS,
                optical_passes=fields*6,first_field_and_all_charged_capture_retained=True,
                physical_hardware_time_measured=False,subtract_overlapping_simulation_spans_from_wall=False,
                changed_optical_timing_may_change_GPU_contention=True,
                overlap_requires_actual_worker_trace=True)

PREPARED_PLAN={
    'status':'PREPARED_CONTRACT_NOT_YET_MEASURED',
    'scope':'LG Temporal only, original 16-video 64-frame coherent fields',
    'workload':{'calls':2048,'unique_TEST_videos':512,'cycles':4,'rounds_initially':1},
    'baseline_changes':False,
    'candidates':['online CPU tokenizer one field ahead with bounded queue',
                  'two independent input/prefix buffers; next-field conv/position/pool/text embedding/adapters',
                  'frontend stream overlaps prior-field original backbone when actual resources permit',
                  'bounded pending outputs and final GPU-score-ready synchronization instead of each-field sync'],
    'required_evidence':['every original score bitwise against sealed reference',
                         'all first fields and each actual graph construction included',
                         'CPU tokenizer/pinning/H2D and excluded image preparation recorded separately',
                         'actual CUDA start/end events, buffer reuse dependencies, CPU dispatch/sync and output ready',
                         'no cached embeddings, scores, detector outputs or substituted model branches'],
    'safety':'wait for current baseline suite/lock/private MPS release; no shared GPU timing windows',
    'reporting':'measured full-software async wall and fixed-optics critical-path projection separate',
    'not_done':['actual async worker integration','GPU validation and measured overlap','matched LG energy trace'],
}
