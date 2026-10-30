"""SDK-free, fail-closed six-plane acquisition ledger. Never owns a device."""
import hashlib
import json
from pathlib import Path
import numpy as np

STAGES=('vision_router','vision_expert','vision_global','language_router','language_expert','language_global')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def array_sha(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def encode(amplitude,phase):
    """Logical codes only; phase floor256, amplitude nearest255. No renormalization."""
    a,p=np.asarray(amplitude,dtype=np.float32),np.asarray(phase,dtype=np.float32)
    if a.shape!=(1,478,478)or p.shape!=a.shape or not all(np.isfinite(x).all()for x in(a,p)):
        raise ValueError('Only finite B1 canonical478 payloads')
    if a.min()<0 or a.max()>1.000001:raise ValueError('Require already once-bounded amplitude')
    amp=np.rint(a.clip(0,1)*255).astype(np.uint8)
    phase=np.floor(np.remainder(p.astype(np.float64),2*np.pi)*256/(2*np.pi)).clip(0,255).astype(np.uint8)
    return a,p,amp,phase


class NeedCapture(Exception):
    """Controlled prefix stop, never a request for ideal/FFT fallback."""


class _StageAdmission:
    """Process-local proof of one whole-cohort prefix check, not a saved bypass."""
    def __init__(self,ledger,stage):
        self.ledger=ledger;self.stage=stage;self.key=ledger.key


class Ledger:
    def __init__(self,root,session,models,requests,*,create=False):
        self.root=Path(root);self.session=session;self.models=models;self.requests=requests
        if session.get('schema')!='scene_lab_session_v1' or not session.get('calibration_approved'):
            raise ValueError('A reviewed, explicit session is required; no inherited defaults')
        required={'session_id','camera','wavelength_nm','logical_pitch_um','distance_m','amplitude_pitch_um',
                  'phase_pitch_um','amplitude_native_shape','phase_native_shape','mapping_sha256','orientation',
                  'exposure_us','gain','wait_ms','phase_lut_sha256','amplitude_inverse_sha256',
                  'dark_sha256','gain_map_sha256','intensity_units','calibration_evidence_sha256','scope'}
        if not required<=session.keys() or any(session[k]in('',None)for k in required):raise ValueError('Incomplete calibration identity')
        if session['scope']not in('offline_synthetic','reviewed_physical'):raise ValueError('Explicit physical vs synthetic scope')
        for k in('wavelength_nm','logical_pitch_um','distance_m','amplitude_pitch_um','phase_pitch_um','exposure_us'):
            if not np.isfinite(session[k])or session[k]<=0:raise ValueError('Invalid geometry/exposure')
        if not np.isfinite(session['wait_ms'])or session['wait_ms']<0:raise ValueError('Invalid wait')
        for k in('amplitude_native_shape','phase_native_shape'):
            if len(session[k])!=2 or any(type(v)is not int or v<=0 for v in session[k]):raise ValueError('Invalid native shape')
        for k in required:
            if k.endswith('_sha256')and(len(session[k])!=64 or any(x not in '0123456789abcdef'for x in session[k])):raise ValueError(k)
        role=session.get('cohort_role','train')
        if role not in ('train','dev'):raise ValueError('Only TRAIN or explicit evaluation-only DEV; never TEST')
        if role=='dev':
            manifest=session.get('evaluation_manifest_sha256','')
            if session.get('evaluation_only')is not True or len(manifest)!=64 or any(x not in '0123456789abcdef'for x in manifest):
                raise ValueError('DEV requires an explicit frozen evaluation-only manifest')
        if not requests or len({r['sample_id']for r in requests})!=len(requests)or any(r['split']!=role for r in requests):
            raise ValueError('This ledger requires unique requests from its one explicit cohort role')
        if any(set(r)-{'sample_id','split','reference','reference_sha256','prompt','seed','category'}for r in requests):
            raise ValueError('No targets/masks/hidden/features/SKU captions in requests')
        if set(models)!={'F9','F6','source_commit'}or any(len(models[k])!=64 for k in('F9','F6')):raise ValueError('Frozen model identities required')
        self.contract=dict(schema='scene_lab_ledger_v1',session=session,models=models,requests=requests,
                           encoding='round255_amplitude_floor256_total_phase_v1',stages=STAGES,
                           acquisition_recomputes_prefix=True,inference_Qwen_calls_per_request=1)
        self.key=identity(self.contract)
        if create:
            self.root.mkdir(parents=True,exist_ok=False);put(self.root/'contract.json',self.contract)
        else:
            if identity(json.loads((self.root/'contract.json').read_text(encoding='utf-8')))!=self.key:raise ValueError('Session/model/cohort changed')

    def folder(self,stage,request):
        if stage not in STAGES or request not in self.requests:raise ValueError('Unknown stage/request')
        # Paths never use untrusted sample_id text.
        return self.root/stage/identity(request)

    def receipt(self,stage,request):
        folder=self.folder(stage,request);r=json.loads((folder/'receipt.json').read_text())
        ticket=json.loads((folder/'ticket.json').read_text())
        if r['ticket_sha256']!=identity(ticket)or r['ledger_sha256']!=self.key or sha(folder/'calibrated.npy')!=r['calibrated_file_sha256']or sha(folder/'raw.npy')!=r['raw_file_sha256']:
            raise ValueError('Changed receipt/ticket/frame')
        value=np.load(folder/'calibrated.npy',allow_pickle=False)
        if value.shape!=(1,478,478)or value.dtype!=np.float32 or not np.isfinite(value).all()or value.min()<0:raise ValueError('Invalid CCD')
        return r,value

    def ready(self,stage):
        index=STAGES.index(stage)
        for previous in STAGES[:index]:
            for request in self.requests:self.receipt(previous,request)

    def admit_stage(self,stage):
        """Check every predecessor once; accept still hashes this sample's prefix.

        Acquisition is exclusive and accepted data is never overwritten. Re-reading
        the entire cohort for EVERY incoming frame was quadratic in cohort size.
        The local capability cannot be serialized/reused by another Ledger/stage.
        Each accepted frame additionally revalidates its own full prefix lineage.
        """
        self.ready(stage)
        return _StageAdmission(self,stage)

    def prepare(self,stage,runner,*,admission=None):
        """runner(request, numpy_provider) performs native graph prefix replay.

        Whole cohort at layer k must be accepted before preparing k+1. Native
        prefixes are recomputed for acquisition only, never counted as one-forward
        deployment timing. A final inference calls runner exactly once.
        """
        if admission is None:self.ready(stage)
        elif not isinstance(admission,_StageAdmission)or admission.ledger is not self or admission.stage!=stage or admission.key!=self.key:
            raise ValueError('Wrong process-local stage admission')
        pending=[]
        for request in self.requests:
            if(self.folder(stage,request)/'receipt.json').exists():
                self.receipt(stage,request)
                continue  # Resume preserves valid frames; never request a blind recapture.
            order=[];prefix=[]
            def provider(current,amplitude,phase,ideal=None):
                if ideal is not None or current!=STAGES[len(order)]:raise ValueError('Ideal fallback/wrong order forbidden')
                a,p,ag,pg=encode(amplitude,phase);folder=self.folder(current,request)
                ticket=dict(ledger_sha256=self.key,request_sha256=identity(request),stage=current,
                    prefix_receipts=list(prefix),amplitude_sha256=array_sha(a),total_phase_sha256=array_sha(p),
                    logical_amplitude_code_sha256=array_sha(ag),logical_phase_code_sha256=array_sha(pg))
                order.append(current)
                if STAGES.index(current)<STAGES.index(stage):
                    saved=json.loads((folder/'ticket.json').read_text())
                    if ticket!=saved:raise ValueError('Prefix input changed; invalidate downstream, do not reuse CCD')
                    receipt,value=self.receipt(current,request);prefix.append(identity(receipt));return value
                if current!=stage:raise ValueError('Runner continued beyond pending layer')
                if(folder/'ticket.json').exists():
                    if json.loads((folder/'ticket.json').read_text())!=ticket:raise ValueError('Pending field changed')
                    with np.load(folder/'logical_fields.npz',allow_pickle=False)as z:
                        if not all(np.array_equal(z[k],v)for k,v in [('amplitude',a),('total_phase',p),('amplitude_code',ag),('phase_code',pg)]):raise ValueError('Changed pending arrays')
                else:
                    folder.mkdir(parents=True,exist_ok=True)
                    with(folder/'logical_fields.npz').open('xb')as f:np.savez_compressed(f,amplitude=a,total_phase=p,amplitude_code=ag,phase_code=pg)
                    put(folder/'ticket.json',ticket)
                pending.append(dict(sample_id=request['sample_id'],ticket_sha256=identity(ticket),folder=str(folder)))
                raise NeedCapture(current)
            try:runner(request,provider)
            except NeedCapture:pass
            else:raise ValueError('Native runner did not pause at requested CCD boundary')
        return pending

    def accept(self,stage,request,raw,metadata,dark,gain_map,*,admission=None):
        """Import a manually supplied canonical camera array, not device capture.

        raw/dark in camera linear units; gain_map maps dark-subtracted values to
        fixed simulation intensity units. Same reviewed maps for all frames.
        """
        if admission is None:self.ready(stage)
        elif not isinstance(admission,_StageAdmission)or admission.ledger is not self or admission.stage!=stage or admission.key!=self.key:
            raise ValueError('Wrong process-local stage admission')
        folder=self.folder(stage,request);ticket=json.loads((folder/'ticket.json').read_text())
        if ticket['ledger_sha256']!=self.key or ticket['stage']!=stage or ticket['request_sha256']!=identity(request):raise ValueError('Ticket ownership changed')
        prefix=[identity(self.receipt(previous,request)[0])for previous in STAGES[:STAGES.index(stage)]]
        if ticket['prefix_receipts']!=prefix:raise ValueError('Accepted frame prefix lineage changed')
        with np.load(folder/'logical_fields.npz',allow_pickle=False)as z:
            if any(array_sha(z[k])!=ticket[target]for k,target in [('amplitude','amplitude_sha256'),('total_phase','total_phase_sha256'),('amplitude_code','logical_amplitude_code_sha256'),('phase_code','logical_phase_code_sha256')]):raise ValueError('Payload hash changed')
        expected=dict(ticket_sha256=identity(ticket),session_sha256=identity(self.session),stage=stage,
                      request_sha256=identity(request),canonical_orientation=self.session['orientation'],
                      exposure_us=self.session['exposure_us'],gain=self.session['gain'],wait_ms=self.session['wait_ms'])
        if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong frame/session/request/settings')
        raw=np.asarray(raw);dark=np.asarray(dark);gain_map=np.asarray(gain_map)
        if any(v.shape!=(1,478,478)or not np.isfinite(v).all()for v in(raw,dark,gain_map))or(gain_map<=0).any()or(raw<0).any():raise ValueError('Invalid camera/map array')
        if array_sha(dark)!=self.session['dark_sha256']or array_sha(gain_map)!=self.session['gain_map_sha256']:raise ValueError('Unreviewed dark/gain maps')
        provenance='synthetic_fixture'if self.session['scope']=='offline_synthetic'else'physical_linear_camera'
        if metadata.get('frame_provenance')!=provenance or not metadata.get('native_frame_sha256') or not metadata.get('capture_UTC'):
            raise ValueError('Explicit physical frame provenance/UTC required')
        if metadata.get('canonical_raw_sha256')!=array_sha(raw):raise ValueError('Raw array provenance mismatch')
        calibrated=((raw.astype(np.float32)-dark.astype(np.float32))*gain_map.astype(np.float32)).clip(0)
        if not np.isfinite(calibrated).all():raise ValueError('Calibration overflow')
        if(folder/'receipt.json').exists():raise FileExistsError('Never overwrite an accepted frame')
        with(folder/'raw.npy').open('xb')as f:np.save(f,raw,allow_pickle=False)
        with(folder/'calibrated.npy').open('xb')as f:np.save(f,calibrated,allow_pickle=False)
        receipt=dict(**expected,metadata=metadata,ledger_sha256=self.key,raw_file_sha256=sha(folder/'raw.npy'),calibrated_file_sha256=sha(folder/'calibrated.npy'),fixed_dark_gain=True,no_per_frame_peak_normalization=True)
        put(folder/'receipt.json',receipt);return receipt

    def infer(self,request,runner):
        for stage in STAGES:self.receipt(stage,request)
        order=[];prefix=[]
        def provider(stage,amplitude,phase,ideal=None):
            if ideal is not None or stage!=STAGES[len(order)]:raise ValueError('No FFT/ideal or order fallback')
            a,p,ag,pg=encode(amplitude,phase);folder=self.folder(stage,request);ticket=json.loads((folder/'ticket.json').read_text())
            expected=dict(ledger_sha256=self.key,request_sha256=identity(request),stage=stage,prefix_receipts=list(prefix),amplitude_sha256=array_sha(a),total_phase_sha256=array_sha(p),logical_amplitude_code_sha256=array_sha(ag),logical_phase_code_sha256=array_sha(pg))
            if ticket!=expected:raise ValueError('Final replay input identity changed')
            r,v=self.receipt(stage,request);order.append(stage);prefix.append(identity(r));return v
        result=runner(request,provider)
        if tuple(order)!=STAGES:raise ValueError('All six measured stages required')
        return result
