"""Fixed-scale camera fingerprints, not enhanced experimental CCD features.

Reject constant/dim/saturated frames, old/other phase matches and ambiguous banks.
Enrollment is a repeatability check, NOT independent proof of physical phase LUT.
"""
import hashlib
import json
import numpy as np
from PIL import Image

DEFAULTS={'minimum_pcc':.97,'minimum_margin':.01,'maximum_mean_ratio_error':.20,
          'minimum_std':2.,'minimum_p99':16.,'maximum_saturation_fraction':.01}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def settings_signature(meta):
    return {k:meta['camera'][k]['value'] for k in ('ExposureTime','Gain','AcquisitionFrameRate','PixelFormat')}

def fingerprint(raw,roi):
    a=np.asarray(raw)
    if a.dtype!=np.uint8 or a.ndim!=2:raise ValueError('Expected original Mono8 frame')
    x0,y0,x1,y1=map(int,roi)
    if not (0<=x0<x1<=a.shape[1] and 0<=y0<y1<=a.shape[0]):raise ValueError('Fingerprint ROI outside raw image')
    crop=a[y0:y1,x0:x1]
    vector=np.asarray(Image.fromarray(crop).resize((96,96),Image.Resampling.BOX),dtype=np.float64)
    stats={'std':float(crop.std()),'p99':float(np.percentile(crop,99)),
           'saturation_fraction':float((crop==255).mean()),'mean':float(vector.mean())}
    return vector,stats

def pcc(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():return -1.
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>1e-9 else -1.

def classify(vector,stats,references,target,thresholds=None):
    t={**DEFAULTS,**(thresholds or {})}
    if target not in references:raise ValueError('Target absent from reference bank')
    scores={k:pcc(vector,v) for k,v in references.items()}
    other=max((v for k,v in scores.items() if k!=target),default=-1.)
    refmean=float(np.asarray(references[target]).mean())
    ratio=abs(float(np.asarray(vector).mean())/max(refmean,1e-9)-1)
    reasons=[]
    if stats['std']<t['minimum_std'] or stats['p99']<t['minimum_p99']:reasons.append('insufficient_signal')
    if stats['saturation_fraction']>t['maximum_saturation_fraction']:reasons.append('saturation')
    if scores[target]<t['minimum_pcc']:reasons.append('target_mismatch')
    if scores[target]-other<t['minimum_margin']:reasons.append('old_or_ambiguous_phase')
    if ratio>t['maximum_mean_ratio_error']:reasons.append('brightness_drift')
    return {'passed':not reasons,'reasons':reasons,'scores':scores,'target':target,
            'margin':scores[target]-other,'mean_ratio_error':ratio,'stats':stats,'thresholds':t}

def validate_bank(samples,stats,thresholds=None):
    if len(samples)<2 or any(len(v)<3 for v in samples.values()):raise ValueError('Need >=2 phases and >=3 independent sweeps')
    references={k:np.median(v,axis=0) for k,v in samples.items()}
    checks=[]
    for k,frames in samples.items():
        for i,v in enumerate(frames):
            # Leave this capture OUT of its own reference (no self-correlation).
            refs=dict(references);refs[k]=np.median([f for j,f in enumerate(frames) if j!=i],axis=0)
            checks.append({'phase':k,'repeat':i,**classify(v,stats[k][i],refs,k,thresholds)})
    return references,{'passed':all(x['passed'] for x in checks),'checks':checks}
