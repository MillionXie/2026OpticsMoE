"""Explicit acceptance of a saved real Router CCD; NEVER opens hardware."""
import argparse
import time
import numpy as np
from PIL import Image
from common import ROOT,config,read,write,sha,hardware_identity,setup_imports
from hardware import canonical
from router_quality import assess,require_accepted,warn


def recover(session,stage,c,accept_legacy_saved=False):
    from run import load_session,measured
    setup_imports(); from abo_dual.common import routing_from_ccd
    if stage not in ('vision_router','language_router'): raise ValueError('Only Router CCDs can be recovered')
    root,state=load_session(session,c); p=root/'play'/stage; mf=read(p/'manifest.json')
    if mf['stage']!=stage or mf['hardware_identity']!=hardware_identity(c):
        raise ValueError('Prepared stage/config identity mismatch')
    if sha(ROOT/mf['phase_file'])!=mf['phase_sha256']: raise ValueError('Prepared phase changed')
    contract=read(root/'optical_contract.json'); mapping={s['id']:s for s in state['samples']}
    recovered=[]; skipped=[]; absent=[]
    for e in mf['entries']:
        if e['id'] not in mapping: raise ValueError('Prepared sample not in session')
        if stage in measured(root,mapping[e['id']]): skipped.append(e['id']); continue
        out=root/'ccd'/e['id']/stage
        sources=[out.with_suffix(s) for s in ('.png','.tif','.json')]
        if not any(f.exists() for f in sources): absent.append(e['id']); continue
        if not all(f.is_file() for f in sources): raise ValueError('Incomplete saved capture: '+str(out))
        if sha(p/e['bmp'])!=e['sha256']: raise ValueError('Prepared amplitude changed')
        meta=read(out.with_suffix('.json'))
        if meta['hardware_identity']!=hardware_identity(c): raise ValueError('Saved CCD belongs to another hardware configuration')
        files={f.name:sha(f) for f in sources}
        pending=out.with_suffix('.capture.json')
        if pending.exists():
            provenance=read(pending)
            for key,expected in [('stage',stage),('sample',e['id']),('phase_sha256',mf['phase_sha256']),
                                 ('amplitude_sha256',e['sha256']),('hardware_identity',hardware_identity(c))]:
                if provenance[key]!=expected: raise ValueError('Saved capture provenance mismatch: '+key)
            if provenance['files']!=files: raise ValueError('Saved capture content changed')
            files[pending.name]=sha(pending); source='capture_time_hashes_verified'
        else:
            # Legacy failures predate .capture.json. Explicit approval is required;
            # never claim phase/BMP hashes were recorded at their capture time.
            if not accept_legacy_saved:
                raise ValueError('Legacy CCD has no capture-time phase/BMP hashes. Add --accept-legacy-saved only after confirming this saved image belongs to the prepared stage.')
            source='legacy_user_confirmed_stage_no_capture_time_phase_amplitude_hashes'
        with Image.open(sources[0]) as im: frame=np.asarray(im).copy()
        with Image.open(sources[1]) as im: raw=np.asarray(im).copy()
        if list(raw.shape)!=meta['raw_shape'] or str(raw.dtype)!=meta['raw_dtype']:
            raise ValueError('Raw TIFF and saved metadata disagree')
        if not np.array_equal(canonical(raw,c),frame): raise ValueError('Saved PNG is not the configured linear warp of its raw TIFF')
        route=routing_from_ccd(frame,contract); quality=assess(frame,route,contract)
        require_accepted(quality)
        qp=out.with_suffix('.quality.json'); write(qp,quality); files[qp.name]=sha(qp)
        rp=out.with_suffix('.route.json'); write(rp,route); files[rp.name]=sha(rp)
        write(out.with_suffix('.record.json'),{'stage':stage,'sample':e['id'],'files':files,
            'phase_sha256':mf['phase_sha256'],'amplitude_sha256':e['sha256'],
            'hardware_identity':hardware_identity(c),'capture_mode':'real','router_quality':quality,
            'recovery':{'method':'revalidate_saved_raw_and_canonical_CCD','source':source,
                        'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'hardware_opened':False}})
        warn(quality); recovered.append(e['id']); print('Recovered WITHOUT recapture:',e['id'],flush=True)
    report={'stage':stage,'recovered':recovered,'already_valid':skipped,'not_yet_captured':len(absent),'hardware_opened':False}
    write(root/'results'/f'recovery_{stage}.json',report); print(report,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',required=True); p.add_argument('--stage',required=True,choices=['vision_router','language_router'])
    p.add_argument('--config'); p.add_argument('--accept-legacy-saved',action='store_true')
    args=p.parse_args(); c,_=config(args.config)
    recover(args.session,args.stage,c,args.accept_legacy_saved)
