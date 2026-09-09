"""Keep measured Top-2; make its small probability margin diagnostic only."""
from copy import deepcopy
import numpy as np

POLICY = 'measured_top2_margin_warning_v1'


def assess(frame, route, contract, validator=None):
    a=np.asarray(frame)
    if a.shape!=(478,478) or a.dtype!=np.uint8:
        raise ValueError('Router capture must be canonical 478x478 uint8')
    probabilities=np.asarray(route['probabilities'],dtype=float)
    energies=np.asarray(route['detector_energy'],dtype=float)
    if probabilities.shape!=(1,4) or energies.shape!=(1,4) or not np.isfinite(probabilities).all() or not np.isfinite(energies).all():
        raise ValueError('Invalid router probabilities or energies')
    ordered=np.sort(probabilities[0])[::-1]; margin=float(ordered[1]-ordered[2])
    p01,p99=np.percentile(a,[1,99]); threshold=contract['router_quality']['minimum_topk_probability_margin']
    report={'policy':POLICY,'accepted':False,'hard_failure':None,
            'p01_uint8':float(p01),'p99_uint8':float(p99),'dynamic_range_uint8':float(p99-p01),
            'saturated_pixel_fraction':float((a>=255).mean()),
            'top2_probability_margin':margin,'top2_margin_warning_threshold':threshold,
            'warnings':['ambiguous_top2_margin'] if margin<threshold else [],
            'probabilities':probabilities[0].tolist(),'selected_indices':route['selected_indices'][0],
            'measured_routing_unchanged':True}
    # No persisted contract change. All the other original rejection criteria
    # are still evaluated by the original validator, not reimplemented here.
    effective=deepcopy(contract)
    effective['router_quality']['minimum_topk_probability_margin']=0.0
    if validator is None:
        from common import setup_imports
        setup_imports()
        from abo_dual.common import validate_router_capture
        validator=validate_router_capture
    try: validator(frame,route,effective)
    except ValueError as ex: report['hard_failure']=str(ex)
    else: report['accepted']=True
    return report


def require_accepted(report):
    if not report['accepted']:
        raise ValueError(f"{report['hard_failure']} (p99={report['p99_uint8']:g}, "
                         f"dynamic_range={report['dynamic_range_uint8']:g}, "
                         f"saturation={report['saturated_pixel_fraction']:.4g}); Top-2 margin is warning-only.")


def warn(report):
    if report['warnings']:
        print(f"Router warning only: p2-p3={report['top2_probability_margin']:.6g}; "
              f"using MEASURED Top-2 {report['selected_indices']}.",flush=True)
