"""Calibration-only camera aperture derivation and plots; no TEST/model fitting.

Three independent known input spots estimate an affine source-to-sensor map.
The fourth repeated spot is timing evidence, not an independent geometry fit.
Candidate aperture is separately saved; the supplied TXT is never overwritten.
"""
import argparse,json,hashlib
from pathlib import Path
import cv2
import numpy as np


def derive(points,old_corners):
    points=np.asarray(points,dtype=np.float32);old=np.asarray(old_corners,np.float32)
    assert points.shape==(4,2) and old.shape==(4,2)
    inverse=cv2.getPerspectiveTransform(np.float32([[0,0],[477,0],[477,477],[0,477]]),old)
    sensor=cv2.perspectiveTransform(points[None],inverse)[0]
    source=np.float32([[149.5,149.5],[329.5,149.5],[149.5,329.5]])
    affine=cv2.getAffineTransform(source,sensor[:3]);quad=cv2.transform(np.float32([[[0,0],[477,0],[477,477],[0,477]]]),affine)[0]
    assert (quad>=0).all() and (quad<2048).all()
    scales=[float(np.linalg.norm(points[k]-points[0])/180) for k in (1,2)]
    return dict(old_sensor_corners=old.tolist(),old_aperture_scale_in_logical_image=scales,candidate_sensor_corners=quad.tolist(),source_to_sensor_affine=affine.tolist(),fit_sensor_spots=sensor[:3].tolist(),TRAIN_TEST_used=False,source_positions=source.tolist(),applied_to_devices=False)


def run(a):
    from lab_caltech10_20261015 import corners
    cfg=json.loads(a.profile.read_text());orientation=json.loads(a.orientation.read_text());assert orientation['passed'] and orientation['selected']=='identity'
    assert not a.out.exists();a.out.mkdir(parents=True)
    result=derive(orientation['centroids'],corners(Path(cfg['coordinates'])))
    q=np.asarray(result['candidate_sensor_corners']);txt=a.out/'coordinates_measured_spots.txt'
    txt.write_text('\n'.join(f'{name} [{v[0]:.6f},{v[1]:.6f}]' for name,v in zip(('TL','TR','BR','BL'),q))+'\n')
    result.update(old_coordinates_SHA256=hashlib.sha256(Path(cfg['coordinates']).read_bytes()).hexdigest(),spot_proof_SHA256=hashlib.sha256(a.orientation.read_bytes()).hexdigest(),candidate_coordinates_SHA256=hashlib.sha256(txt.read_bytes()).hexdigest(),candidate_coordinates=str(txt.resolve()),new_aperture_requires_independent_spot_validation=True)
    (a.out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


def preview(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    result=json.loads((a.out/'report.json').read_text());raw=np.asarray(Image.open(a.raw));old=np.array(result['old_sensor_corners']);new=np.array(result['candidate_sensor_corners'])
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    axes[0].imshow(raw,vmin=0,vmax=65535,cmap='gray');axes[0].plot(*np.r_[old,old[:1]].T,'--',color='#E64B35',lw=1.4,label='Supplied aperture');axes[0].plot(*np.r_[new,new[:1]].T,'-',color='#00A6A6',lw=1.4,label='Spot-derived aperture');axes[0].legend(fontsize=8);axes[0].set_title('Native camera frame');axes[0].set_xlabel('Camera x (pixel)');axes[0].set_ylabel('Camera y (pixel)')
    for ax,quad,title in zip(axes[1:],(old,new),('Supplied aperture','Spot-derived aperture (proposed)')):
        warp=cv2.getPerspectiveTransform(quad.astype(np.float32),np.float32([[0,0],[477,0],[477,477],[0,477]]));crop=cv2.warpPerspective(raw.astype(np.float32),warp,(478,478),flags=cv2.INTER_AREA)
        ax.imshow(crop,vmin=0,vmax=65535,cmap='gray');ax.set_title(title);ax.set_xlabel('Logical x (pixel)');ax.set_ylabel('Logical y (pixel)')
    fig.savefig(a.out/'aperture_comparison.png',dpi=170);plt.close(fig)


def selftest():
    points=np.float32([[110.25,104.25],[378.45,104.25],[110.25,372.45],[110.25,104.25]])
    result=derive(points,[[590,615],[1418,615],[1418,1437],[590,1437]])
    assert np.allclose(result['old_aperture_scale_in_logical_image'],[1.49,1.49],atol=1e-6)
    assert np.array(result['candidate_sensor_corners']).shape==(4,2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path);p.add_argument('--orientation',type=Path);p.add_argument('--out',type=Path);p.add_argument('--raw',type=Path);p.add_argument('--selftest',action='store_true');a=p.parse_args()
    if a.selftest:selftest();print('passed')
    elif a.raw:preview(a)
    else:run(a)
