"""Read-only analysis of captured calibration frames; never edits/apply coordinates."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .scene_lab_protocol import put, sha


def review(folder,out):
    folder,out=Path(folder),Path(out);out.mkdir(parents=True,exist_ok=False)
    proof=json.loads((folder/'geometry.json').read_text());bg=np.load(folder/'dark/canonical.npy',allow_pickle=False)
    targets=np.asarray([r['expected']for r in proof['rows']]);observed=np.asarray([r['observed']for r in proof['rows']])
    a=np.column_stack([targets,np.ones(len(targets))]);fit=np.linalg.lstsq(a[:4],observed[:4],rcond=None)[0]
    residual=observed-a@fit;thresholds=[];panels=[]
    # A separate calibration proposal uses ONLY the three source-axis spots.
    # All five geometry points above remain held out from this proposed fit.
    import cv2
    from .scene_lab_hardware import corners
    protocol=json.loads((folder/'protocol.json').read_text());config=protocol['config']
    axis=json.loads((folder/'orientation.json').read_text())
    if axis['selected']!='identity':raise ValueError('Explicit orientation composition required for nonidentity session')
    source_axes=np.float64([[149.5,149.5,1],[329.5,149.5,1],[149.5,329.5,1]])
    observed_axes=np.asarray(axis['centroids'],dtype=np.float64)
    fit_axes=np.linalg.solve(source_axes,observed_axes[:3]);matrix=np.vstack([fit_axes.T,[0,0,1]])
    predicted=np.column_stack([targets,np.ones(len(targets))])@fit_axes
    holdout_error=np.linalg.norm(observed-predicted,axis=1)
    original_warp=cv2.getPerspectiveTransform(corners(config['coordinates']),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    proposed_warp=np.linalg.inv(matrix)@original_warp
    mapped=cv2.perspectiveTransform(observed.astype(np.float64)[None],np.linalg.inv(matrix))[0]
    admitted=bool(max(holdout_error)<3 and max(np.linalg.norm(mapped-targets,axis=1))<3 and axis['repeat_error']<1)
    mapping=dict(analysis_only=True,applied_to_capture=False,calibration_approved=admitted,original_coordinate_file=config['coordinates'],original_coordinate_SHA256=sha(config['coordinates']),
        original_warp=original_warp.tolist(),canonical_warp=proposed_warp.tolist(),source_axis_fit_points=source_axes[:,:2].tolist(),measured_axis_points=observed_axes[:3].tolist(),
        heldout_expected=targets.tolist(),heldout_observed=observed.tolist(),heldout_errors_before_rewarp_px=holdout_error.tolist(),heldout_errors_after_rewarp_px=np.linalg.norm(mapped-targets,axis=1).tolist(),
        native_amplitude_origin_xy=[452,32],native_phase_origin_xy=[452,92],active_shape=[1016,1016],logical_shape=[478,478],
        no_TEST_or_model_frames=True,no_physical_movement=True,no_device_LUT_edit=True,
        interpretation='Fixed calibration-plane spatial registration only. Near-unit scale with a displacement; root cause not uniquely identified. Never sample-specific ideal fitting.')
    put(out/'derived_mapping_proposal.json',mapping)
    for i,row in enumerate(proof['rows']):
        value=np.load(folder/f'geometry_{i}/canonical.npy',allow_pickle=False);w=(value.astype(float)-bg).clip(0);peak=np.percentile(w,99.7);y,x=np.indices(w.shape);centers=[]
        for threshold in (.1,.25,.5,.75):
            mask=np.where(w>=threshold*peak,w,0);point=[float((mask*x).sum()/mask.sum()),float((mask*y).sum()/mask.sum())];centers.append(dict(threshold=threshold,point=point))
        thresholds.append(dict(pattern=i,centers=centers))
        low,high=np.percentile(value,[1,99.8]);preview=np.rint(((value-low)/max(1,high-low)).clip(0,1)*255).astype(np.uint8)
        canvas=Image.new('RGB',(498,530),'white');canvas.paste(Image.fromarray(preview).convert('RGB'),(10,36));draw=ImageDraw.Draw(canvas)
        for point,color in [(row['expected'],'#ff4040'),(row['observed'],'#00ff90')]:
            xx,yy=point[0]+10,point[1]+36;draw.ellipse([xx-7,yy-7,xx+7,yy+7],outline=color,width=2);draw.line((xx-10,yy,xx+10,yy),fill=color,width=1);draw.line((xx,yy-10,xx,yy+10),fill=color,width=1)
        draw.text((10,8),f"Spot {i+1}; error {row['error_px']:.2f} logical pixels",fill='black')
        draw.text((10,515),'Red: expected; green: observed. Preview-only contrast stretch.',fill='black');panels.append(canvas)
    overview=Image.new('RGB',(1494,1060),'white')
    for i,panel in enumerate(panels):overview.paste(panel,((i%3)*498,(i//3)*530))
    draw=ImageDraw.Draw(overview);draw.text((1010,580),'Existing 20261029 coordinates, unchanged.',fill='black');draw.text((1010,610),'No model inference / TEST / exposure or LUT edits.',fill='black')
    draw.text((1010,640),'Approx. displacement: right 16, up 32 pixels.',fill='black');draw.text((1010,670),'Affine fit is diagnostic only; not applied.',fill='black');overview.save(out/'geometry_overview.png')
    files={str(p.relative_to(folder)):sha(p)for p in folder.rglob('*')if p.is_file()}
    put(out/'review.json',dict(analysis_only=True,coordinates_changed=False,model_changed=False,SDK_used=False,frames=11,calibration_file_SHA256=files,
        affine_expected_to_observed=fit.tolist(),fit_first_four_holdout_center=True,residuals_px=residual.tolist(),max_residual_px=float(np.linalg.norm(residual,axis=1).max()),
        mean_displacement_px=(observed-targets).mean(0).tolist(),threshold_sensitivity=thresholds,
        conclusion='Translation-dominated discrepancy in this current mapping; does not by itself identify a faulty camera, corner file, SLM center or optical tilt. Not admitted for model acquisition.'))
    print(json.dumps(dict(complete=True,analysis_only=True,mean_displacement_px=(observed-targets).mean(0).tolist(),max_affine_residual_px=float(np.linalg.norm(residual,axis=1).max()))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();review(a.folder,a.out)


if __name__=='__main__':main()
