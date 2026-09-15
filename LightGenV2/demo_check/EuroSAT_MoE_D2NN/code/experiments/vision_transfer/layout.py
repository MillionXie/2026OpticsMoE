"""One spatial contract for phase windows, CCD regions and task ownership.

Coordinates use the simulation's common (y down, x right) orientation.
Hardware camera mirroring/rotation must be calibrated before physical replay.
"""
import torch

A_EXPERTS = (0, 1)
B_EXPERTS = (2, 3)
POSITIONS = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
CCD_WINDOW_SIZE = 60


def configure_detector_windows(settings, geometry):
    """Center every router CCD integration window on its matching expert tile."""
    a=geometry.active_aperture
    intervals=[]
    for i in (0,1):
        tile=geometry.expert_apertures[i]
        center=(tile.x0+tile.x1)//2-a.x0
        intervals.append((center-CCD_WINDOW_SIZE//2,center+CCD_WINDOW_SIZE//2))
    settings.optical_router_detector_intervals=tuple(intervals)


def spatial_table(geometry, router):
    """CCD ROIs are local to the 478 square active crop; phase ROIs to 518 canvas."""
    return [dict(expert=i, position=POSITIONS[i], task='A' if i in A_EXPERTS else 'B',
                 ccd_xyxy=list(router.detector_bounds[i]),
                 phase_xyxy=[a.x0,a.y0,a.x1,a.y1], weight_index=i)
            for i,a in enumerate(geometry.expert_apertures)]


def check_layout(geometry, router):
    geometry.validate()
    if (geometry.grid_rows,geometry.grid_cols)!=(2,2):
        raise RuntimeError('This protocol requires a 2x2 expert grid')
    rows=spatial_table(geometry,router)
    for i,row in enumerate(rows):
        x0,y0,x1,y1=row['ccd_xyxy']
        side=(int((y0+y1)/2 >= geometry.active_size/2),int((x0+x1)/2 >= geometry.active_size/2))
        a=geometry.expert_apertures[i]
        crop=geometry.active_aperture
        if ((x0+x1)/2,(y0+y1)/2)!=((a.x0+a.x1)/2-crop.x0,(a.y0+a.y1)/2-crop.y0):
            raise RuntimeError(f'CCD center and expert center differ at E{i}')
        phase_side=(int((a.y0+a.y1)/2 >= geometry.canvas_size/2),int((a.x0+a.x1)/2 >= geometry.canvas_size/2))
        if side!=divmod(i,2) or phase_side!=divmod(i,2):
            raise RuntimeError(f'CCD/expert position mismatch at E{i}')
        expected=torch.zeros_like(router.detector_masks[i]);expected[y0:y1,x0:x1]=1
        if not torch.equal(expected,router.detector_masks[i]):
            raise RuntimeError(f'CCD mask/index mismatch at E{i}')
    return rows
