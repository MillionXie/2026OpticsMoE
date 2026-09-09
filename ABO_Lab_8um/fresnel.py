"""Numerical, contiguous Fresnel tiles; no resizing of wrapped-phase BMPs.

The three user-supplied 15/20/40 cm examples match positive quadratic phase
and floor(255 * fractional_turn) EXACTLY. This is the gray-ramp convention of
those files, not a claim that the SLM's calibrated physical phase has that sign.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from common import ROOT, config, write, sha


def encode(turns):
    return np.floor(np.mod(turns, 1.0) * 255.0).astype(np.uint8)


def four_array(c):
    """Centers are physical ROI corners; tiles meet at the ROI center lines.

    A complete 2x2 square array would have side 2*span. Clip only at panel edges,
    never shrink the ROI/center separation to make four complete circles fit.
    Coordinates are zero-based pixel centers, with panel edges at -0.5/W-0.5.
    """
    p = c['phase_slm']; w, h = p['size_wh']; cx, cy = p['center_xy']
    pitch = float(p['pixel_pitch_um'])
    span = c['model_active_pixels'] * c['model_pitch_um'] / pitch
    wavelength = float(c['wavelength_nm']) * 1e-9
    distance = float(c['distance_m'])
    if min(span, pitch, wavelength, distance) <= 0:
        raise ValueError('All optical scales must be positive')
    if not (-.5 <= cx-span/2 < cx+span/2 <= w-.5 and
            -.5 <= cy-span/2 < cy+span/2 <= h-.5):
        raise ValueError('ROI corner lens centers must fit on the phase SLM')
    y, x = np.indices((h, w), dtype=np.float64)
    # Equal-sized adjacent ideal square tiles, clipped to the physical panel.
    # owner order is physical TL, TR, BL, BR. No circular aperture or inner gap.
    owner = (x >= cx).astype(np.uint8) + 2*(y >= cy).astype(np.uint8)
    xs = np.array([cx-span/2, cx+span/2])
    ys = np.array([cy-span/2, cy+span/2])
    r2 = (x-xs[owner % 2])**2 + (y-ys[owner // 2])**2
    turns = r2 * (pitch*1e-6)**2 / (2*wavelength*distance)
    support = (x >= cx-span) & (x < cx+span) & (y >= cy-span) & (y < cy+span)
    step_x=np.abs(np.diff(turns,axis=1)); step_y=np.abs(np.diff(turns,axis=0))
    valid_x=support[:,1:] & support[:,:-1] & (owner[:,1:]==owner[:,:-1])
    valid_y=support[1:,:] & support[:-1,:] & (owner[1:,:]==owner[:-1,:])
    max_step=max(float(step_x[valid_x].max()),float(step_y[valid_y].max()))
    bmp = encode(turns); bmp[~support] = 0
    centers = [[float(fx), float(fy)] for fy in ys for fx in xs]
    logical = np.array([['TL','TR'], ['BL','BR']])
    if p.get('flip_vertical'): logical = np.flipud(logical)
    if p.get('flip_horizontal'): logical = np.fliplr(logical)
    metadata = {
        'layout': 'four_contiguous_square_tiles_clipped_at_panel_edges',
        'phase_size_wh': [w,h], 'phase_center_xy': [cx,cy],
        'pixel_pitch_um': pitch, 'wavelength_nm': c['wavelength_nm'],
        'distance_m': distance, 'effective_width_um': span*pitch,
        'center_spacing_px': span, 'centers_xy': centers,
        'coordinate_convention': 'zero-based pixel centers; continuous ROI edges',
        'physical_order': ['TL','TR','BL','BR'],
        'logical_labels_in_physical_order': logical.ravel().tolist(),
        'ideal_tile_width_px': span, 'ideal_array_width_px': 2*span,
        'ideal_array_bounds_xyxy': [cx-span,cy-span,cx+span,cy+span],
        'clipped_by_panel': bool(cx-span<-.5 or cy-span<-.5 or cx+span>w-.5 or cy+span>h-.5),
        'support_pixel_count': int(support.sum()), 'panel_pixel_count': int(w*h),
        'formula_turns': '+((x-xc)^2+(y-yc)^2)*pitch_m^2/(2*lambda_m*f_m)',
        'encoding': 'uint8 floor(255*mod(turns,1)); matches supplied BMP gray-ramp sign',
        'not_a_physical_phase_sign_calibration': True,
        'illumination': 'full-white amplitude SLM; not four amplitude holes',
        'no_internal_zero_phase_gap': True,
        'sampling': {'max_within_tile_phase_step_cycles_per_pixel':max_step,
                     'minimum_local_period_px':1/max_step,
                     'nyquist_limit_cycles_per_pixel':0.5,
                     'exceeds_nyquist':max_step>0.5,
                     'critical_axis_radius_px':wavelength*distance/(2*(pitch*1e-6)**2),
                     'warning':'10 cm plus this wide ROI produces sub-2-pixel outer fringes. Do not mistake correct center coordinates for alias-free optical focusing.'},
        'warning': 'Connected phase tiles do not guarantee four isolated pure CCD spots; full-white illumination and panel clipping affect diffraction.',
    }
    return bmp, owner, support, metadata


def audit_references(folder):
    """Read only: reconstruct the original files, including their MATLAB offset."""
    rows=[]
    for cm in (15,20,40):
        for path in sorted(Path(folder).glob(f'phase_fresnel_lens_4_{cm}cm_8um*.bmp')):
            with Image.open(path) as im: measured=np.asarray(im.convert('L'))
            if measured.shape != (1200,1920):
                raise ValueError('Unexpected reference size: '+str(path))
            y,x=np.indices((400,400),dtype=np.float64)
            # MATLAB meshgrid(1:400), center=400/2 => zero at local index 199.
            tile=encode(((x-199)**2+(y-199)**2)*(8e-6)**2/(2*532e-9*(cm/100)))
            expected=np.zeros_like(measured); expected[200:1000,560:1360]=np.tile(tile,(2,2))
            diff=np.abs(expected.astype(np.int16)-measured.astype(np.int16))
            rows.append({'file':path.name,'sha256':sha(path),'focal_length_cm':cm,
                         'max_gray_error':int(diff.max()),'matching_pixel_fraction':float((diff==0).mean()),
                         'centers_xy':[[759,399],[1159,399],[759,799],[1159,799]],
                         'center_spacing_px':400,'tile_width_px':400})
    return rows


def preview(path, bmp, meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(10.5,7.1),layout='constrained')
    ax.imshow(bmp,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    cx,cy=meta['phase_center_xy']; span=meta['center_spacing_px']
    ax.add_patch(Rectangle((cx-span/2,cy-span/2),span,span,fill=False,ec='#ffb000',lw=1.5))
    for label,(x,y) in zip(meta['logical_labels_in_physical_order'],meta['centers_xy']):
        ax.plot(x,y,'+',color='#00d8ff',ms=12,mew=1.8)
        ax.annotate(f'{label} ({x:.3f}, {y:.3f})',(x,y),xytext=(0,14 if y<cy else -23),
                    textcoords='offset points',ha='center',color='#00d8ff',fontsize=8,
                    bbox=dict(facecolor='black',alpha=.8,edgecolor='none',pad=1))
    ax.set(xlabel='Phase SLM x / px',ylabel='Phase SLM y / px',
           title=f'10 cm | 8 um | four contiguous tiles\nROI corner spacing: {span:g} px; outer lens areas clipped by panel')
    fig.savefig(path,dpi=160); plt.close(fig)


def generate(c, destination=None, source_config=None):
    dest=Path(destination) if destination else ROOT/'generated/cal/Phase_BMP'
    dest.mkdir(parents=True,exist_ok=True)
    bmp,owner,support,meta=four_array(c)
    target=dest/'P_F4_10cm.bmp'; Image.fromarray(bmp).save(target)
    aw,ah=c['amplitude_slm']['expected_resolution_wh']
    Image.fromarray(np.full((ah,aw),255,np.uint8)).save(dest/'A_WHITE.bmp')
    meta.update(phase_file=target.name,phase_sha256=sha(target),source_config=str(source_config),
                generator_sha256=sha(Path(__file__)),
                source_config_sha256=sha(source_config) if source_config else None,
                reference_audit=audit_references(dest))
    preview(dest/'P_F4_10cm_preview.png',bmp,meta)
    write(dest/'P_F4_10cm.geometry.json',meta)
    print('Fresnel BMP:',target)
    print('Centers (physical TL/TR/BL/BR):',meta['centers_xy'])
    print('Spacing:',meta['center_spacing_px'],'px; outer tiles clipped:',meta['clipped_by_panel'])
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config'); p.add_argument('--output-dir',type=Path)
    args=p.parse_args(); c,path=config(args.config); generate(c,args.output_dir,path)
