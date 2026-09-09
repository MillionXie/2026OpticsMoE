"""Physical-coordinate amplitude resampling and wrapped-phase nearest raster."""
import argparse
import numpy as np
from PIL import Image,ImageDraw
from common import ROOT,STAGES,config,write,sha,hardware_identity

def orient(a,c):
    if c.get('flip_vertical'): a=np.flipud(a)
    if c.get('flip_horizontal'): a=np.fliplr(a)
    return np.ascontiguousarray(a)

def raster(a,device,c,phase=False):
    h,w=a.shape
    if (h,w)!=(478,478): raise ValueError('Expected canonical 478 x 478')
    pitch=device['pixel_pitch_um']; width,height=device.get('size_wh',device.get('expected_resolution_wh'))
    cx,cy=device['center_xy']; extent=478*c['model_pitch_um']/pitch
    if cx-extent/2<-.5 or cy-extent/2<-.5 or cx+extent/2>width-.5 or cy+extent/2>height-.5:
        raise ValueError('Physical 8.126 mm active aperture does not fit SLM')
    a=orient(np.asarray(a,np.float32),device)
    xx=(np.arange(width,dtype=np.float64)-cx)*pitch/c['model_pitch_um']+238.5
    yy=(np.arange(height,dtype=np.float64)-cy)*pitch/c['model_pitch_um']+238.5
    validx=(xx>=-.5)&(xx<477.5); validy=(yy>=-.5)&(yy<477.5)
    if phase:
        sampled=a[np.clip(np.floor(yy+.5).astype(int),0,477)[:,None],np.clip(np.floor(xx+.5).astype(int),0,477)[None,:]]
        out=np.floor(np.mod(sampled,2*np.pi)/(2*np.pi)*256).clip(0,255).astype(np.uint8)
    else:
        # Bilinear amplitude (NOT intensity, NOT wrapped phase); preserve FOV.
        x=np.clip(xx,0,477); y=np.clip(yy,0,477)
        x0=np.floor(x).astype(int); y0=np.floor(y).astype(int)
        dx=x-x0; dy=y-y0
        top=a[y0[:,None],x0[None,:]]*(1-dx)+a[y0[:,None],np.minimum(x0+1,477)[None,:]]*dx
        bottom=a[np.minimum(y0+1,477)[:,None],x0[None,:]]*(1-dx)+a[np.minimum(y0+1,477)[:,None],np.minimum(x0+1,477)[None,:]]*dx
        out=np.rint(np.clip(top*(1-dy[:,None])+bottom*dy[:,None],0,255)).astype(np.uint8)
    out[~validy,:]=0; out[:,~validx]=0
    return out

def amplitude(a,c):
    x=np.asarray(a,np.float32)
    if not np.isfinite(x).all() or (x<0).any(): raise ValueError('Invalid amplitude')
    positive=x[x>0]; scale=float(np.percentile(positive,99.5)) if positive.size else 1
    encoded=np.rint(np.clip(x/max(scale,1e-12),0,1)*255).astype(np.uint8)
    out=raster(encoded,c['amplitude_slm'],c)
    lut=c['amplitude_slm'].get('gray_lut_file')
    if lut:
        table=np.loadtxt(ROOT/lut)
        if table.shape!=(256,) or not np.isfinite(table).all() or table.min()<0 or table.max()>255: raise ValueError('gray LUT must contain 256 values in [0,255]')
        out=np.rint(table[out]).astype(np.uint8)
    return out,{'positive_percentile':99.5,'scale':scale,'amplitude_interpolation':'bilinear_physical_17_to_8_um'}

def save(path,a):
    path.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(a).save(path)

def calibration(c):
    dest=ROOT/'generated/cal'; a=c['amplitude_slm']; p=c['phase_slm']
    aw,ah=a['expected_resolution_wh']; pw,ph=p['size_wh']
    save(dest/'A_WHITE.bmp',np.full((ah,aw),255,np.uint8)); save(dest/'A_BLACK.bmp',np.zeros((ah,aw),np.uint8))
    save(dest/'P_ZERO.bmp',np.zeros((ph,pw),np.uint8))
    yy,xx=np.indices((478,478))
    for cell in (32,64): save(dest/f'A_CHECK_{cell}.bmp',raster(((xx//cell+yy//cell)%2*255).astype(np.float32),a,c))
    asym=np.zeros((478,478),np.float32); asym[35:180,35:95]=255; asym[120:180,35:240]=255
    save(dest/'A_L.bmp',raster(asym,a,c))
    for name,(x,y) in {'TL':(0,0),'TR':(430,0),'BR':(430,430),'BL':(0,430)}.items():
        marker=np.zeros((478,478),np.float32); marker[y:y+48,x:x+48]=255
        save(dest/f'A_{name}.bmp',raster(marker,a,c))
    Y,X=np.indices((ph,pw),dtype=float)
    for name,axis in [('X',X),('Y',Y)]:
        save(dest/f'P_GRAT_{name}.bmp',np.rint((axis%24)/24*255).astype(np.uint8))
    cx,cy=p['center_xy']; span=478*c['model_pitch_um']/p['pixel_pitch_um']
    manifest={'effective_width_um':8126,'effective_width_device_px':span,'f_m':.1,'wavelength_nm':532,
        'phase_background':'zero phase, not an opaque aperture','amplitude_for_fresnel':'A_WHITE.bmp',
        'warning':'Fresnel arrays locate logical field boundaries; zero-order background can remain with full-white amplitude. Not diffraction-free synthetic crosses.',
        'arrays':{}}
    for n,grid in [(1,[0.0]),(4,[-.5,.5]),(9,[-.5,0,.5])]:
        phase=np.zeros((ph,pw),float); centers=[]
        window=c['fresnel_single_window_px'] if n==1 else c['fresnel_corner_window_px']
        for vy in grid:
            for vx in grid:
                fx=cx+vx*span; fy=cy+vy*span
                # Teacher's unwrapped negative quadratic, rectangular patches.
                if fx-window/2<-.5 or fy-window/2<-.5 or fx+window/2>pw-.5 or fy+window/2>ph-.5:
                    raise ValueError('Fresnel sublens would be clipped; reduce corner window or correct phase center.')
                region=(np.abs(X-fx)<window/2)&(np.abs(Y-fy)<window/2)
                phase[region]=-np.pi*((X[region]-fx)**2+(Y[region]-fy)**2)*(p['pixel_pitch_um']*1e-6)**2/(c['wavelength_nm']*1e-9*c['distance_m'])
                centers.append([fx,fy])
        save(dest/f'P_F{n}.bmp',np.rint(np.mod(phase,2*np.pi)/(2*np.pi)*255).astype(np.uint8))
        manifest['arrays'][str(n)]={'centers_xy':centers,'square_window_px':window}
    write(dest/'geometry.json',manifest)
    # Fast diagnostic: 32 values, 3 frames each; user chooses valid exposure first.
    for g in np.rint(np.linspace(0,255,32)).astype(int):
        patch=np.zeros((478,478),np.float32); patch[39:439,39:439]=g
        save(dest/f'gray/G{g:03d}.bmp',raster(patch,a,c))
    print('Calibration BMPs:',dest)

def export(c):
    phase_source=ROOT/'assets/phases'
    out=ROOT/'generated/P'; hashes={}
    for i,stage in enumerate(STAGES,1):
        source=phase_source/(stage+'.npy')
        reference=ROOT/'original_optics/reference_phases'/(stage+'.npy')
        if reference.is_file(): source=reference
        phase=np.load(source,allow_pickle=False)
        target=out/f'{i:02d}_{stage}.bmp'; save(target,raster(phase,c['phase_slm'],c,phase=True)); hashes[stage]=sha(target)
    write(out/'manifest.json',{'phase_sha256':hashes,'hardware_identity':hardware_identity(c),
        'quantization':'original checkpoint export: uint8 floor(mod(phi,2pi)/2pi*256); nearest phase sampling',
        'phase_flips':{k:c['phase_slm'][k] for k in ('flip_vertical','flip_horizontal')}})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config'); args=parser.parse_args()
    cfg,_=config(args.config); calibration(cfg); export(cfg)
