"""Centroid/distribution diagnostics only; never rewrite raw images or config."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def centroid(a):
    smooth=gaussian_filter(a.astype(float),2);floor=float(np.median(smooth))
    w=np.maximum(smooth-floor-.25*(smooth.max()-floor),0);yy,xx=np.indices(w.shape)
    return [float((w*xx).sum()/w.sum()),float((w*yy).sum()/w.sum())] if w.sum() else None

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    rows=[];images={}
    for f in sorted(a.directory.glob('[0-9][0-9]_*.png')):
        im=np.array(Image.open(f));images[f.stem]=im
        rows.append({'file':f.name,'centroid_xy':centroid(im),'maximum':int(im.max()),'saturation_fraction':float((im==255).mean())})
    report={'method':'Gaussian sigma2 for centroid localization only, above 25% peak over median background; RAW files unchanged',
            'phase_lut_linearity_verified':False,'rows':rows}
    print(json.dumps(report,indent=2))
    (a.directory/'centroids.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
