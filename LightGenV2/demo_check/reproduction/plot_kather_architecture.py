"""Draw the actual RGB encoding, dense MoE and full-aperture D2NN contracts."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(15,9));ax.set(xlim=(0,15),ylim=(0,9));ax.axis('off')
    def box(x,y,w,h,text,c='#e4edf6',size=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',linewidth=1,edgecolor='#465567',facecolor=c));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
    def arrow(x,y,u,v):ax.annotate('',(u,v),(x,y),arrowprops=dict(arrowstyle='->',lw=1.5,color='#465567'))
    ax.text(.2,8.65,'Kather2016: image encoding and optical classification',fontsize=19,weight='bold')
    box(.2,7.15,2.05,1.05,'RGB image\nB × 3 × 150 × 150')
    box(2.8,7.15,2.45,1.05,'Resize to 100 × 100\nTrain: affine transform')
    box(5.8,7.15,2.45,1.05,'Resize each channel\nto 50 × 50')
    box(8.85,7.0,2.5,1.35,'R  |  G\nB  |  mean(R,G,B)\nOne 100 × 100 amplitude')
    box(12,7.15,2.7,1.05,'Zero input phase\nOne simulated wavelength')
    for x,u in [(2.25,2.8),(5.25,5.8),(8.25,8.85),(11.35,12)]:arrow(x,7.68,u,7.68)
    ax.text(.2,6.6,'D2NN: one serial optical path',fontsize=14,weight='bold')
    box(.2,5.05,3,1.2,'Enlarge each input tile\nFill 474 / 472 / 470 square\nRestore total input power')
    box(4,5.05,4.4,1.2,'[Phase plate → propagation → (OEO)]\nRepeat L = 2 / 4 / 6 times', '#eee8f6')
    box(9.2,5.05,2.4,1.2,'Final propagation\n500 × 500 field')
    box(12.25,5.05,2.45,1.2,'8 detector energies\nNormalize → argmax')
    for x,u in [(3.2,4),(8.4,9.2),(11.6,12.25)]:arrow(x,5.65,u,5.65)
    ax.text(.2,4.55,'MoE: one input-dependent routing step, then coupled expert/global cycles',fontsize=14,weight='bold')
    box(.2,2.95,3,1.2,'Optical router: 100 × 100\nPhase → propagation → 9 scores\nSoftmax → power allocation','#fae6e7',10)
    box(3.8,2.95,2.45,1.2,'4 tiles: 50 → 73 each\n146 × 146 expert input\nPower-matched 9-port relay','#fae6e7',10)
    box(6.85,2.75,4.75,1.6,'[9 local phases → propagation → (OEO)\n→ global phase → propagation → (OEO)]\nRepeat L/2 cycles\n146 × 146 experts; 498 × 498 global','#fae6e7',10)
    box(12.25,2.95,2.45,1.2,'Final propagation\nSame 8 detectors\nNormalize → argmax')
    for x,u in [(3.2,3.8),(6.25,6.85),(11.6,12.25)]:arrow(x,3.55,u,3.55)
    box(.2,.95,14.5,1.1,'OEO ON, after EVERY main layer: intensity |U|² → non-affine full-panel LayerNorm → ReLU → Softsign → amplitude, phase = 0\nOEO OFF: keep the complex field. No CNN/Qwen, electronic residual classifier, or learned electronic weights.', '#e6f1e9',11)
    ax.text(.25,.35,'L counts main optical layers; the MoE router adds one phase plate. Ideal relay and different local/global propagation remain modeling assumptions.',fontsize=10)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(a.out/('architecture.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
