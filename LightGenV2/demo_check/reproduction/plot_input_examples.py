"""Real data and measured input geometry. Does not invent biological images."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from analyze_oeo_suite import export,save,sha

CLASSES={'bloodmnist':['Basophil','Eosinophil','Erythroblast','Immature granulocyte','Lymphocyte','Monocyte','Neutrophil','Platelet'], 'kather2016':['Tumour epithelium','Simple stroma','Complex stroma','Immune cells','Debris / mucus','Mucosal glands','Adipose','Background']}
SOURCES={'bloodmnist':('https://zenodo.org/records/10519652','https://www.medmnist.com/'), 'kather2016':('https://zenodo.org/records/53169','https://huggingface.co/datasets/1aurent/Kather-texture-2016/tree/4fd718bff27676d01e291057824449d6d1f569d0')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',type=Path);a=p.parse_args();manifest=json.loads((a.input/'manifest.json').read_text());dataset=manifest['dataset'];a.out.mkdir(parents=True,exist_ok=False);assert sha(a.input/'input_examples.npz')==manifest['arrays_sha256'];z=np.load(a.input/'input_examples.npz',allow_pickle=False);names=CLASSES[dataset]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,8,figsize=(15,4.5))
    for k in range(8):
        for j in range(2):
            i=k*2+j;ax=axes[j,k];ax.imshow(z['rgb'][i],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel(str(z['sample_ids'][i]),fontsize=8)
            if j==0:ax.set_title(f'{k}: {names[k]}',fontsize=9,wrap=True)
    fig.suptitle(f'{dataset}: real training images ({z["rgb"].shape[1]} × {z["rgb"].shape[2]} RGB)');fig.tight_layout();export(fig,a.out,'real_training_examples')
    src=z['amplitude_100'][0,0];wide=z['amplitude_wide_L4'][0,0];smallcanvas=np.zeros((500,500));smallcanvas[200:300,200:300]=src;widecanvas=np.zeros((500,500));widecanvas[14:486,14:486]=wide
    # One expert's local input is measured; no fabricated routing prediction.
    expert=np.zeros((146,146));expert[23:123,23:123]=src
    fig,axes=plt.subplots(1,5,figsize=(15,4.2),layout='constrained');axes[0].imshow(z['rgb'][0],interpolation='nearest');axes[0].set_title('Original RGB')
    for ax,arr,title in zip(axes[1:],[src,smallcanvas,widecanvas,expert],['Fixed amplitude encoding\n[R, G; B, mean], 100 × 100','D2NN: central 100 × 100\n500 × 500 canvas','D2NN: full 472 × 472 input\n500 × 500 canvas','MoE: one 146 × 146 expert\ncentral 100 × 100 input']):
        im=ax.imshow(arr,cmap='gray',vmin=0,vmax=max(arr.max(),1e-12),interpolation='nearest');ax.set_title(title,fontsize=10)
    for ax in axes:ax.set_xticks([]);ax.set_yticks([])
    for ax in axes[2:4]:ax.add_patch(Rectangle((13.5,13.5),472,472,fill=False,edgecolor='#db514c',lw=1))
    axes[4].add_patch(Rectangle((-.5,-.5),146,146,fill=False,edgecolor='#db514c',lw=1))
    fig.suptitle('4-layer geometry; red outline = first trainable phase aperture\nAmplitudes independently scaled for display; numerical power is conserved. MoE branch routing scale omitted.',fontsize=11);export(fig,a.out,'input_geometry')
    if a.smoke:
        fig,axes=plt.subplots(2,3,figsize=(10,7),layout='constrained');evidence=[]
        for col,arch in enumerate(['d2nn','d2nn_wide','moe']):
            report=json.loads((a.smoke/f'{arch}_L4.json').read_text());name='phases.0.phase' if arch.startswith('d2nn') else next(n for n in report if 'expert' in n);rec=report[name];maps=np.load(a.smoke/f'{arch}_L4_maps.npz');key=rec['map_key']
            for row,(suffix,label,field) in enumerate([('power','Input intensity','illuminated_exact_fraction'),('class_gradient','Classification gradient magnitude','classification_gradient_exact_fraction')]):
                v=maps[key+'_'+suffix];im=axes[row,col].imshow(np.log10(np.maximum(v/(v.max()+1e-30),1e-8)),cmap='magma',vmin=-8,vmax=0);axes[row,col].set_title(f'{arch}: {100*rec[field]:.2f}% nonzero',fontsize=10);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                if col==0:axes[row,col].set_ylabel(label)
            evidence.append(dict(arch=arch,phase=name,**rec))
        fig.colorbar(im,ax=axes,shrink=.75,label='log10(value / map maximum)');fig.suptitle('Initial phase coverage: two balanced training batches; 4 main layers');export(fig,a.out,'initial_phase_coverage');save(a.out/'coverage_evidence.json',evidence)
    save(a.out/'figure_manifest.json',dict(input_manifest=manifest,plotter_sha256=sha(Path(__file__)),dataset_sources=SOURCES[dataset],license='CC BY 4.0; cite original dataset creators. Kather cache uses the explicitly pinned public mirror.',illustration_notes='Original images and actual preprocessing tensors. Geometry is not an optical device measurement. Each image amplitude panel has its own display scale. Coverage maps are measured numerical forward/backward quantities at initialization.'))
    print(str(a.out))
if __name__=='__main__':main()
