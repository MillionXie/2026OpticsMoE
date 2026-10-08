"""CPU export of actual fixed preprocessing tensors, using training images only."""
import argparse,json
from pathlib import Path
import bloodmnist_multiseed as m
from bloodmnist_multiseed import b,r,np,torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--dataset',choices=['bloodmnist','kather2016'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    with np.load(a.data,allow_pickle=False) as z:
        raw=z['train_images'];y=z['train_labels'].reshape(-1);ids=z['train_ids'] if 'train_ids' in z else np.array([f'train_{i}' for i in range(len(y))]);keep=np.arange(len(y))
        if a.dataset=='bloodmnist':
            import hashlib
            seen={hashlib.sha256(x.tobytes()).hexdigest() for x in z['val_images']};keep=[]
            for i,x in enumerate(raw):
                h=hashlib.sha256(x.tobytes()).hexdigest()
                if h not in seen:keep.append(i);seen.add(h)
            keep=np.array(keep)
    indices=np.concatenate([keep[y[keep]==k][:2] for k in range(8)]);assert len(indices)==16
    rgb=raw[indices];labels=y[indices];sampleids=ids[indices];x=torch.from_numpy(rgb.transpose(0,3,1,2).copy()).float()/255;encoded=b.encode(x)
    arrays=dict(rgb=rgb,labels=labels,sample_ids=sampleids,amplitude_100=encoded.numpy())
    errors={}
    for depth,side in [(2,474),(4,472),(6,470)]:
        wide=m.enlarge(encoded,side);arrays[f'amplitude_wide_L{depth}']=wide.numpy();errors[str(depth)]=float(((wide.square().sum((1,2,3))-encoded.square().sum((1,2,3))).abs()/encoded.square().sum((1,2,3))).max());assert errors[str(depth)]<1e-6
    np.savez_compressed(a.out/'input_examples.npz',**arrays)
    r.save(a.out/'manifest.json',dict(dataset=a.dataset,data_sha256=r.sha(a.data),selection='First two retained training images in stored order from each class; no test images or predictions used.',ids=sampleids.tolist(),labels=labels.tolist(),raw_shape=list(rgb.shape),sources=m.sources(),exporter_sha256=r.sha(__file__),power_relative_errors=errors,arrays_sha256=r.sha(a.out/'input_examples.npz'),electronic_trainable_frontend=False,preprocessing='Actual b.encode and m.enlarge on CPU, without augmentation.'))
if __name__=='__main__':main()
