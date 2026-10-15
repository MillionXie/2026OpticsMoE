"""Session-only radiometric inverse; no hardware, learning or image normalization."""
import numpy as np


def monotone(values):
    blocks=[]
    for value in values:
        blocks.append([float(value),1])
        while len(blocks)>1 and blocks[-2][0]>blocks[-1][0]:
            right=blocks.pop();left=blocks.pop();n=left[1]+right[1]
            blocks.append([(left[0]*left[1]+right[0]*right[1])/n,n])
    return np.concatenate([np.full(n,v) for v,n in blocks])


def inverse_field(grays,means):
    x=np.asarray(grays,float);y=monotone(means)
    assert x[0]==0 and x[-1]==255 and np.all(np.diff(x)>0)
    assert y[-1]-y[0]>1,'Insufficient measured dynamic range'
    response=np.clip((y-y[0])/(y[-1]-y[0]),0,1)
    unique,first=np.unique(response,return_index=True)
    drive=np.rint(np.interp((np.arange(256)/255.)**2,unique,x[first])).astype(int)
    drive[0]=0;drive[-1]=255
    assert np.all(np.diff(drive)>=0)
    return drive.tolist(),response.tolist()
