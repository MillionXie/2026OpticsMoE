"""Original admissions plus independent NumPy synthetic augmentation pixels."""
import types,numpy as np,torch
import caltech20_rich_appearance_mix46 as m
import check_caltech20_rich_currentKD46 as previous
import audit_caltech20_rich_appearance_mix46 as audit
for name in ('independent_teacher','original_run','run'):
    f=getattr(previous,name);ns=dict(f.__globals__);ns.update(m=m,audit=audit)
    for k in ('independent_teacher','original_run'):
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
inherited=run
def numpy_transform(images,plan):
    result=[]
    for raw,row in zip(images,plan):
        x=raw.astype(np.float32)/255;branches=[]
        for chain in row['chains']:
            z=x.copy()
            for op,v in chain:
                if op==0:z=np.power(np.clip(z,0,1),v)
                elif op==1:
                    gray=z.mean(0,keepdims=True);z=gray+v*(z-gray)
                elif op==2:
                    lo=z.min((1,2),keepdims=True);span=z.max((1,2),keepdims=True)-lo;stretch=np.where(span>1e-6,(z-lo)/np.maximum(span,1e-6),z);z=(1-v)*z+v*stretch
                else:
                    p=np.pad(z,((0,0),(1,1),(1,1)),mode='reflect');blur=sum(p[:,a:a+150,b:b+150] for a in range(3) for b in range(3))/9;z=blur+v*(z-blur)
                z=np.clip(z,0,1)
            branches.append(z)
        z=(1-row['mix'])*x+row['mix']*sum(w*v for w,v in zip(row['weights'],branches))
        if row['erase']:
            y0,y1,x0,x1=row['box'];z[:,y0:y1,x0:x1]=x.mean((1,2),keepdims=True)
        result.append(np.rint(np.clip(z,0,1)*255).astype(np.uint8))
    return np.stack(result)
def run():
    inherited();images=torch.arange(4*3*150*150,dtype=torch.int64).remainder(251).byte().reshape(4,3,150,150);before=images.clone();maximum=0
    for erase in (False,True):
        plan=m.appearance_plan(4,1,0,erase);plan[0]['chains']=[[[0,.7],[1,1.5]],[[2,.75],[3,1.75]]];plan[0]['erase']=erase;plan[0]['box']=[10,40,20,50];proxy=m.AugmentedBatchOnly(images,np.arange(4),plan);aug=proxy[np.arange(4)];clean=proxy[np.arange(4)];expected=numpy_transform(images.numpy(),plan);error=int(np.abs(aug.numpy().astype(int)-expected.astype(int)).max());maximum=max(maximum,error);assert error<=1 and torch.equal(clean,images) and torch.equal(before,images) and not torch.equal(aug,clean)
    proof=m.s.read(m.CPU/'checks.json');proof.update(independent_numpy_appearance_pixels=True,maximum_uint8_rounding_difference=maximum,clean_view_exactly_unchanged=True,source_images_unchanged=True,synthetic_images_only=True);m.s.save(m.CPU/'checks.json',proof)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
