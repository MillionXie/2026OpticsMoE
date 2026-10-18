"""Independent NumPy bilinear resize-and-pad and unchanged clean-view proof."""
import types,numpy as np,torch
import caltech20_rich_rare_zoomout46 as m
import check_caltech20_rich_currentKD46 as previous
import audit_caltech20_rich_rare_zoomout46 as audit
for name in ('independent_teacher','original_run','run'):
    f=getattr(previous,name);ns=dict(f.__globals__);ns.update(m=m,audit=audit)
    for k in ('independent_teacher','original_run'):
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
inherited=run
def numpy_transform(images,plan):
    result=[]
    for image,row in zip(images,plan):
        if not row['active']:result.append(image.copy());continue
        z=image.astype(float);n=row['side'];pos=(np.arange(n)+.5)*150/n-.5;a=np.floor(pos).astype(int);t=pos-a;a=np.clip(a,0,149);b=np.clip(a+1,0,149);small=(1-t)[None,:,None]*((1-t)[None,None,:]*z[:,a[:,None],a[None,:]]+t[None,None,:]*z[:,a[:,None],b[None,:]])+t[None,:,None]*((1-t)[None,None,:]*z[:,b[:,None],a[None,:]]+t[None,None,:]*z[:,b[:,None],b[None,:]]);border=np.concatenate((z[:,0,:],z[:,-1,:],z[:,:,0],z[:,:,-1]),1).mean(1);out=np.broadcast_to(border[:,None,None],z.shape).copy();y,x=row['y'],row['x'];out[:,y:y+n,x:x+n]=small;result.append(np.rint(np.clip(out,0,255)).astype(np.uint8))
    return np.stack(result)
def run():
    inherited();cfg=m.config();labels=np.r_[np.zeros(160,dtype=int),np.ones(68,dtype=int)];idx=np.array([0,160]);plan=m.zoom_plan(labels,idx,1,0,cfg);assert [q['TRAIN_class_support'] for q in plan]==[160,68];plan[0]['active']=True;plan[1]['active']=False;images=torch.arange(228*3*150*150,dtype=torch.int64).remainder(251).byte().reshape(228,3,150,150);before=images.clone();proxy=m.AugmentedBatchOnly(images,idx,plan);aug=proxy[idx];clean=proxy[idx];error=int(np.abs(aug.numpy().astype(int)-numpy_transform(images[idx].numpy(),plan).astype(int)).max());assert error<=1 and torch.equal(clean,images[idx]) and torch.equal(images,before) and not torch.equal(aug[0],clean[0]) and torch.equal(aug[1],clean[1]);proof=m.s.read(m.CPU/'checks.json');proof.update(independent_numpy_zoomout_pixels=True,maximum_uint8_rounding_difference=error,clean_view_exactly_unchanged=True,TRAIN_counts_not_VALID_or_TEST_adaptation=True,source_images_not_mutated=True);m.s.save(m.CPU/'checks.json',proof)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
