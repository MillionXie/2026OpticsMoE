"""Independent separate-view KD gradient, teacher identity and RGB encoding."""
import types,numpy as np,torch
import caltech20_rich_augmented_teacher46 as m
import check_caltech20_rich_rare_zoomout46 as previous
import audit_caltech20_rich_augmented_teacher46 as audit
for name in ('independent_teacher','original_run','inherited','numpy_transform','run'):
    f=getattr(previous,name);ns=dict(f.__globals__);ns.update(m=m,audit=audit)
    for k in ('independent_teacher','original_run','inherited','numpy_transform'):
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
prior=run
def run():
    prior();cfg=m.teacher_source.config();assert m.s.sha(cfg['teacher_checkpoint'])==cfg['teacher_checkpoint_sha256'];r=np.random.default_rng(39);n=5;zc=torch.tensor(r.normal(size=(n,20)),requires_grad=True);za=torch.tensor(r.normal(size=(n,20)),requires_grad=True);qc=torch.tensor(r.uniform(.1,1,(n,20)));qc=qc/qc.sum(1,keepdim=True);qa=torch.tensor(r.uniform(.1,1,(n,20)));qa=qa/qa.sum(1,keepdim=True);mc=torch.tensor(r.uniform(.1,1,n));ma=torch.tensor(r.uniform(.1,1,n));loss=.5*m.old.kd.KD_loss(zc.softmax(1),qc,mc,dict(temperature=4))+.5*m.old.kd.KD_loss(za.softmax(1),qa,ma,dict(temperature=4));gc,ga=torch.autograd.grad(loss,(zc,za))
    for z,q,mass,g in ((zc,qc,mc,gc),(za,qa,ma,ga)):assert float((g-.5*4*mass[:,None]*(torch.softmax(z.detach()/4,1)-q)/n).abs().max())<1e-12
    image=torch.arange(2*3*150*150).remainder(251).byte().reshape(2,3,150,150);theta=torch.eye(2,3).repeat(2,1,1);rgb=m.teacher_RGB100(image,theta);mosaic=torch.cat((torch.cat((rgb[:,0:1],rgb[:,1:2]),3),torch.cat((rgb[:,2:3],rgb.mean(1,keepdim=True)),3)),2);import resisc20_input_encoder as encoder
    actual=encoder.rms_encode(torch.nn.functional.interpolate(mosaic,(224,224),mode='bilinear',align_corners=False)[:,0]);expected=m.base.encode(image,theta,100);assert torch.equal(actual,expected) and m.TEACHER is None
    p=m.CPU/'checks.json';proof=m.s.read(p);proof.update(independent_augmented_teacher_KD_gradient=True,teacher_RGB_matches_student_encoder=True,official_teacher_checkpoint_SHA256=cfg['teacher_checkpoint_sha256'],teacher_real_models=0,teacher_real_forwards=0,synthetic_RGB_encoding_inputs=2);m.s.save(p,proof)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
