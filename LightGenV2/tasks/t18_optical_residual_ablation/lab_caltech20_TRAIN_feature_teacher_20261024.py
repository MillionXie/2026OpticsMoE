"""Read-only original simulation TRAIN feature cache, CPU, no camera/optimizer."""
import argparse,json,time
from pathlib import Path
import lab_caltech20_tucam_layerwise_20261023 as base


def run(profile,out):
    np,torch,m,loop=base.numerical_modules();assert not out.exists();out.mkdir();torch.set_num_threads(2)
    c=m.read(profile);base.validate_config(c);assert c['initial_SHA256']=='7b0d7f055a16f7f6bbf9858c9ecaea572ecd0b0d6e96c618630f4a4e077c09a5'
    data,dm=m.data_for(base.ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256']=='e6fae9878ddb8fb7b622aeb7d53d7eb2e224db4d95a69cdf20c017d25b2f7d1c'
    pt=base.ROOT/c['initial_PT'];reference=base.ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256'];model,_=m.model_for(pt,c['initial_SHA256'],4,reference);model.cpu().eval().requires_grad_(False)
    before=m.state_cpu(model);protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(profile),source_PT_SHA256=m.sha(pt),reference_SHA256=m.sha(reference),data_SHA256=dm['sha256'],role='train',TRAIN=1600,TEST_used=False,VALID_used=False,gradient=False,optimizer_updates=0,SDK_used=False,GPU_used=False)
    m.save(out/'protocol.json',protocol);features=[];logits=[];error=0.;start=time.monotonic()
    with torch.inference_mode():
        for begin in range(0,1600,8):
            images,n,end=m.padded(data,'train',begin);assert n==8;model.measured={};prob=model(m.encode_images(images))[0];f=model.readout['features'];z=model.linear(f);error=max(error,float((z.softmax(1)-prob).abs().max()));features.append(f.numpy().copy());logits.append(z.numpy().copy())
            if begin%80==0:m.save(out/'progress.json',dict(state='original_teacher_CPU_TRAIN_features',TRAIN_consumed=end,total=1600,TEST_used=False,elapsed_s=time.monotonic()-start))
    x=np.concatenate(features);z=np.concatenate(logits);assert x.shape==(1600,256) and z.shape==(1600,20) and np.isfinite(x).all() and np.isfinite(z).all() and error<=3e-6
    assert all(torch.equal(v,model.state_dict()[k])for k,v in before.items())
    cache=out/'teacher_TRAIN_features.npz';np.savez_compressed(cache,ids=data['train'][2],labels=data['train'][1],features=x,logits=z)
    report=dict(state='complete',**protocol,cache_SHA256=m.sha(cache),features_shape=list(x.shape),logits_shape=list(z.shape),TRAIN_correct=int((z.argmax(1)==data['train'][1]).sum()),native_feature_probability_maxabs=error,original_state_bitwise_unchanged=True,elapsed_s=time.monotonic()-start,protocol_SHA256=m.sha(out/'protocol.json'))
    m.save(out/'report.json',report);m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),cache_SHA256=m.sha(cache),TEST_used=False,all_parameters_bitwise_frozen=True));print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();run(args.profile,args.out)
