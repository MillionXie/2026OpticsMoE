"""Both models fill the first phase aperture from the SAME 50-pixel RGB tiles."""
from pathlib import Path
import copy,sys
import kather_followup as f
from kather_followup import b,m,k,r,torch,np
from models import RelayFanoutMoE

HERE=Path(__file__).resolve()
BASE_SOURCES=f.sources()
BASE_BUILD=m.build

class CoveredRelay(RelayFanoutMoE):
    def global_fanout_convolution(self,field,prompt_transmission):
        ap=self.layout.input_aperture
        source=field[:,ap.y0:ap.y1,ap.x0:ap.x1]
        assert not torch.is_complex(source) or float(source.imag.abs().max())==0
        # Each tile enlarged separately; avoid interpolation across RGB boundaries.
        image=m.enlarge(source.real[:,None],self.layout.expert_size)[:,0]
        weights=torch.stack([prompt_transmission[:,*p.center].abs() for p in self.layout.expert_apertures],1)
        amplitude=weights/weights.square().sum(1,keepdim=True).sqrt().clamp_min(1e-12)
        output=torch.zeros_like(field)
        for i,p in enumerate(self.layout.expert_apertures):output[:,p.y0:p.y1,p.x0:p.x1]=image*amplitude[:,i,None,None]
        return output

def build(arch,depth,cfg):
    model=BASE_BUILD(arch,depth,cfg)
    if arch.startswith('moe') and cfg.get('expert_input_coverage')=='full':
        assert isinstance(model.net,RelayFanoutMoE);model.net.__class__=CoveredRelay
    return model

def sources():
    result=dict(BASE_SOURCES);result[HERE.relative_to(b.TASK).as_posix()]=r.sha(HERE);return result

def smoke(a):
    f.smoke(a)
    data=k.load_data(a.data,'train');records=[]
    for depth in [2,4,6]:
        for arch in ['moe','moe_nooeo']:
            cfg=f.config('long');r.setseed(17);old=BASE_BUILD(arch,depth,cfg);r.setseed(17);model=build(arch,depth,cfg)
            assert all(torch.equal(p,dict(old.named_parameters())[n]) for n,p in model.named_parameters())
            x=b.encode(data[0][:8]);net=model.net;field=torch.zeros(8,500,500,dtype=torch.complex64,device='cuda');ap=net.layout.input_aperture;field[:,ap.y0:ap.y1,ap.x0:ap.x1]=x[:,0]
            transmission=torch.ones_like(field);out=net.global_fanout_convolution(field,transmission)
            err=float(((out.abs().square().sum((1,2))-x.square().sum((1,2,3))).abs()/x.square().sum((1,2,3))).max());assert err<1e-6
            audit,_=m.pixel_audit(model,arch,data)
            first=[v['illuminated_exact_fraction'] for n,v in audit.items() if 'experts.' in n and n.endswith('phase_layers.0')];assert len(first)==9 and min(first)==1
            clone=copy.deepcopy(model);assert isinstance(clone.net,CoveredRelay)
            with torch.no_grad():assert torch.equal(b.forward(model,x,arch)[0],b.forward(clone,x,arch)[0])
            records.append(dict(arch=arch,depth=depth,phase_audit=audit,power_relative_error=err,identical_initial_parameters=True,ema_copy_verified=True));del model,old,clone;torch.cuda.empty_cache()
    r.save(a.out/'coverage_smoke.json',dict(passed=True,records=records,sources=sources()))

original_smoke=f.smoke
# f.smoke is called inside the extended smoke, so dispatch it explicitly below.
f.SPEC=copy.deepcopy(f.SPEC)
f.SPEC['candidate_order']=['long','lower_capture']
f.SPEC['candidates']={name:dict(f.SPEC['candidates'][name],expert_input_coverage='full') for name in f.SPEC['candidate_order']}
f.SPEC['coverage_contract']='Same 50x50 source tiles; MoE enlarges each to73x73, tiles146x146, restores image power then routes. D2NN enlarges each to half its phase side and restores power. Router still receives100x100. No extra source-image information.'
f.HERE=HERE
f.sources=sources;k.sources=sources;m.sources=sources
m.build=build;b.build=build

if __name__=='__main__':
    if '--phase' in sys.argv and sys.argv[sys.argv.index('--phase')+1]=='smoke':
        import argparse
        p=argparse.ArgumentParser();p.add_argument('--phase');p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);smoke(p.parse_args())
    else:f.main()
