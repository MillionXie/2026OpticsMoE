import hashlib
import json
import zipfile
import pytest
from LightGenV2.tasks.t04_semantic_interaction.install_shs import safe_name,install,digest


@pytest.mark.parametrize('name',['../escape','/absolute','C:/drive','a\\b'])
def test_zip_escape_rejected(name):
    with pytest.raises(ValueError):safe_name(name)


def test_verified_install_and_no_overwrite(tmp_path):
    base=tmp_path/'base.zip';overlay=tmp_path/'overlay.zip';out=tmp_path/'project'
    value=b'known data';module=b'known code'
    with zipfile.ZipFile(base,'w') as z:
        z.writestr('old/data.bin',value)
        z.writestr('old/MANIFEST.json',json.dumps({'files':[{'path':'data.bin','sha256':hashlib.sha256(value).hexdigest()}]}))
    with zipfile.ZipFile(overlay,'w') as z:
        z.writestr('runtime/entry.py',module)
        z.writestr('SHS_SOURCE.json',json.dumps({'source_commit':'test','base_simulation_zip_sha256':digest(base),
                                              'files':{'runtime/entry.py':hashlib.sha256(module).hexdigest()}}))
    with pytest.raises(ValueError):install(base,overlay,'wrong',out)
    assert not out.exists()
    install(base,overlay,digest(overlay),out)
    assert (out/'data.bin').read_bytes()==value
    assert (out/'runtime/entry.py').read_bytes()==module
    with pytest.raises(FileExistsError):install(base,overlay,digest(overlay),out)


def test_six_boundary_order_replay_restore():
    try:
        import torch
    except (ImportError,OSError) as exc:
        pytest.skip('Local Torch unavailable: '+str(exc))
    from types import SimpleNamespace
    from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import OpticalBoundary,STAGES,replay
    class Prop(torch.nn.Module):
        def forward(self,x):return torch.fft.fft2(x,norm='ortho')
    class Phase(torch.nn.Module):
        def forward(self,x):return x*torch.exp(x.new_tensor(.3j))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.tensor(0.))
            g=SimpleNamespace(canvas_size=8,active_aperture=SimpleNamespace(y0=0,y1=8,x0=0,x1=8))
            for name in ('language','vision'):
                core=SimpleNamespace(geometry=g,propagator=Prop(),expert_layers=[Phase()],global_phase=Phase(),
                    router=SimpleNamespace(propagator=Prop(),active_phase=lambda:torch.full((8,8),.3)))
                setattr(self,name+'_core',SimpleNamespace(optical_branch=SimpleNamespace(core=core)))
        def forward(self,x,prompt):
            for name in ('language','vision'):
                core=getattr(self,name+'_core').optical_branch.core
                for prop in (core.router.propagator,core.propagator,core.propagator):
                    x=prop(x.to(torch.complex64)*torch.exp(x.new_tensor(.3).to(torch.complex64)*1j)).abs().square().sqrt()
            return x
    model=Model();batch={'source_image':torch.rand(1,8,8),'prompt_hidden':[]}
    expected=model(**{'x':batch['source_image'],'prompt':[]})
    out,tap=replay(model,batch)
    assert tuple(tap.amplitudes)==STAGES
    restored,_=replay(model,batch,tap.detectors)
    assert torch.allclose(out,restored,atol=1e-4)
    assert torch.allclose(model(batch['source_image'],[]),expected)
    with pytest.raises(ValueError):OpticalBoundary(model,{'vision_router':torch.ones(1,8,8)})
    _,partial=replay(model,batch,stop_before='language_expert')
    assert tuple(partial.amplitudes)==STAGES[:2]
