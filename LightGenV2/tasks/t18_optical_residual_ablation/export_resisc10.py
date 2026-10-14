"""Ten retained windows metadata around the sealed native BMP exporter."""
import inspect,textwrap
from pathlib import Path
import export_caltech_repair234 as e
from resisc10_model import contract,INDICES
import resisc20 as s
_source=textwrap.dedent(inspect.getsource(e.export_selected))
for old,new in (('len(classes)==20 and len(set(classes))==20','len(classes)==10 and len(set(classes))==10'),('class_label=list(range(20))','class_label=list(range(10))')):
    assert _source.count(old)==1;_source=_source.replace(old,new)
_ns=dict(e.__dict__);_ns['deployment_contract']=contract;exec(compile(_source,__file__+':sealed10export','exec'),_ns)
def export_selected(model,cfg,out,checkpoint,identity,classes,sources):
    assert cfg['input_tile']==100 and len(model.detectors)==10
    v=_ns['export_selected'](model,cfg,out,checkpoint,identity,classes,sources)
    v.update(contract=contract(model.depth,model.rho),original_detector_indices=list(INDICES),export_source_sha256=s.sha(__file__),physical_export_source_sha256=s.sha(e.__file__),frozen_encoder_source_sha256=s.sha(Path(__file__).with_name('resisc20_input_encoder.py')),hardware_sequence='Whole dataset router then4main stages; normalize10 retained original coordinate detector energies; offline calibration required',input_api='resisc20_input_encoder.encode(images,theta,100)')
    s.save(Path(out)/'manifest.json',v);return v
