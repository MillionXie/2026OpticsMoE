"""Input-aware metadata around the unchanged full-size joint native BMP exporter."""
from pathlib import Path
import export_caltech_repair234 as e
import resisc20 as s
from resisc20_input_encoder import contract,encode

def export_selected(model,cfg,out,checkpoint,identity,classes,sources):
    assert cfg['input_tile'] in (50,100)
    value=e.export_selected(model,cfg,out,checkpoint,identity,classes,sources)
    value.update(contract=contract(model.depth,model.rho,cfg['input_tile']),
        frozen_encoder_source_sha256=e.sha(Path(__file__).with_name('resisc20_input_encoder.py')),
        export_source_sha256=e.sha(__file__),physical_export_source_sha256=e.sha(e.__file__),
        input_api='resisc20_input_encoder.encode(images,theta,input_tile); fixed RGB150 cache, no learned encoder')
    s.save(Path(out)/'manifest.json',value)
    return value
