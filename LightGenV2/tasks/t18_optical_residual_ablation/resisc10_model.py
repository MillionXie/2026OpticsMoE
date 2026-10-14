"""Same full-size optical graph, fixed retained10 detector coordinates."""
import inspect,textwrap
import torch
from caltech_repair234_model import CorrectedMoE
from resisc20_input_encoder import contract as input_contract
# Fixed before this campaign; unchanged coordinates from the original20 masks.
INDICES=(2, 3, 7, 8, 9, 10, 11, 16, 18, 19)
class SubsetMoE(CorrectedMoE):
    def __init__(self,depth,rho):
        super().__init__(depth,rho)
        self.detectors=self.detectors[list(INDICES)].clone()
# Reuse the exact frozen propagation/router/OEO body; only probability pseudocount
# follows the retained10 output windows. No learned head or phase-grid resize.
_source=textwrap.dedent(inspect.getsource(CorrectedMoE.forward))
assert _source.count('+20e-12')==1
_ns={'torch':torch};exec(compile(_source.replace('+20e-12','+15e-12'),__file__+':sealed10readout','exec'),_ns)
SubsetMoE.forward=_ns['forward']
def contract(depth,rho):
    v=input_contract(depth,rho,100)
    v.update(readout='fixed15 retained original20 equal64x64 windows, normalized selected energy; no learned electronic head',class_count=10,original_detector_indices=list(INDICES),retained_detector_coordinates=True)
    return v
