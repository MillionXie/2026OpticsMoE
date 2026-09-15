import hashlib,json
from pathlib import Path
from .data import plan,signature
ROOT=Path(__file__).resolve().parents[2]
def require_scope(mode,stage=None):
    scope=json.loads((ROOT/'RUN_SCOPE.json').read_text(encoding='utf-8'))
    if mode not in scope['allowed_modes'] or (stage is not None and stage not in scope['allowed_stages']):
        raise RuntimeError('Current user scope allows only pilot A, B_only and M1; further experiments require new user confirmation')
    return scope

def verify_source():
    auth=json.loads((ROOT/'OFFICE_AUTHORIZATION.json').read_text())
    manifest=json.loads((ROOT/'OFFICE_MANIFEST.json').read_text())
    assert auth['approved'] and auth['plan_sha256']==signature(plan())
    assert auth['manifest_sha256']==signature(manifest)
    for name,sha in manifest.items():
        file=(ROOT/name).resolve()
        if not file.is_relative_to(ROOT) or hashlib.sha256(file.read_bytes()).hexdigest()!=sha:raise RuntimeError('Source mismatch: '+name)

def setup(seed,out):
    import torch
    from experiments.vision_transfer.settings import load_settings
    from experiments.vision_transfer.backbone import load_backbone
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import seed_everything
    s=load_settings(ROOT/'experiments/vision_transfer/config.yaml');s.output_dir=Path(out)
    s.classification_head_seed=seed+1000;s.router_optimization_seed=seed;s.split_seed=seed;s.num_workers=0
    s.run_final_test=False;s.inference_batch_size=60;s.batch_size=60
    torch.set_num_threads(8);seed_everything(seed)
    loaded=load_backbone(s,torch.device('cuda'))
    seed_everything(seed)
    return loaded,s

def build(loaded,s,architecture):
    from experiments.vision_transfer import model as M
    import torch
    class DiagnosticRouter(M.ReservedRouter):
        uniform_ablation=False
        def forward(self,fields):
            route=super().forward(fields)
            if self.uniform_ablation:route['weights']=torch.full_like(route['weights'],.5)
            return route
    r,h=M.build(loaded,s,architecture)
    if architecture=='moe':r.vision_surrogate.core.optical_branch.core.router.__class__=DiagnosticRouter
    return r,h
