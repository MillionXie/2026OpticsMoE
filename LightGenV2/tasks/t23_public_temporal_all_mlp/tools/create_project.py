"""Create the independent MLP project from the approved frontend components."""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil


ROOT=Path(__file__).resolve().parents[4]
TASK=ROOT/'LightGenV2/tasks/t23_public_temporal_all_mlp'
PROJECT=ROOT/'LightGen-II-MLP'
SOURCE=ROOT/'LightGen-II'
if (PROJECT/'model.py').exists():
    raise FileExistsError('The independent model already exists')


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


source=(SOURCE/'model.py').read_text('utf-8')
tree=ast.parse(source)
classes={node.name:ast.get_source_segment(source,node) for node in tree.body if isinstance(node,ast.ClassDef)}
frontend=('VisionFrontGeometry','VisionFrameBatch','QwenSpatialLayout','QwenPatchProjection',
          'QwenPositionInterpolation','VisionSpatialPooling','FrozenVisionFront','QwenFrontCheckpoint')
components=('VisionAdapter','LanguageAdapter','VisionMerger','QualityReadout','Router','Experts')
config='''@dataclass(frozen=True)
class ModelConfig:
    width: int = 192
    global_hidden: int = 32
    router_hidden: int = 128
    readout_hidden: int = 32
    prompt_length: int = 38
    videos: int = 16
    frames: int = 4
    tokens: int = 49
    temperature: float = 1.1

    def __post_init__(self):
        dimensions=(self.width,self.router_hidden,self.readout_hidden,self.prompt_length,
                    self.videos,self.frames,self.tokens)
        if dimensions!=(192,128,32,38,16,4,49):
            raise ValueError('Checkpoint dimensions differ from the MLP reference architecture')
        if type(self.global_hidden) is not int or self.global_hidden<=0:
            raise ValueError('The global MLP hidden width must be a positive integer')
        if not math.isfinite(self.temperature) or self.temperature<=0:
            raise ValueError('The router temperature must be finite and positive')
'''
global_mlp='''class GlobalMLP(nn.Module):

    def __init__(self,config,token_count,output_width):
        super().__init__()
        self.token_count=token_count
        self.input_width=config.width
        self.output_width=output_width
        self.fc1=nn.Linear(token_count*config.width,config.global_hidden)
        self.activation=nn.ReLU()
        self.fc2=nn.Linear(config.global_hidden,token_count*output_width)

    def forward(self,tokens):
        if tuple(tokens.shape[-2:])!=(self.token_count,self.input_width):
            raise ValueError('Global MLP input dimensions differ')
        leading=tokens.shape[:-2]
        flattened=tokens.flatten(-2)
        hidden=self.activation(self.fc1(flattened))
        output=self.fc2(hidden)
        return output.reshape(*leading,self.token_count,self.output_width)
'''
network=classes['LightGen2'].replace("Global(config, 'vision')",'GlobalMLP(config, config.tokens, config.width)')
network=network.replace("Global(config, 'language')",'GlobalMLP(config, config.frames + config.prompt_length, 96)')
network=network.replace('    @torch.inference_mode()\n','')
loader='''def load_lightgen2(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['schema']!=3 or payload['architecture']!=ARCHITECTURE:
        raise ValueError('Expected a LightGen-II MLP global checkpoint')
    model=LightGen2(ModelConfig(**payload['config']))
    frontend=QwenFrontCheckpoint(payload,model.vision_frontend.geometry)
    model.load_state_dict(frontend.restore_state(model.vision_frontend),strict=True)
    return model.to(device).eval(),payload['metadata']
'''
header="from __future__ import annotations\n\nfrom dataclasses import dataclass\nimport math\n\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n\nARCHITECTURE = 'lightgen2_temporal_mlp_global_v1'\nQUALITY_WORDS = ('Bad', 'Poor', 'Fair', 'Good', 'Excellent')\n\n\n"
parts=[config]
for name in frontend:
    text=classes[name]
    if name in ('VisionFrontGeometry','VisionFrameBatch'):
        text='@dataclass(frozen=True)\n'+text
    parts.append(text)
parts.extend(classes[name] for name in components)
parts.extend((global_mlp,network,loader))
(PROJECT/'model.py').write_text(header+'\n\n\n'.join(parts)+'\n',encoding='utf-8',newline='\n')
for name in ('LICENSE','requirements.txt','licenses/QWEN_LICENSE.txt','licenses/NOTICE.txt'):
    target=PROJECT/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(SOURCE/name,target)
for name in ('input_for_review.npz','reference_mos.csv'):
    target=PROJECT/'data'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    shutil.copy2(SOURCE/'data'/name,target)
protected={p.relative_to(ROOT).as_posix():file_hash(p) for p in SOURCE.rglob('*') if p.is_file()}
for relative in ('LightGenV2/tasks/t22_public_temporal_mlp/configs/selected.json',
                 'LightGenV2/tasks/t22_public_temporal_mlp/runs/simulation/current/LightGen.pt',
                 'LightGenV2/tasks/t22_public_temporal_mlp/releases/LightGen2.zip'):
    protected[relative]=file_hash(ROOT/relative)
out=TASK/'runs/smoke/initialization_20261117'
out.mkdir(parents=True,exist_ok=True)
(out/'protected_files.json').write_text(json.dumps(protected,indent=2),encoding='utf-8')
(out/'source_components.json').write_text(json.dumps(dict(base_checkpoint_sha256=file_hash(SOURCE/'weights/LightGen2.pt'),
                                                       inherited_classes=list(frontend+components)),indent=2),encoding='utf-8')
print(json.dumps(dict(project=str(PROJECT),model_lines=len((PROJECT/'model.py').read_text('utf-8').splitlines()),
                     protected_files=len(protected)),indent=2))
