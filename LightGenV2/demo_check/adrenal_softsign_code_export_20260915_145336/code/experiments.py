from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent
CONFIG=yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
EXP=CONFIG['experiment'];SEARCH=CONFIG['search'];VARIANTS=CONFIG['variants']
TOTAL_RUNS=SEARCH['formal_runs']
CONFIGS={v['id']:yaml.safe_load((ROOT/'configs'/f"{v['id']}.yaml").read_text()) for v in VARIANTS}
assert len(VARIANTS)==24 and TOTAL_RUNS==60
for v in VARIANTS:
    c=CONFIGS[v['id']];d=v['depth'];on=v['oeo']
    assert c['variant']==v
    assert c['nonlinearity']['type']=='intensity_layernorm_activation'
    assert c['nonlinearity']['reencoding']['amplitude_source']=='activation_output'
    assert c['nonlinearity']['activation']['type'] in SEARCH['candidates']
    assert c['experiment']['oeo_enabled']==c['nonlinearity']['enabled']==c['global_oeo']['enabled']==on
    assert c['experiment']['baseline_layers']==d
    assert c['model']['num_cycles']==c['expert_bank']['d2nn']['num_layers']==c['global_fc']['num_masks']==d//2
    assert c['expert_bank']['d2nn']['nonlinear_schedule']==[on]*(d//2)
    for k,value in EXP.items():assert c['experiment'][k]==value
def run_dir(v,seed):return ROOT/'runs'/v['id']/f'seed{seed}'
def parameters(v):return CONFIGS[v['id']]['experiment']['expected_parameters'][v['architecture']]
def formal_variants(selected):return [v for v in VARIANTS if v['activation'] in [selected,'off']]
