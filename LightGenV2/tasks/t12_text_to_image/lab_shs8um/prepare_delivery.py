"""Prepare fixed T12 small-model inputs without access to laboratory hardware."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples-per-mode',type=int,default=1,help='Predeclared examples per category and edit mode; 1 keeps original six-item pilot')
    a=p.parse_args()
    from LightGenV2.tasks.t12_text_to_image.sealed_editor import build_sealed
    from LightGenV2.tasks.t12_text_to_image.audited_unified import architecture_report,audited_settings
    from LightGenV2.tasks.t12_text_to_image.product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
    from LightGenV2.tasks.t12_text_to_image.qwen_mini_small import PromptEmbeddingLookup
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=build_sealed(saved).eval()
    audit=architecture_report(model)
    assert model.kind=='small' and audit['counted_parameters']==9958098
    settings=audited_settings()
    assert settings.language_optical_pixel_pitch_um==17 and settings.language_optical_distance_m==0.1
    data=a.assets/'datasets'
    dataset=ExpandedUnifiedProductEditDataset(data/'abo_cleanrender_lamp_table_pillow_256_v1','test',256,
        data/'abo_unified_expanded_instructions_qwen2_v2.pt')
    lookup=PromptEmbeddingLookup(data/'abo_unified_expanded_qwen_embeddings_v2.pt')
    if not 1<=a.samples_per_mode<=16:raise ValueError('samples-per-mode must be1..16')
    indices=[base+12*j+mode for base in (0,1152) for j in range(a.samples_per_mode) for mode in (0,4,8)]
    items=[dataset[i] for i in indices]
    embeddings,mask,_=lookup.batch([r['prompt'] for r in items],torch.device('cpu'))
    reference=torch.stack([r['reference'] for r in items])
    noise=torch.randn(reference.shape,generator=torch.Generator().manual_seed(1042))
    with torch.inference_mode():
        simulation=torch.cat([model(reference[i:i+6],embeddings[i:i+6].float(),mask[i:i+6],noise[i:i+6])
                              for i in range(0,len(reference),6)])
    torch.save(dict(reference=reference,target=torch.stack([r['target'] for r in items]),
        embeddings=embeddings,mask=mask,noise=noise,simulation=simulation,
        metadata=[{k:r[k] for k in ['sample_id','prompt','category','mode']} for r in items],
        indices=indices),a.output/'pilot_inputs.pt')
    shutil.copy2(a.checkpoint,a.output/'small.pt')
    contract=dict(schema=1,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        counted_parameters=audit['counted_parameters'],physical_pixel_pitch_um=8,simulation_pixel_pitch_um=17,
        distance_m=0.1,wavelength_nm=532,active_model_size=478,physical_active_size=1016,
        phase_orientation='flip_h_and_flip_v',phase_gray_inversion=True,camera_orientation='flip_v',
        preprocessing='frozen token embedding only; trainable language and both optics execute live',
        scope=f'fixed {len(indices)}-item sample set, not full test-set metrics',indices=indices,
        source_note='Final 20260926 small detail checkpoint, strict sealed loading; no retraining')
    (a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    print(json.dumps(contract),flush=True)


if __name__=='__main__':main()
