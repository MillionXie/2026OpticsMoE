from pathlib import Path
import json
import sys
from collections import Counter
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from abo_image_student_saved_shape_mac import call_mac,validate_counts,META,CONFIG


def test_one_main_merger_and_actual_length_not_optical_capacity():
    meta=json.loads(META.read_text());c=json.loads(CONFIG.read_text());a=call_mac(meta,c,76);b=call_mac(meta,c,77)
    assert a['selected_components']['main_native_merger_only']==49*(4096**2+4096*2048)
    assert a['selected_components']['vision.output_adapter']==196*192*1024
    assert a['selected_components']['language.output_adapter']==2*76*192*2048
    assert b['actual_MAC']>a['actual_MAC']
    assert a['actual_MAC']==a['selected_serial_MAC']+a['parallel_residual_MAC']+192*76
    assert a['selected_components']['CPU_100title_search']==6400


def test_auxiliary_mergers_are_not_silently_counted():
    counts=Counter({'frontend.patch_embedding':1,'bridge.native_merger':1,'head.retrieval_readout':1,'output.cpu_original_cosine_nonstable_rank':1})
    for mode in ('vision','language'):
        counts.update({mode+'.'+k:v for k,v in dict(input_adapter=1,output_adapter=1 if mode=='vision' else 2,
            amplitude_linear_norm_softplus=2,expert_ccd_output_linear=1,global_ccd_output_linear=1,
            parallel_residual1=1,parallel_residual2=1,router_simulation=1,expert_global_simulation=2).items()})
    validate_counts(counts);counts['bridge.deepstack_merger0']=1
    with pytest.raises(ValueError):validate_counts(counts)
