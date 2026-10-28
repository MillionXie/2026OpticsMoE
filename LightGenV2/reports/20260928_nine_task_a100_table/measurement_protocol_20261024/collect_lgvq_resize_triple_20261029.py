"""New immutable resize/normalization-inclusive pooled LG result."""
import json
from pathlib import Path
import sys
from collect_lgvq_compute_triple_20261029 import collect

def main(lg,qwen,destination):
    for repeat in (1,2,3):
        report=json.loads((Path(lg)/f'lg_repeat{repeat:02d}'/'compute_report.json').read_text())
        if not report.get('CPU_resize_and_normalization_included'):raise ValueError('Resize/normalization-inclusive report required')
    return collect(lg,qwen,destination,extra_metadata=dict(
        measurement_scope='native_resize_and_normalization_inclusive_LG_20261029',
        LG_CPU_input_scope=['native_65percent_crop_to_448_OpenCV_resize','official_processor_resize',
            'native_input_tensor_formatting_and_rescale_normalization','exact_patch_with_original_BF16_cast','tokenizer'],
        LG_CPU_excluded=['video_file_reading_and_decoding','RGB_conversion_and_65percent_crop','quality_pixel_input_preparation',
                         'dispatch_queue_synchronization_waiting'],
        warmup_loading_scope_unchanged=True,old_resize_excluded_result_not_overwritten=True))

if __name__=='__main__':
    result=main(*sys.argv[1:]);print(json.dumps({k:v for k,v in result.items() if k not in ('rounds','equivalent_s_per_group')},ensure_ascii=True))
