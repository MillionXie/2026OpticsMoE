import sys
import unittest
from audit_lgvq_a100_layout_20261027 import validate_layout


class AuditLayoutTests(unittest.TestCase):
    def test_original_dtype_geometry_and_scope_required(self):
        info=dict(hardware_scope='ORIGINAL_A100_250W_108SM',actual_original_kernel=[2,16,16],
            actual_original_stride=[2,16,16],weight_shape=[1024,3,2,16,16],weight_dtype='torch.bfloat16',
            bias_dtype='torch.bfloat16',class_source_sha256='b80abab0ebe915256e6c0630c70472670e4e6aa99d6a8d1f4c0cfc62cd7977e0',
            original_convolution_and_bias_retained=True,original_A100_reference_retained=True,
            all_original_math_and_timing_gates_retained=True,input_layout_conversion_inside_actual_forward=True,
            weight_values_or_precision_changed=False,GEMM_or_geometry_replacement=False,extra_forward_calls=0)
        validate_layout(info)
        for key,value in [('actual_original_kernel',[2,14,14]),('GEMM_or_geometry_replacement',True),
                ('weight_dtype','torch.float16'),('extra_forward_calls',1)]:
            with self.assertRaises(ValueError):validate_layout(dict(info,**{key:value}))

    def test_no_model_import(self):self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
