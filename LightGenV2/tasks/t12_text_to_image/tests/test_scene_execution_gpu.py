import unittest
from ..scene_execution_gpu import configured_gpu,LEGACY_GPU


class ExecutionGPU(unittest.TestCase):
    def test_legacy_default_is_unchanged_without_override(self):
        self.assertEqual(configured_gpu({}),LEGACY_GPU)

    def test_explicit4090_UUID_is_returned_without_changing_other_environment(self):
        env={'T12_GPU_UUID':'GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd','CUDA_VISIBLE_DEVICES':''}
        before=dict(env);self.assertEqual(configured_gpu(env),env['T12_GPU_UUID']);self.assertEqual(env,before)

    def test_indices_names_empty_and_malformed_UUID_are_rejected(self):
        for value in ['3','RTX4090','','GPU-1234',None]:
            with self.assertRaises(ValueError):configured_gpu({'T12_GPU_UUID':value})


if __name__=='__main__':unittest.main()
