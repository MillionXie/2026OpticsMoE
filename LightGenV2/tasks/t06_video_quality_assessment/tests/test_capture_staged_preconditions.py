"""Exercise capture preconditions only, without SDKs, model assets or image capture."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from LightGenV2.tasks.t06_video_quality_assessment import lab_bench


class CapturePreconditionTests(unittest.TestCase):
    stages = ('vision_router', 'vision_expert', 'vision_global')

    def invoke(self, measured, phase_ready=False):
        a = SimpleNamespace(stage='vision_router', phase_ready=phase_ready)
        state = {'measured_stages': measured, 'hardware_sha256': 'hw', 'release_sha256': 'release'}
        manifest = {'hardware_sha256': 'hw', 'release_sha256': 'release', 'phase_file': 'phase.bmp', 'phase_sha256': 'phase'}
        opener = lambda args: (Path('unused'), Path('unused'), state, {}, {})
        with patch.dict(sys.modules, {'cv2': SimpleNamespace()}), \
             patch.object(lab_bench, 'read', return_value=manifest), \
             patch.object(lab_bench, 'sha', return_value='phase'), \
             patch.object(lab_bench, 'write', side_effect=AssertionError('Must not write')), \
             patch.object(lab_bench.os, 'open', side_effect=AssertionError('Must not open lock')):
            return lab_bench.capture_staged(a, opener, self.stages)

    def test_t03_capture_import_is_the_published_shared_function(self):
        from LightGenV2.tasks.t03_saliency.lab_bench import capture_staged
        self.assertIs(capture_staged, lab_bench.capture_staged)

    def test_completed_stage_does_not_capture(self):
        self.assertIsNone(self.invoke(['vision_router']))

    def test_phase_confirmation_required_before_device_or_writes(self):
        with self.assertRaisesRegex(ValueError, 'explicitly pass --phase-ready'):
            self.invoke([])

    def test_noncontiguous_state_rejected_before_device_or_writes(self):
        with self.assertRaisesRegex(ValueError, 'Non-contiguous capture'):
            self.invoke(['vision_expert'])


if __name__ == '__main__':
    unittest.main()
