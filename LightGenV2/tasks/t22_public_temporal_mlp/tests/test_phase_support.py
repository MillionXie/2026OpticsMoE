from types import SimpleNamespace
import unittest

import torch

from ..tools.audit_unilluminated_phase import constant_unilluminated_modulation
from ..tools.phase_regularization import illumination_support


class PhaseSupportTests(unittest.TestCase):
    def test_constant_outside_support_preserves_the_actual_incident_product(self):
        for frame_level in (True, False):
            with self.subTest(frame_level=frame_level):
                raw = torch.randn(16, 111, 111)
                values = torch.exp(1j * raw)
                stage = SimpleNamespace(frame_level=frame_level, raw_global_phase=raw,
                                        modulation=lambda: values)
                support = illumination_support(frame_level)
                changed = constant_unilluminated_modulation(stage)()
                self.assertTrue(torch.equal(changed[:, support], values[:, support]))
                self.assertTrue(torch.equal(changed[:, ~support], torch.ones_like(changed[:, ~support])))
                incident = torch.rand_like(raw) * support
                self.assertTrue(torch.equal(incident * changed, incident * values))


if __name__ == '__main__':
    unittest.main()
