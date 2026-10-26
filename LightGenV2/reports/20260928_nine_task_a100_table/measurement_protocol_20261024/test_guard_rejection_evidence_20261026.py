import unittest
from types import SimpleNamespace
from lab_corrected_energy_triples_20261026 import diagnostic_guard_check


def original_check(self):
    clients=set(self.clients)
    if clients-self.self_ids:
        raise RuntimeError('Foreign MPS client')
    self.audit.append(dict(clients=sorted(clients)))


class EvidenceTests(unittest.TestCase):
    def test_rejection_condition_unchanged_and_snapshot_retained(self):
        check=diagnostic_guard_check(original_check)
        owner=SimpleNamespace(clients=[2],self_ids={1},audit=[])
        with self.assertRaisesRegex(RuntimeError,'Foreign MPS client'):check(owner)
        self.assertEqual(owner.audit,[dict(clients=[2])])

    def test_qualified_snapshot_once(self):
        check=diagnostic_guard_check(original_check)
        owner=SimpleNamespace(clients=[1],self_ids={1},audit=[])
        check(owner)
        self.assertEqual(owner.audit,[dict(clients=[1])])

    def test_original_also_rejects_identical_client(self):
        owner=SimpleNamespace(clients=[2],self_ids={1},audit=[])
        with self.assertRaisesRegex(RuntimeError,'Foreign MPS client'):original_check(owner)
        self.assertEqual(owner.audit,[])


if __name__=='__main__':unittest.main()
