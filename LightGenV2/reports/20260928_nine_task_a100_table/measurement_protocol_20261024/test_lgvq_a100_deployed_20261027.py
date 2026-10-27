import sys
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
import run_lgvq_a100_deployed_20261027 as entry
from lgvq_phase_ledger_20261026 import electronic_graph_build_nodes


class DeployedTests(unittest.TestCase):
    def test_original_worker_owns_new_leaf_creation(self):
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/'leaf';args=SimpleNamespace(output=output)
            def original(arg):
                self.assertFalse(arg.output.exists());arg.output.mkdir();return 'complete'
            with patch.object(entry.core,'parallel_worker',original):
                self.assertEqual(entry.deployed_worker(args),'complete')
            self.assertTrue((output/'a100_deployed_scope_identity.json').exists())

    def test_original_42_blobs_and_worker_unchanged(self):
        original=entry.core.ORIGINAL_SOURCES();worker=entry.core.original.worker
        with entry.configured():
            sources=entry.source_map()
            self.assertEqual({k:sources[k] for k in original},original)
            self.assertEqual(len(sources),45)
        self.assertIs(entry.core.original.worker,worker)
        self.assertNotIn('torch',sys.modules)

    def test_excludes_exact_electronic_capture_only(self):
        raw=dict(explicit_warmup_forwards=0,records=[dict(field=f,graph_construction=[dict(kind='electronics')]*22+
            [dict(kind='optical_simulation')]*6) for f in (3,4)])
        nodes=[dict(id=f'f{f}.capture{i}',field=f,kind='capture_electronics') for f in (3,4) for i in range(22)]
        nodes.append(dict(id='f5.g0.queue',field=5,kind='dispatch'))
        self.assertEqual(len(electronic_graph_build_nodes(raw,nodes)),44)
        self.assertNotIn('f5.g0.queue',electronic_graph_build_nodes(raw,nodes))
        with self.assertRaises(ValueError):electronic_graph_build_nodes(dict(raw,explicit_warmup_forwards=1),nodes)
        with self.assertRaises(ValueError):electronic_graph_build_nodes(raw,nodes[:-2])


if __name__=='__main__':unittest.main()
