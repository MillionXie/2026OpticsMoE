import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from LightGenV2.tasks.t07_abo_image_retrieval.hardware import inspect_binding as gate


class BindingTests(unittest.TestCase):
    def row(self):
        return {'schema_version':1,'task':'t07_rank72_sealed','checkpoint_sha256':gate.CHECKPOINT,
                **{name:'fixture/'+name for name in gate.PATH_FIELDS}}

    def write(self,folder,row):
        path=Path(folder)/'binding.json';path.write_text(json.dumps(row),encoding='utf8');return path

    def test_requires_sealed_identity_and_inspection_only(self):
        with tempfile.TemporaryDirectory() as folder:
            for change in ({'mode':'full'},{'checkpoint_sha256':'0'*64},{'task':'other'},{'processor':''}):
                with self.assertRaises(ValueError):gate.load_binding(self.write(folder,{**self.row(),**change}))

    def test_bad_original_asset_sha_fails_before_any_runtime_load(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(gate,'digest',return_value='0'*64),patch.object(gate,'inspect_paths') as inspect:
                with self.assertRaises(ValueError):gate.inspect_binding(self.write(folder,self.row()))
                inspect.assert_not_called()

    def test_default_inspection_has_no_model_or_device_operation(self):
        def digest(path):
            return {'protocol':gate.PROTOCOL_SHA,'geometry':gate.GEOMETRY_SHA}.get(path.name,'config_sha')
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(gate,'digest',side_effect=digest),patch.object(gate,'inspect_paths',return_value={'sample_count':2400}),patch.object(gate,'inspect_machine',return_value={'passed':True}):
                result=gate.inspect_binding(self.write(folder,self.row()))
                self.assertTrue(result['passed'])
                for name in ('model_loaded','devices_opened','dataset_evaluated','outputs_created'):
                    self.assertFalse(result[name])

    def test_concurrent_config_change_is_not_accepted(self):
        calls=iter([gate.PROTOCOL_SHA,gate.GEOMETRY_SHA,'before','after'])
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(gate,'digest',side_effect=lambda _:next(calls)),patch.object(gate,'inspect_paths',return_value={}),patch.object(gate,'inspect_machine',return_value={'passed':True}):
                with self.assertRaises(RuntimeError):gate.inspect_binding(self.write(folder,self.row()))


if __name__=='__main__':unittest.main()
