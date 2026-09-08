import io
import json
from pathlib import PurePosixPath
import unittest

from TransferFromElectricity.tasks.t01_object_retrieval.collect_results import remote_state


class RemoteFiles:
    def __init__(self, **files):self.files=files
    def open(self,path,mode):
        name=PurePosixPath(path).stem
        if name not in self.files:raise FileNotFoundError(path)
        return io.BytesIO(json.dumps(self.files[name]).encode())


class CollectionTests(unittest.TestCase):
    def test_completion_waits_for_independent_gpu_audit(self):
        remote=RemoteFiles(status={'status':'complete'},history=[{},{}])
        self.assertEqual(remote_state(remote,PurePosixPath('/run'),True),('waiting_gpu_audit',2))
        remote.files['gpu_execution']={'status':'complete','returncode':0}
        self.assertEqual(remote_state(remote,PurePosixPath('/run'),True),('complete',2))

    def test_failed_run_and_failed_gpu_audit_stop_collection(self):
        for remote in (RemoteFiles(status={'status':'failed'}),RemoteFiles(status={'status':'complete'},gpu_execution={'status':'failed','returncode':1})):
            with self.assertRaises(RuntimeError):remote_state(remote,PurePosixPath('/run'),True)

    def test_run_not_launched_is_pending(self):
        self.assertEqual(remote_state(RemoteFiles(),PurePosixPath('/run'),True),('waiting',0))


if __name__=='__main__':unittest.main()
