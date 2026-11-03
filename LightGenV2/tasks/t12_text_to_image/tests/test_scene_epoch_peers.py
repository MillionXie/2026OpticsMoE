import json
from pathlib import Path
import tempfile
import unittest
from ..scene_native_edit_peers import preview_first,verify_assets,prepare_reference
from ..scene_resnet_frontend import experiment_profile,checked_evaluation_epoch
from ..scene_resnet_supervised_train import REQUESTED_EPOCHS,requested_training_epochs,partial_training_updates,partial_last_update_scale
from ..scene_resnet_frontend_metrics import fixed_last_records


class EpochPeerTests(unittest.TestCase):
    def test_only_explicit_partial_training_can_reduce_final_step_lr(self):
        self.assertEqual(partial_last_update_scale(.5,9),.5)
        self.assertEqual(partial_last_update_scale(1.,None),1.)
        for scale,partial in [(.5,None),(0.,9),(1.1,9),(float('nan'),9),(float('inf'),9)]:
            with self.assertRaises(ValueError):partial_last_update_scale(scale,partial)

    def test_partial_budget_does_not_impersonate_epoch_or_weaken_old_guard(self):
        p=experiment_profile('qwen70_30_supervised')
        for n in [1,2,77]:
            self.assertEqual(partial_training_updates(n),n)
            saved=dict(epoch=0,step=n,explicit_partial_updates=n)
            checked_evaluation_epoch(p,saved,'last',0,n)
            with self.assertRaises(ValueError):checked_evaluation_epoch(p,saved,'last',0)
            with self.assertRaises(ValueError):checked_evaluation_epoch(p,saved,'last',0,n+1)
        for n in [0,78,-1,1.5]:
            with self.assertRaises(ValueError):partial_training_updates(n)
        with self.assertRaises(ValueError):partial_training_updates(2,True)

    def test_native512_interface_keeps_only_RGB256_information(self):
        from PIL import Image
        original=Image.new('RGB',(1024,1024),'red')
        small,ip=prepare_reference(original,'ip2p');same,flux=prepare_reference(original,'flux2_klein4b')
        self.assertEqual(small.size,(256,256));self.assertEqual(ip.size,(512,512))
        self.assertEqual(flux.size,(256,256));self.assertEqual(small.tobytes(),same.tobytes())
        self.assertEqual(ip.tobytes(),small.resize((512,512),Image.Resampling.BICUBIC).tobytes())

    def test_zero_and_short_epoch_authorization_does_not_weaken_fixed50(self):
        p=experiment_profile('qwen70_30_supervised')
        self.assertEqual(REQUESTED_EPOCHS,(0,1,2,5,10));self.assertEqual(requested_training_epochs(),50)
        self.assertEqual(requested_training_epochs(True),10)
        for epoch in REQUESTED_EPOCHS:checked_evaluation_epoch(p,dict(epoch=epoch,requested_epoch_sweep=True),'last',epoch)
        checked_evaluation_epoch(p,dict(epoch=50),'last')
        for student,candidate,epoch in [(dict(epoch=0),'last',0),(dict(epoch=1,requested_epoch_sweep=True),'last',2),(dict(epoch=50),'best',50)]:
            with self.assertRaises(ValueError):checked_evaluation_epoch(p,student,candidate,epoch)
        with self.assertRaises(ValueError):checked_evaluation_epoch(experiment_profile('qwen70_30'),dict(epoch=0,requested_epoch_sweep=True),'last',0)

    def test_preview_is_balanced_without_changing_requests(self):
        rows=[dict(sample_id=f'{cat}{ref}{condition}',category=cat,reference_id=f'{cat}{ref}',condition_id=str(condition))
              for cat in ['pillow','lamp']for ref in range(3)for condition in range(4)]
        ordered=preview_first(rows)
        self.assertEqual({r['sample_id']for r in ordered},{r['sample_id']for r in rows})
        self.assertEqual([r['category']for r in ordered[:8]].count('lamp'),4)
        self.assertEqual(len({r['reference_id']for r in ordered[:8]}),4)

    def test_bad_model_asset_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name);(root/'weight').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_assets(root,dict(complete=True,repo='timbrooks/instruct-pix2pix',revision='a'*40,
                all_files_bound_to_official_revision=True,files={'weight':dict(bytes=7,sha256='b'*64)}),'ip2p')

    def test_epoch_metric_reader_rejects_wrong_epoch_before_any_image_access(self):
        from ..scene_resnet_frontend import REQUESTS,ELECTRONIC
        with tempfile.TemporaryDirectory()as name:
            root=Path(name);(root/'report.json').write_text(json.dumps(dict(complete=True,scope='TEST_fixed_requested_epoch',
                fixed_generator_sha256=ELECTRONIC,TEST_used_for_model_selection=False,TEST_selection_protocol=dict(request_sha256=REQUESTS),evaluation_epoch=1)))
            with self.assertRaises(ValueError):list(fixed_last_records(root,root,experiment_profile('qwen70_30_supervised'),2))


if __name__=='__main__':unittest.main()
