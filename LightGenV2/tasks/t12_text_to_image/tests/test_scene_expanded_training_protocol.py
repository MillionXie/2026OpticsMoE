import copy
import json
from pathlib import Path
import tempfile
import unittest
from LightGenV2.tasks.t12_text_to_image.scene_paper_expanded_data import make_requests,PROFILE
from LightGenV2.tasks.t12_text_to_image.scene_expanded_paired_data import checked_dataset,PAIRED_SCHEMA
from LightGenV2.tasks.t12_text_to_image.scene_expanded_selection_render import checked_revision,public_report
from LightGenV2.tasks.t12_text_to_image.scene_expanded_middle import checked_profile,F6
from LightGenV2.tasks.t12_text_to_image.scene_instruction_data import sha
from LightGenV2.tasks.t12_text_to_image.scene_paper_prompt_protocol import write
from LightGenV2.tasks.t12_text_to_image.scene_test_PT_selection import snapshots
from LightGenV2.tasks.t12_text_to_image.scene_expanded_teacher_batch import balanced_order


class ExpandedTrainingProtocol(unittest.TestCase):
    def dataset(self,root):
        refs=[]
        for split,cat,n in [('train','lamp',36),('train','pillow',40),('test','lamp',13),('test','pillow',13)]:
            for i in range(n):
                rid=f'{split}-{cat}-{i}';p=root/(rid+'.png');p.write_bytes(rid.encode())
                refs.append(dict(split=split,category=cat,reference_id=rid,reference=p.name,reference_sha256=sha(p),
                    source_product_id=rid,source_family_id=rid,source_RGBA_sha256='b'*64))
        train,test=make_requests(refs);by_ref={r['reference_id']:r for r in refs};pairs=[]
        for row in train[:4]:
            name=row['sample_id']+'.png';(root/name).write_bytes(b'train-only-target');ref=by_ref[row['reference_id']]
            pairs.append(dict(row,split='train',target=name,target_sha256=sha(root/name),source_product_id=ref['source_product_id'],
                source_RGBA_sha256=ref['source_RGBA_sha256'],QA_accepted=True,training_admission_policy='basic_subject_quality_v2',supervision_weight=.5))
        write(root/'pairs.json',pairs);write(root/'references.json',refs);write(root/'inference_inputs.json',test)
        report=dict(schema=PAIRED_SCHEMA,complete=True,eligible_for_training=True,prompt_profile=PROFILE,validation_set=False,
            TEST_used_for_model_selection=True,role_disjoint=True,pairs_sha256=sha(root/'pairs.json'),request_sha256=sha(root/'inference_inputs.json'))
        write(root/'report.json',report);write(root/'visual_review.json',dict(approved_for_training=True,all_TRAIN_candidates_reviewed=True,pairs_sha256=report['pairs_sha256']))
        write(root/'selection_protocol.json',dict(test_request_sha256=report['request_sha256'],TEST_used_for_model_selection=True))
        return pairs,test,report

    def test_no_validation_or_TEST_target_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pairs,test,report=self.dataset(root)
            rows,inputs,_=checked_dataset(root);self.assertEqual(len(rows),4);self.assertEqual(len(inputs),500)
            self.assertTrue(all(r['split']=='train' for r in rows))
            pairs[0]['split']='dev';write(root/'pairs.json',pairs)
            report['pairs_sha256']=sha(root/'pairs.json');write(root/'report.json',report)
            write(root/'visual_review.json',dict(approved_for_training=True,all_TRAIN_candidates_reviewed=True,pairs_sha256=report['pairs_sha256']))
            with self.assertRaises(ValueError):checked_dataset(root)

    def test_raw_selection_checks_all500_and_no_GT_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pairs,test,report=self.dataset(root)
            state=dict(pair_sha256=report['pairs_sha256'],no_validation_test_PT_selection=True)
            for row in pairs:(root/row['target']).unlink()
            # Raw inference only checks public inputs; TRAIN target pixels are absent.
            rows,protocol=checked_revision(root,root,state,[])
            self.assertEqual(len(rows),500);self.assertTrue(protocol['TEST_used_for_model_selection'])
            test.pop();write(root/'inference_inputs.json',test)
            with self.assertRaises(ValueError):checked_revision(root,root,state,[])

    def test_explicit_middle_graph_and_actual_coverage(self):
        profile=dict(schema='scene_expanded_middle_v1',prompt_profile=PROFILE,vision_keep=6,language_keep=4,UNet_deep_keep=9,student_steps=8,
            parent_generator_sha256='a'*64,parent_frontend_sha256=F6,electronic_teacher_sha256='c'*64,pairs_sha256='d'*64,TRAIN_contexts=1200)
        self.assertEqual(checked_profile(profile),profile)
        for key,value in [('TRAIN_contexts',1521),('UNet_deep_keep',8),('student_steps',6)]:
            c=dict(profile);c[key]=value
            with self.assertRaises(ValueError):checked_profile(c)

    def test_preserve_distinct_best_last_without_epoch_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);train=root/'fit';train.mkdir()
            (train/'best_checkpoint.pt').write_bytes(b'actual-best');(train/'last_checkpoint.pt').write_bytes(b'actual-last')
            write(train/'report.json',dict(complete=True,best_sha256=sha(train/'best_checkpoint.pt'),final_TEST_PT_selection_pending=True))
            bank=snapshots(train,root/'bank');self.assertEqual(len(bank),2)
            self.assertEqual(len({r['checkpoint_sha256'] for r in bank}),2)
            self.assertEqual(sha(train/'best_checkpoint.pt'),bank[0]['checkpoint_sha256'])

    def test_batched_schedule_keeps_all_requests_and_both_categories(self):
        rows=[dict(sample_id=f'{c}-{i}-{j}',category=c,reference_id=f'{c}-{i}',condition_id=str(j))
              for c in ['lamp','pillow'] for i in range(5) for j in range(3)]
        scheduled=balanced_order(rows)
        self.assertEqual({r['sample_id']for r in scheduled},{r['sample_id']for r in rows})
        self.assertEqual(len(scheduled),len(rows));self.assertEqual({r['category']for r in scheduled[:2]},{'lamp','pillow'})


if __name__=='__main__':unittest.main()
