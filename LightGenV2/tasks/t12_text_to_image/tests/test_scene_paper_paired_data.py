import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
from LightGenV2.tasks.t12_text_to_image import scene_paper_paired_data as data
from LightGenV2.tasks.t12_text_to_image.scene_paper_prompt_protocol import write,condition,PROFILE
from LightGenV2.tasks.t12_text_to_image.scene_generation_data import request_seed
from LightGenV2.tasks.t12_text_to_image.scene_paper_paired_data import accepted_ids


class NewTargets(unittest.TestCase):
    def test_explicit_basic_policy_retains_hard_examples_without_full_success(self):
        rows=[dict(sample_id=str(i),target_sha256='a'*64)for i in range(4)]
        review=dict(schema='scene_paper_new_target_review_v1',all_candidates_reviewed=True,admission_policy='basic_subject_quality_v2',candidate_manifest_sha256='b'*64,rows=[dict(sample_id=str(i),target_sha256='a'*64,facts='YNN'if i==0 else'NYY'if i==1 else'YYY',coarse_form='N',quality='N'if i==2 else'Y',gross_subject_failure=i==3,accepted=i==0)for i in range(4)])
        self.assertEqual(accepted_ids(rows,review,'b'*64),{'0'})
        review.pop('admission_policy')
        with self.assertRaises(ValueError):accepted_ids(rows,review,'b'*64)
    def test_uncertain_facts_and_shape_rejected(self):
        rows=[dict(sample_id=str(i),target_sha256='a'*64)for i in range(4)]
        review=dict(schema='scene_paper_new_target_review_v1',all_candidates_reviewed=True,candidate_manifest_sha256='b'*64,rows=[dict(sample_id=str(i),target_sha256='a'*64,facts=['YYY','YYU','YYY','YYY'][i],coarse_form='N'if i==2 else'Y',quality='N'if i==3 else'Y',accepted=i==0)for i in range(4)])
        self.assertEqual(accepted_ids(rows,review,'b'*64),{'0'})
        review['rows'][1]['accepted']=True
        with self.assertRaises(ValueError):accepted_ids(rows,review,'b'*64)

    def test_missing_review_or_changed_pixels(self):
        row=dict(sample_id='a',target_sha256='a'*64);review=dict(schema='scene_paper_new_target_review_v1',all_candidates_reviewed=True,candidate_manifest_sha256='b'*64,rows=[])
        with self.assertRaises(ValueError):accepted_ids([row],review,'b'*64)
        review['rows']=[dict(sample_id='a',target_sha256='c'*64,facts='YYY',coarse_form='Y',quality='Y',accepted=True)]
        with self.assertRaises(ValueError):accepted_ids([row],review,'b'*64)

    def test_new_targets_and_full_dev_survive_parent_transfer(self):
        # Synthetic byte fixtures exercise curation/identity only, not image quality.
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);plan=root/'plan';parent=root/'parent';train=root/'train';dev=root/'dev';out=root/'out'
            for p in [plan,parent,train,dev]:p.mkdir();(p/'references').mkdir();(p/'targets').mkdir()
            catalog=[];parent_rows=[];tr=[];dv=[]
            for i in range(92):
                rid='r'+str(i);split='dev'if 44<=i<60 else'train';relative='references/'+rid+'.png';(plan/relative).write_bytes(('synthetic source '+rid).encode());digest=data.sha(plan/relative)
                source=dict(sample_id=rid,split=split,product_id='product'+str(i),rgba_file_sha256=format(i,'064x'),white_reference_sha256=digest);catalog.append(source)
                if i>=60:continue
                parent_rows.append(dict(reference_id=rid,split=split,source_product_id=source['product_id'],source_RGBA_sha256=source['rgba_file_sha256'],reference_sha256=digest))
                for c in range(3):
                    d=condition('lamp'if i<20 else'pillow',c)
                    for draw in ([2]if split=='dev'else[0,1]):
                        row=dict(sample_id=rid+'-paper-scene'+str(c)+'-draw'+str(draw),reference=relative,reference_id=rid,reference_sha256=digest,seed=request_seed(digest,draw),**{k:d[k]for k in ['category','condition_id','prompt','native_caption','reward_caption','prompt_profile']})
                        (dv if split=='dev'else tr).append(row)
            write(parent/'pairs.json',parent_rows);write(parent/'report.json',dict(role_disjoint=True));parent_sha=data.sha(parent/'pairs.json');sources=root/'sources.json';write(sources,catalog)
            write(plan/'train_inputs.json',tr);write(plan/'dev_inputs.json',dv);write(plan/'report.json',dict(complete=True,prompt_profile=PROFILE,source_pairs_sha256=parent_sha,TEST_opened=False,TRAIN_request_sha256=data.sha(plan/'train_inputs.json'),request_sha256=data.sha(plan/'dev_inputs.json')))
            for p,rows,filename in [(train,tr,'candidates.json'),(dev,dv,'images.json')]:
                records=[]
                for row in rows:
                    target='targets/'+row['sample_id']+'.png';(p/target).write_bytes(('synthetic target '+row['sample_id']).encode());digest=data.sha(p/target)
                    records.append(dict(row,target=target,target_sha256=digest,generated_PNG=target,generated_sha256=digest,teacher_revision=data.REVISION,GT_masks_or_targets_as_inputs=False))
                write(p/filename,records)
                review=dict(schema='scene_paper_new_target_review_v1',all_candidates_reviewed=True,candidate_manifest_sha256=data.sha(p/filename),rows=[dict(sample_id=r['sample_id'],target_sha256=r['target_sha256'],facts='YYY'if j==0 else'YNY',coarse_form='Y',quality='Y',accepted=j==0)for j,r in enumerate(records)])
                write(p/'review.json',review)
            write(train/'report.json',dict(complete=True,schema='scene_paper_TRAIN_candidates_v1',DEV_or_TEST_generated=False,TRAIN_request_sha256=data.sha(plan/'train_inputs.json'),candidate_manifest_sha256=data.sha(train/'candidates.json')))
            write(dev/'report.json',dict(complete=True,prompt_profile=PROFILE,independent_TEST_opened=False,generated_count=48,official_weights_revision=data.REVISION))
            with patch.object(data,'PARENT_PAIRS',parent_sha),patch.object(data,'SOURCE_CATALOG_SHA',data.sha(sources)):
                data.curate(plan,parent,sources,train,train/'review.json',dev,dev/'review.json',out)
                pairs=__import__('json').loads((out/'pairs.json').read_text());self.assertEqual(len(pairs),2)
                self.assertEqual(len(__import__('json').loads((out/'inference_inputs.json').read_text())),48)
                self.assertTrue(data.checked_parent_transfer(out,parent,parent_sha,data.sha(out/'pairs.json'))['new_targets_and_new_prompts_not_recaptioned_old_pixels'])
                from LightGenV2.tasks.t12_text_to_image.scene_paper_paired_render import checked_revision
                current,protocol=checked_revision(out,out,dict(pair_sha256=data.sha(out/'pairs.json')),dv)
                self.assertEqual(len(current),48);self.assertTrue(protocol['model_trained_on_revised_dataset'])
                with self.assertRaises(ValueError):checked_revision(out,out,dict(pair_sha256=parent_sha),dv)
                (out/pairs[0]['target']).write_bytes(b'changed')
                with self.assertRaises(ValueError):data.checked_parent_transfer(out,parent,parent_sha,data.sha(out/'pairs.json'))


if __name__=='__main__':unittest.main()
