import json,tempfile,unittest
from pathlib import Path
from ..scene_instruction_data import condition,sha,write
from ..scene_instruction_native_prior import visible_caption
from ..scene_omnigen_peer import matched_visible_inputs


class VisiblePeerInputs(unittest.TestCase):
    def fixture(self,p):
        rows=[]
        for i in range(16):
            category='lamp'if i<8 else'pillow';rid=f'{category}-{i}';f=p/(rid+'.png');f.write_bytes(str(i).encode())
            for code in('B','D','W')if category=='lamp'else('S','A','D'):
                c=condition(category,code)
                rows.append(dict(sample_id=rid+'-visible-'+code+'-draw2',reference=f.name,prompt=c['prompt'],seed=42+i,reference_id=rid,reference_sha256=sha(f),category=category,condition_id=c['condition_id'],native_caption=visible_caption(category,code),reward_caption=c['reward_caption'],prompt_profile=c['prompt_profile']))
        rows.sort(key=lambda r:(r['reference_id'],r['condition_id']))
        write(p/'inference_inputs.json',rows);write(p/'inference_dev_inputs.json',rows)
        write(p/'report.json',dict(complete=True,schema='scene_balanced_visible_anchor_dataset_v1',prompt_profile=c['prompt_profile'],TEST_opened=False,role_disjoint=True,eligible_for_training=True,all_reference_conditions_covered=True,TRAIN_products=44,TRAIN_reference_views=44,TRAIN_condition_groups=132,pairs_sha256='a'*64,source_pairs_sha256='b'*64,native_caption_style='public_fact_equivalent_description_v1',request_sha256=sha(p/'inference_inputs.json')))
        return rows

    def test_exact_public_inputs_without_target_files(self):
        with tempfile.TemporaryDirectory()as folder:
            p=Path(folder);expected=self.fixture(p);rows,protocol=matched_visible_inputs(p)
            self.assertEqual(rows,expected);self.assertEqual(len(rows),48)
            self.assertTrue(protocol['peer_has_no_task_tuning']);self.assertFalse((p/'pairs.json').exists())

    def test_changed_prompt_seed_fields_or_cohort_rejected(self):
        for mode in ['prompt','seed','target','drop','pixels']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as folder:
                p=Path(folder);rows=self.fixture(p)
                if mode=='drop':rows.pop()
                elif mode=='pixels':(p/rows[0]['reference']).write_bytes(b'changed')
                elif mode=='target':rows[0]['target']='forbidden.png'
                elif mode=='seed':rows[0]['seed']+=1
                else:rows[0]['prompt']='Generate anything.'
                write(p/'inference_inputs.json',rows);r=json.loads((p/'report.json').read_text());r['request_sha256']=sha(p/'inference_inputs.json');write(p/'report.json',r)
                with self.assertRaises(ValueError):matched_visible_inputs(p)


if __name__=='__main__':unittest.main()
