import copy,json
from pathlib import Path
import pytest
from PIL import Image
from LightGenV2.tasks.t02_keypoint_detection.personal_prepare import prepare,sha256
from LightGenV2.tasks.t02_keypoint_detection.personal_split import GROUPS,split_groups,crop_box
from LightGenV2.tasks.t02_keypoint_detection.personal_data import validate_manifest
from LightGenV2.tasks.t02_keypoint_detection.personal_data import fewshot_split

def test_requested_exclusions_exact():
    from LightGenV2.tasks.t02_keypoint_detection.personal_curate import EXCLUDED
    assert len(EXCLUDED)==len(set(EXCLUDED))==29
    assert 'photo_003_p00' not in EXCLUDED and 'photo_089_p00' not in EXCLUDED
    assert 'photo_040_p01' in EXCLUDED and 'photo_041_p01' in EXCLUDED

def test_frozen_digest_includes_buffers_and_scalar():
    torch=pytest.importorskip('torch')
    from LightGenV2.tasks.t02_keypoint_detection.personal_finetune import state_digest,force_frozen_eval
    model=torch.nn.Sequential(torch.nn.BatchNorm1d(2),torch.nn.Linear(2,1));model.requires_grad_(False)
    before=state_digest(model);model.register_forward_pre_hook(force_frozen_eval)
    model.train();model(torch.ones(3,2));assert state_digest(model)==before
    with torch.no_grad():model[1].bias.add_(1)
    assert state_digest(model)!=before


def test_fewshot_whole_groups():
    data={'images':[{'id':str(i),'group_id':str(g),'split':'test'} for g,items in enumerate(GROUPS) for i in items]}
    result=fewshot_split(data,10)
    assert result==fewshot_split(data,10)
    assert sum(r['split']=='train' for r in result['images'])==10
    assert all(r['split']=='test' for r in data['images'])
    for g in range(len(GROUPS)):
        assert len({r['split'] for r in result['images'] if r['group_id']==str(g)})==1
    with pytest.raises(ValueError):fewshot_split(data,101)


def sample(tmp_path):
    rows=[]
    for i,split in enumerate(['train','test']):
        im=tmp_path/f'x{i}.png';Image.new('RGB',(64,80),(i,0,0)).save(im)
        rows.append({'id':str(i),'image':im.name,'image_sha256':sha256(im),'original_sha256':str(i),
                     'orientation_reviewed':True,'split':split,'group_id':str(i),'size_wh':[64,80],
                     'people':[{'id':'0','include':True,'reviewed':False,'manual_required':[12,13],
                                'keypoints':[[20,30,1]]*14}]})
    f=tmp_path/'a.json';f.write_text(json.dumps({'images':rows}),encoding='utf-8');return f,rows


def test_manual_review_gate(tmp_path):
    f,rows=sample(tmp_path)
    with pytest.raises(ValueError,match='Human review'):validate_manifest(f)
    assert validate_manifest(f,True)[1]=={'train':1,'test':1}
    rows[1]['group_id']='0';f.write_text(json.dumps({'images':rows}))
    with pytest.raises(ValueError,match='leakage'):validate_manifest(f,True)


def test_group_split():
    assert sorted(sum(GROUPS,[]))==list(range(101))
    held=split_groups(GROUPS);assert held==split_groups(GROUPS)
    assert sum(len(GROUPS[g]) for g in held)==20
    assert 34 in GROUPS[10] and 35 in GROUPS[10]


def test_ingest_preserves_original_and_rotates(tmp_path):
    src=tmp_path/'raw';src.mkdir();p=src/'a.jpg';Image.new('RGB',(80,40)).save(p)
    before=sha256(p);prepare(src,tmp_path/'out',{'a.jpg':90})
    assert sha256(p)==before
    d=json.loads((tmp_path/'out/annotations.json').read_text())
    assert d['images'][0]['size_wh']==[40,80]
    with pytest.raises(FileExistsError):prepare(src,tmp_path/'out')


def test_crop_square_and_margin():
    box=crop_box([[10,10,1],[50,90,1],[9000,9000,0]])
    assert box[2]-box[0]==box[3]-box[1]==100
