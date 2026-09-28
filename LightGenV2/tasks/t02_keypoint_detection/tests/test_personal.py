import copy,json
from pathlib import Path
import pytest
from PIL import Image
from LightGenV2.tasks.t02_keypoint_detection.personal_prepare import prepare,sha256
from LightGenV2.tasks.t02_keypoint_detection.personal_split import GROUPS,split_groups,crop_box
from LightGenV2.tasks.t02_keypoint_detection.personal_data import validate_manifest


def sample(tmp_path):
    im=tmp_path/'x.png';Image.new('RGB',(64,80)).save(im)
    rows=[]
    for i,split in enumerate(['train','test']):
        rows.append({'id':str(i),'image':'x.png','image_sha256':sha256(im),'original_sha256':str(i),
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
