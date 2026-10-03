import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_reviewed_replacements import prepare
from test_main_preparation import git, synthetic_repo


def fixture(tmp_path):
    synthetic_repo(tmp_path)
    path = 'LightGenV2/tasks/t03_saliency/modeling.py'
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_bytes(b'value = 1\n')
    git(tmp_path,'add',path)
    git(tmp_path,'commit','-m','old')
    main = git(tmp_path,'rev-parse','HEAD').decode().strip()
    git(tmp_path,'update-ref','refs/heads/main',main)
    target.write_bytes(b'value = 2\n')
    git(tmp_path,'add',path)
    git(tmp_path,'commit','-m','validated source')
    manifest = {'expected_main':main,'source_commit':git(tmp_path,'rev-parse','HEAD').decode().strip(),
                'task_prefix':'LightGenV2/tasks/t03_saliency/', 'paths':[
                    {'path':path,'source_sha256':hashlib.sha256(b'value = 2\n').hexdigest(),
                     'expected_target_sha256':hashlib.sha256(b'value = 1\n').hexdigest(),
                     'review_reason':'Validated task-only runtime change'}]}
    return manifest, target


def test_candidate_preserves_dirty_user_files_and_index(tmp_path):
    manifest,target = fixture(tmp_path)
    target.write_bytes(b'user_uncommitted = True\n')
    (tmp_path/'user.py').write_text('user = True\n')
    git(tmp_path,'add','user.py')
    before = git(tmp_path,'status','--porcelain')
    result = prepare(tmp_path,manifest,'reviewed candidate')
    assert git(tmp_path,'status','--porcelain') == before
    assert git(tmp_path,'show',result['candidate_commit']+':'+manifest['paths'][0]['path']) == b'value = 2\n'
    assert git(tmp_path,'rev-parse','main').decode().strip() == manifest['expected_main']


@pytest.mark.parametrize('field',['source_sha256','expected_target_sha256'])
def test_rejects_unreviewed_identity(tmp_path,field):
    manifest,_ = fixture(tmp_path)
    manifest['paths'][0][field] = '0'*64
    with pytest.raises(RuntimeError,match='SHA mismatch'):
        prepare(tmp_path,manifest,'refused')


def test_rejects_shared_path(tmp_path):
    manifest,_ = fixture(tmp_path)
    manifest['paths'][0]['path'] = 'LightGenV2/common/shared.py'
    with pytest.raises(RuntimeError,match='Unsafe'):
        prepare(tmp_path,manifest,'refused')


def test_rejects_stale_main(tmp_path):
    manifest,_ = fixture(tmp_path)
    manifest['expected_main'] = '0'*40
    with pytest.raises(RuntimeError,match='main changed'):
        prepare(tmp_path,manifest,'refused')
