import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prune_missing_worktrees import run


def fixture(tmp_path):
    root=tmp_path/'repository';root.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.STDOUT)
    git('init');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
    (root/'protected.txt').write_text('protected\n')
    git('add','protected.txt');git('commit','-m','fixture')
    head=git('rev-parse','HEAD').decode().strip()
    admin=root/'.git/worktrees/missing_fixture';admin.mkdir(parents=True)
    (admin/'HEAD').write_text(head+'\n')
    (admin/'commondir').write_text('../..\n')
    (admin/'gitdir').write_text(str(tmp_path/'absent' / '.git')+'\n')
    return root,git,admin


def test_prunes_missing_metadata_but_preserves_history_and_files(tmp_path):
    root,git,admin=fixture(tmp_path)
    before=git('status','--porcelain');head=git('rev-parse','HEAD')
    preview=run(root,['missing_fixture'])
    assert not preview['apply'] and admin.exists()
    result=run(root,['missing_fixture'],tmp_path/'recovery',True)
    assert result['removed_registration_count']==1 and not admin.exists()
    assert (root/'protected.txt').read_text()=='protected\n'
    assert git('status','--porcelain')==before and git('rev-parse','HEAD')==head
    assert git('rev-parse',result['rows'][0]['recovery_ref'])==head
    assert Path(result['backup']).is_file()


def test_rejects_any_unreviewed_missing_registration(tmp_path):
    root,_,admin=fixture(tmp_path)
    with pytest.raises(RuntimeError,match='differs'):
        run(root,['different'],tmp_path/'recovery',True)
    assert admin.exists() and not (tmp_path/'recovery').exists()


def test_rejects_missing_gitfile_in_existing_directory(tmp_path):
    root,_,admin=fixture(tmp_path)
    (tmp_path/'absent').mkdir()
    with pytest.raises(RuntimeError,match='directory exists'):
        run(root,['missing_fixture'],tmp_path/'recovery',True)
    assert admin.exists()
