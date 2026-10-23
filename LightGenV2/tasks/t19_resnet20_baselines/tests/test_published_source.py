import pytest

from LightGenV2.tasks.t19_resnet20_baselines import train_easy100 as training


@pytest.mark.parametrize('published',[True,False])
def test_old_published_main_is_valid_but_unpublished_commit_is_not(monkeypatch,published):
    def git(*args):
        if args[0]=='branch':return 'main'
        if args[0]=='status':return ''
        if args[0]=='rev-parse':return 'old_published'
        if args[0]=='merge-base':return 'old_published' if published else 'earlier'
        raise AssertionError(args)
    monkeypatch.setattr(training,'git',git)
    if published:training.require_published_source()
    else:
        with pytest.raises(RuntimeError,match='published commit'):training.require_published_source()
