import pytest

from LightGenV2.tasks.t19_resnet20_baselines import train_easy100 as training


@pytest.mark.parametrize('published',[True,False])
def test_old_published_main_is_valid_but_unpublished_commit_is_not(monkeypatch,published):
    def git(*args):
        if args[0]=='branch':return 'main'
        if args[0]=='status':return ''
        if args[0]=='rev-parse':return 'old_published'
        if args[0]=='merge-base':return 'old_published' if published else 'earlier'
        if args[0]=='ls-remote':return 'unrelated\trefs/heads/main'
        raise AssertionError(args)
    monkeypatch.setattr(training,'git',git)
    if published:training.require_published_source()
    else:
        with pytest.raises(RuntimeError,match='published commit'):training.require_published_source()


def test_directly_published_server_head_with_stale_tracking_ref(monkeypatch):
    def git(*args):
        return {'branch':'main','status':'','rev-parse':'published_on_server',
                'merge-base':'stale_tracking_ref','ls-remote':'published_on_server\trefs/heads/main'}[args[0]]
    monkeypatch.setattr(training,'git',git)
    training.require_published_source()
