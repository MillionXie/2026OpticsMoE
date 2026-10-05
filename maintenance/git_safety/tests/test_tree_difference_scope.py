import importlib.util
from pathlib import Path


def test_separates_main_additions_old_only_and_shared_change(monkeypatch):
    path = Path(__file__).parents[1] / 'review_git.py'
    spec = importlib.util.spec_from_file_location('review_scope', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def fake_git(root, *args):
        if args[0] == 'ls-tree':
            return 'same\0changed\0new_main\0' if args[-1] == 'main' else 'same\0changed\0old_only\0'
        assert args[:4] == ('diff', '--name-only', '--no-renames', '-z')
        return 'changed\0new_main\0old_only\0'
    monkeypatch.setattr(module, 'git', fake_git)
    result = module.tree_difference_scope(Path('.'), 'historic')
    assert result['main_only_paths'] == result['reference_only_paths'] == 1
    assert result['shared_paths_with_changed_content'] == 1
    assert result['total_changed_paths'] == 3
    assert result['working_overlay_included'] is False
