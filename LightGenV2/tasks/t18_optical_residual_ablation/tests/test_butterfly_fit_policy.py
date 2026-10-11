import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'butterfly_fit_policy.py'
spec = importlib.util.spec_from_file_location('butterfly_fit_policy', path)
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def history(n=100, decreasing=False, overfit=False, low=False):
    rows = []
    for epoch in range(1, n + 1):
        loss = 2 - epoch * .003 if decreasing else 1.0
        if overfit:
            loss = .8 if epoch == 50 else 1.0
        rows.append(dict(epoch=epoch, val=dict(balanced_nll=loss, accuracy=.6),
                         train=dict(accuracy=.95 if overfit else .64, balanced_accuracy=.5 if low else .9)))
    return rows


def test_pair_extends_both_on_one_improving_arm_and_caps_budget():
    action = p.pair_action([history(decreasing=True), history()])
    assert action['action'] == 'extend_both' and action['target_epochs'] == 150
    assert p.pair_action([history(150, True), history(150)])['target_epochs'] == 200
    assert p.pair_action([history(200, True), history(200)])['action'] == 'lock_pair'


def test_overfit_requires_loss_reversal_gap_and_old_best():
    assert p.pair_action([history(overfit=True), history()])['candidate'] == 'regularized'
    assert p.pair_action([history(), history()])['action'] == 'lock_pair'


def test_low_fit_recovery_requires_both_arms_and_is_one_candidate():
    assert p.pair_action([history(low=True), history(low=True)])['candidate'] == 'optimization'
    assert p.pair_action([history(low=True), history()])['action'] == 'lock_pair'


def test_choose_one_pair_by_validation_loss_first_tie_not_accuracy():
    def arm(loss, accuracy):
        return dict(test_evaluations=0, selected_validation=dict(balanced_nll=loss, accuracy=accuracy))
    base = [arm(.3, .7), arm(.5, .8)]
    alt = [arm(.4, .99), arm(.4, .99)]
    assert p.choose_pair([('base', base), ('regularized', alt)])['candidate'] == 'base'
    better = [arm(.1, .6), arm(.6, .6)]
    assert p.choose_pair([('base', base), ('optimization', better)])['candidate'] == 'optimization'
    better[0]['test_evaluations'] = 1
    with pytest.raises(AssertionError):
        p.choose_pair([('base', better)])


def test_test_history_is_rejected():
    rows = history(); rows[5]['test_development'] = dict(accuracy=.99)
    with pytest.raises(AssertionError):
        p.diagnose(rows)
