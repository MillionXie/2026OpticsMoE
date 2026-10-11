"""Checkpoint and prepared-data policies; independent of Torch and test scores."""
import math

VALIDATION_RULE = 'validation EMA balanced NLL minimum over 100 epochs; first exact tie retained; test once after checkpoint lock'
LEGACY_TEST_RULE = 'test development accuracy maximum over raw and EMA, ties balanced NLL minimum; validation diagnostic only'


def selection_mode(profile):
    if profile['selection'] == VALIDATION_RULE:
        assert profile['test_development'] is False
        return 'validation'
    assert profile['selection'] == LEGACY_TEST_RULE and profile['test_development'] is True
    return 'test_development'


def validation_improves(metric, best_balanced_nll):
    value = metric['balanced_nll']
    assert math.isfinite(value) and value >= 0
    return value < best_balanced_nll


def validate_prepared_profile(profile, manifest, preparation_profile, preparation_sha):
    """A selection-only profile must retain the exact original preparation identity."""
    assert preparation_sha == manifest['profile_sha256'], 'Prepared profile identity mismatch'
    if selection_mode(profile) == 'validation':
        assert profile['prepared_data_profile_sha256'] == preparation_sha
        allowed = {'selection', 'test_development', 'scope', 'prepared_data_profile_sha256'}
        changed = {key for key in set(profile) | set(preparation_profile)
                   if profile.get(key) != preparation_profile.get(key)}
        assert changed <= allowed, ('Selection-only change altered data/training contract', changed - allowed)
    else:
        assert profile == preparation_profile
    for key in ['dataset', 'source', 'doi', 'license', 'classes', 'split_seed', 'archive_sha256']:
        assert profile[key] == manifest[key], ('Prepared data contract mismatch', key)
