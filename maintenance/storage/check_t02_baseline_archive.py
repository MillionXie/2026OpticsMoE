"""Read-only verification of the preserved LSP baseline timing/head archive.

Never extracts files, loads a model, measures hardware or overwrites old reports.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                std=statistics.pstdev(values), p05=percentile(values, 5),
                p95=percentile(values, 95), min=min(values), max=max(values))


def verify(path, spec, official_manifest=None):
    errors = []
    result = dict(read_only=True, archive_extracted=False, model_loaded=False,
                  benchmark_executed=False, independent_performance_reproduction_proven=False)
    if sha_file(path) != spec['archive_sha256']:
        return {**result, 'errors': ['Archive SHA mismatch']}
    payloads = {}
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            return {**result, 'errors': ['Duplicate archive member names']}
        for key, expected in spec['members'].items():
            try:
                member = archive.getmember(expected['member'])
            except KeyError:
                errors.append('Missing member: '+key)
                continue
            if not member.isfile() or member.size != expected['bytes']:
                errors.append('Member type/size mismatch: '+key)
                continue
            digest = hashlib.sha256()
            chunks = []
            with archive.extractfile(member) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
                    if key != 'head':
                        chunks.append(block)
            if digest.hexdigest() != expected['sha256']:
                errors.append('Member SHA mismatch: '+key)
            elif key != 'head':
                payloads[key] = b''.join(chunks)
    if errors:
        return {**result, 'errors': errors}
    report = json.loads(payloads['report'])
    for key, expected in spec['report_fields'].items():
        if report.get(key) != expected:
            errors.append('Report identity mismatch: '+key)
    if report.get('checkpoint_sha256') != spec['members']['head']['sha256']:
        errors.append('Reported head SHA does not bind preserved head')
    rows = list(csv.DictReader(io.StringIO(payloads['timing'].decode('utf8'))))
    ids = [row['sample_id'] for row in rows]
    if len(rows) != report['timing_samples'] or len(ids) != len(set(ids)):
        errors.append('Timing sample count or unique IDs mismatch')
    if [int(row['sample_index']) for row in rows] != list(range(len(rows))):
        errors.append('Timing sample indices are not consecutive')
    for column, field in (('cuda_ms', 'latency_cuda_ms'), ('host_ms', 'latency_host_ms')):
        values = [float(row[column]) for row in rows]
        if not values or not all(math.isfinite(value) and value > 0 for value in values):
            errors.append('Invalid latency values: '+column)
            continue
        calculated = summary(values)
        for key, value in calculated.items():
            if not math.isclose(value, report[field][key], rel_tol=1e-9, abs_tol=1e-8):
                errors.append('Raw timing summary mismatch: '+field+'.'+key)
        result[field] = calculated
    power = list(csv.DictReader(io.StringIO(payloads['power'].decode('utf8'))))
    idle = [float(row['watts']) for row in power if row['phase'] == 'idle']
    active = [float(row['watts']) for row in power if row['phase'].startswith('active:')]
    if len(idle) != report['power']['idle_samples'] or len(active) != report['power']['active_samples']:
        errors.append('Raw power sample count mismatch')
    for label, values in (('idle', idle), ('active', active)):
        if not values or not all(math.isfinite(value) and value > 0 for value in values):
            errors.append('Invalid raw power: '+label)
        elif not math.isclose(statistics.mean(values), report['power'][label+'_mean_w'], abs_tol=1e-8):
            errors.append('Raw power mean mismatch: '+label)
    if active and not math.isclose(max(active), report['power']['active_peak_w'], abs_tol=1e-8):
        errors.append('Raw power peak mismatch')
    energy = report['power']['active_mean_w'] * report['latency_cuda_ms']['mean'] / 1000
    if not math.isclose(energy, report['power']['measured_active_energy_j_per_sample'], abs_tol=1e-8):
        errors.append('Original power-times-latency energy proxy mismatch')
    membership = 'not checked: official split not supplied'
    if official_manifest is not None:
        if sha_file(official_manifest) != spec['official_manifest_sha256']:
            errors.append('Official split SHA mismatch')
        else:
            with Path(official_manifest).open(encoding='utf8', newline='') as stream:
                official = list(csv.DictReader(stream))
            test_ids = {row['sample_id'] for row in official if row['split'] in ('test', 'periodic_test')}
            if len(test_ids) != report['test_samples'] or not set(ids) <= test_ids:
                errors.append('Timing IDs are not contained in the official TEST split')
            membership = 'checked by original split SHA and TEST membership; not pixel-content comparison'
    result.update(errors=errors, test_samples=report['test_samples'], timing_samples=len(rows),
                  explicit_warmup_forwards=report['explicit_warmup_forwards'],
                  reported_first_test_sample_included=report['first_test_sample_included'],
                  timing_id_sha256=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
                  official_test_membership=membership, head_sha256=report['checkpoint_sha256'],
                  power_samples=len(power), measured_active_energy_j_per_sample=energy,
                  scope='Existing head/report and sampled latency/power binding only; not full-model assets, cold-start timing, coordinate decoding or a new performance evaluation')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, default=Path(__file__).with_name('T02_BASELINE_TIMING_BINDING_20261006.json'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--official-manifest', type=Path)
    args = parser.parse_args()
    spec = json.loads(args.binding.read_text(encoding='utf8'))
    try:
        result = verify(args.archive or Path(spec['server_archive']), spec, args.official_manifest)
    except (OSError, ValueError, KeyError, tarfile.TarError, json.JSONDecodeError) as exc:
        result = dict(errors=[type(exc).__name__+': '+str(exc)], read_only=True, archive_extracted=False)
    print(json.dumps(result, indent=2))
    return bool(result['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
