"""Human-approved02:00 cutoff, only not-yet-qualified original points.

Switch controllers only after a whole point passes and before another begins
formal input processing. Never restart an accepted point or relax a gate.
"""
import json
from pathlib import Path
import sys
import run_rental_energy_remeasure_20261029 as entry

ENTRY=Path(__file__).resolve();PRIOR=entry.ROOT;SECOND=PRIOR.with_name('rental_energy_remeasure_20261029_02')
ROOT=PRIOR.with_name('rental_energy_remeasure_20261030_03')
PIPE='/data/tmp/rental_energy_remeasure_20261030_03';DEADLINE=1793296800.0
NATIVE_CONFIGURE=entry.configure;NATIVE_PLAN=entry.native.plan
FIRST5=['kimi_cap50_repeat01','kimi_cap50_repeat02','kimi_cap50_repeat03','kimi_cap100_repeat01','kimi_cap100_repeat02']

def qualified():
    a=json.loads((PRIOR/'suite_status.json').read_text());b=json.loads((SECOND/'suite_status.json').read_text())
    all_names=[f'{m}_cap{c}_repeat{r:02d}' for m,c,r in NATIVE_PLAN('rental')]
    if a['status']!='stopped' or a['completed']!=FIRST5 or not a.get('mps_released'):raise ValueError('Original five-point root differs')
    if b['status']!='stopped' or not b.get('mps_released') or not b['completed'] or b['completed']!=all_names[5:5+len(b['completed'])]:raise ValueError('Exact clean qualified continuation prefix required')
    for root,state in [(PRIOR,a),(SECOND,b)]:
        p=Path('/proc',str(state['pid']),'stat')
        if p.exists() and p.read_text().rsplit(')',1)[1].split()[0]!='Z':raise ValueError('Prior controller alive')
        for name in state['completed']:
            audit=json.loads((root/name/'cpu_audit.json').read_text())
            if audit['status']!='passed' or audit['energy_qualified'] is not True:raise ValueError('Never reuse invalid point')
    phase=SECOND/b['phase'];calls=phase/'calls.csv'
    if calls.exists() and calls.read_text().strip():raise ValueError('No formal point may be interrupted for deadline transition')
    return a['completed']+b['completed']

def plan(profile,models=None,caps=None,repeats=None):
    accepted=qualified()
    return [(m,c,r) for m,c,r in NATIVE_PLAN(profile,models,caps,repeats) if f'{m}_cap{c}_repeat{r:02d}' not in accepted]

def configure():
    NATIVE_CONFIGURE();entry.SOURCES|={Path(entry.__file__).relative_to(entry.REPO).as_posix(),ENTRY.relative_to(entry.REPO).as_posix()};entry.base.plan=plan
    for fn in (entry.base.worker,entry.base.suite,entry.base.summarize,entry.QUALIFIED_AUDIT):
        fn.__globals__.update(vars(entry.base));fn.__globals__.update(ENTRY=ENTRY,NEW_PROTOCOL=entry.PROTOCOL)

if __name__=='__main__':
    if '--deadline' not in sys.argv or float(sys.argv[sys.argv.index('--deadline')+1])!=DEADLINE:raise ValueError('Only human-approved02:00 cutoff')
    entry.ROOT=ROOT;entry.PIPE=PIPE;entry.ENTRY=ENTRY;entry.configure=configure;entry.main()
