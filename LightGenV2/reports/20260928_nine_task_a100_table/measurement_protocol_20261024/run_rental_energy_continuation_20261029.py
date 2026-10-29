"""One bounded remaining-seven attempt; retain the five qualified prior points.

The failed sixth original leaf is never promoted/resumed/overwritten. Same
models, heads,512 inputs, mathematics and250ms API/support gates. No renewal.
"""
import json
from pathlib import Path
import sys
import run_rental_energy_remeasure_20261029 as entry

ENTRY=Path(__file__).resolve()
PRIOR=entry.ROOT
ROOT=PRIOR.with_name('rental_energy_remeasure_20261029_02')
PIPE='/data/tmp/rental_energy_remeasure_20261029_02'
PREFIX=['kimi_cap50_repeat01','kimi_cap50_repeat02','kimi_cap50_repeat03','kimi_cap100_repeat01','kimi_cap100_repeat02']
DEADLINE=1793294132.4536557
NATIVE_CONFIGURE=entry.configure
NATIVE_PLAN=entry.native.plan

def plan(profile,models=None,caps=None,repeats=None):
    points=NATIVE_PLAN(profile,models,caps,repeats)
    return [(m,c,r) for m,c,r in points if f'{m}_cap{c}_repeat{r:02d}' not in PREFIX]

def predecessor():
    s=json.loads((PRIOR/'suite_status.json').read_text())
    if s['status']!='stopped' or s['completed']!=PREFIX or s.get('mps_released') is not True:raise ValueError('Exact five-qualified-point terminal predecessor required')
    failed=PRIOR/'kimi_cap100_repeat03';f=json.loads((failed/'status.json').read_text())
    if f['status']!='failed' or f['samples']!=267 or 'Power API exceeds original250ms bound' not in f['error'] or (failed/'report.json').exists():raise ValueError('Preserve exact invalid power-only leaf')
    for pid in [s['pid'],f['pid']]:
        p=Path('/proc',str(pid),'stat')
        if p.exists() and p.read_text().rsplit(')',1)[1].split()[0]!='Z':raise ValueError('Prior process still alive')
    for name in PREFIX:
        a=json.loads((PRIOR/name/'cpu_audit.json').read_text())
        if a['status']!='passed' or a['energy_qualified'] is not True:raise ValueError('Keep only actually qualified prior points')

def configure():
    NATIVE_CONFIGURE()
    entry.SOURCES|={Path(entry.__file__).relative_to(entry.REPO).as_posix(),ENTRY.relative_to(entry.REPO).as_posix()}
    entry.base.plan=plan
    for fn in (entry.base.worker,entry.base.suite,entry.base.summarize,entry.QUALIFIED_AUDIT):
        fn.__globals__.update(vars(entry.base));fn.__globals__.update(ENTRY=ENTRY,NEW_PROTOCOL=entry.PROTOCOL)

if __name__=='__main__':
    if '--deadline' not in sys.argv or float(sys.argv[sys.argv.index('--deadline')+1])!=DEADLINE:raise ValueError('Original new-run hard deadline must not move')
    if '--phase' in sys.argv and sys.argv[sys.argv.index('--phase')+1]=='suite':predecessor()
    entry.ROOT=ROOT;entry.PIPE=PIPE;entry.ENTRY=ENTRY;entry.configure=configure
    entry.main()
