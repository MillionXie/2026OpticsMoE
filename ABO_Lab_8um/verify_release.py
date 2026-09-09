"""Python-stdlib-only immutable release verification; local lab config excluded."""
from common import ROOT,read,sha
import argparse
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--immutable-only',action='store_true',help='Verify code/weights/data, excluding regenerated hardware BMPs and manifests')
    args=parser.parse_args(); checked=0; skipped=0
    manifest=read(ROOT/'RELEASE_MANIFEST.json')
    for name,expected in manifest['files'].items():
        if args.immutable_only and name.startswith('generated/'):
            skipped+=1; continue
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or sha(p)!=expected:
            raise RuntimeError('Release file changed or missing: '+name)
        checked+=1
    print('Verified',checked,'files. Git:',manifest['git_commit'])
    if skipped: print('Excluded',skipped,'configuration-generated hardware files; these were NOT compared to release hashes.')
if __name__=='__main__': main()
