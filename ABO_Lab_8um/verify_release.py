"""Python-stdlib-only immutable release verification; local lab config excluded."""
from common import ROOT,read,sha
def main():
    manifest=read(ROOT/'RELEASE_MANIFEST.json')
    for name,expected in manifest['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or sha(p)!=expected:
            raise RuntimeError('Release file changed or missing: '+name)
    print('Verified',len(manifest['files']),'files. Git:',manifest['git_commit'])
if __name__=='__main__': main()
