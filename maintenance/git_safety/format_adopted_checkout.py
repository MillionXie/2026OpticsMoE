"""Mechanical newline repair for newly Git-materialized, SHA-bound sources.

Never waive a manifest hash. Only repair when the exact Git blob has the
required hash and the current file differs solely by CRLF. Existing user
files, original runtime directories and arbitrary source edits are excluded.
"""
import hashlib
from pathlib import Path
import subprocess


def plan(root, git, pin, mismatches, eligible):
    root=Path(root).resolve()
    rows=[]
    seen={}
    for row in mismatches:
        relative=row['path']
        identity=(row['raw_sha256'],row['expected_sha256'])
        if relative in seen:
            if seen[relative]!=identity:raise ValueError('Conflicting manifest contracts: '+relative)
            continue
        seen[relative]=identity
        target=root/relative
        if (relative not in eligible or row['classification']!='line_endings_only'
                or not target.resolve().is_relative_to(root) or target.is_symlink()):
            raise ValueError('Not an eligible newly adopted newline-only source: '+relative)
        raw=target.read_bytes()
        canonical=subprocess.check_output([git,'-C',str(root),'show',pin+':'+relative])
        if (hashlib.sha256(raw).hexdigest()!=row['raw_sha256']
                or hashlib.sha256(canonical).hexdigest()!=row['expected_sha256']
                or raw.replace(b'\r\n',b'\n')!=canonical):
            raise ValueError('Concurrent edit or noncanonical manifest identity: '+relative)
        rows.append((relative,raw,canonical))
    return rows


def apply(root, git, pin, rows):
    root=Path(root).resolve()
    head=subprocess.check_output([git,'-C',str(root),'rev-parse','HEAD']).decode().strip()
    if head!=pin:raise ValueError('HEAD changed')
    index_tree=subprocess.check_output([git,'-C',str(root),'write-tree']).strip()
    head_tree=subprocess.check_output([git,'-C',str(root),'rev-parse',pin+'^{tree}']).strip()
    if index_tree!=head_tree:raise ValueError('Existing staged changes; do not modify index')
    changed=[]
    try:
        for relative,original,canonical in rows:
            target=root/relative
            if target.read_bytes()!=original:raise ValueError('Concurrent source edit')
            target.write_bytes(canonical)  # mechanical Git-byte newline formatting only
            changed.append((target,original,canonical))
        # Future checkouts retain Git LF; pre-existing CRLF user files are not
        # rewritten. Explicit manifest SHA checks remain strict and separate.
        subprocess.check_call([git,'-C',str(root),'config','core.autocrlf','input'])
        refresh_canonical_index(root,git,pin,[relative for relative,_,_ in rows])
    except Exception:
        for target,original,canonical in reversed(changed):
            if target.read_bytes()!=canonical:raise RuntimeError('Concurrent edit prevents rollback')
            target.write_bytes(original)
        raise
    return {'repaired_files':len(changed),'expected_hashes_waived':False,
            'formatting_only':True,'scope':'Only explicitly eligible new main files; original runtime assets untouched'}


def refresh_canonical_index(root,git,pin,paths):
    """Refresh Git conversion/stat metadata without staging any source change."""
    root=Path(root).resolve()
    before=subprocess.check_output([git,'-C',str(root),'write-tree']).strip()
    expected=subprocess.check_output([git,'-C',str(root),'rev-parse',pin+'^{tree}']).strip()
    if before!=expected:raise ValueError('Existing staged changes')
    for relative in paths:
        target=root/relative
        if not target.resolve().is_relative_to(root) or target.is_symlink():
            raise ValueError('Unsafe source path')
        canonical=subprocess.check_output([git,'-C',str(root),'show',pin+':'+relative])
        if target.read_bytes()!=canonical:raise ValueError('Source is not exact canonical Git bytes')
    # Git for Windows can retain the previous checkout conversion metadata
    # after a mechanical CRLF->LF repair. Refresh only byte-identical paths.
    # No new source is staged: the complete index tree must remain unchanged.
    for start in range(0,len(paths),16):
        subprocess.check_call([git,'-C',str(root),'-c','core.autocrlf=false','add','--',*paths[start:start+16]])
    after=subprocess.check_output([git,'-C',str(root),'write-tree']).strip()
    if after!=before:raise RuntimeError('Index tree changed unexpectedly; stop for review')
    return {'refreshed_paths':len(paths),'index_tree_unchanged':True}
