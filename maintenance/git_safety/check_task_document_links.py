"""Read-only Git-tree task Markdown link inventory, separate from runtime validation."""
import argparse
import json
from pathlib import Path
import posixpath
import re
import subprocess
from urllib.parse import unquote,urlsplit


def classify(document,link,paths):
    link=link.strip().strip('<>')
    if link.startswith('#') or urlsplit(link).scheme:return None
    target=posixpath.normpath(posixpath.join(posixpath.dirname(document),unquote(link.split('#',1)[0].split('?',1)[0])))
    if not target or target=='.':return None
    if target in paths or any(p.startswith(target.rstrip('/')+'/') for p in paths):return None
    parts=target.split('/')
    if target.startswith('../') or target.startswith('/'):
        kind='outside_repository'
    elif 'handoffs' in parts:
        kind='private_handoff'
    elif any(p in parts for p in ('runs','releases','outputs','data')) or Path(target).suffix.lower() in ('.pt','.npz','.png','.jpg','.pdf','.zip','.csv','.xlsx'):
        kind='external_or_preserved_artifact'
    else:kind='missing_source_or_document'
    return dict(document=document,link=link,target=target,kind=kind)


def inspect(root,commit):
    ref=subprocess.check_output(['git','-C',str(root),'rev-parse',commit+'^{commit}'],text=True).strip()
    paths=set(subprocess.check_output(['git','-C',str(root),'ls-tree','-r','--name-only','-z',ref]).decode().rstrip('\0').split('\0'))
    docs=sorted(p for p in paths if p.startswith('LightGenV2/tasks/') and p.endswith('.md'))
    unresolved=[];links=0
    for p in docs:
        raw=subprocess.check_output(['git','-C',str(root),'show',ref+':'+p]).decode('utf-8')
        # Inline Markdown destinations only; reference links and anchors are not validated.
        for match in re.finditer(r'\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)',raw):
            links+=1;result=classify(p,match.group(1),paths)
            if result:unresolved.append(result)
    return dict(commit=ref,documents_checked=len(docs),inline_links_checked=links,
                unresolved=unresolved,read_only=True,working_files_used=False,
                limitations='Does not validate anchors, reference-style links, external URLs, private assets or runtime imports')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--commit',required=True)
    args=p.parse_args();print(json.dumps(inspect(Path(__file__).resolve().parents[2],args.commit),indent=2))
