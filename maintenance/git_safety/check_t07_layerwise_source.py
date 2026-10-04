"""Read-only source identity and original AST audit; never load models/devices."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT=Path(__file__).resolve().parents[2]
RECEIPT='maintenance/storage/T07_LAYERWISE_SOURCE_IMPORT_20261005.json'

def git_blob(commit,path):
    return subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+path])

def function(text,name):
    return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)

def audit(commit,archive=False):
    row=json.loads(git_blob(commit,RECEIPT))
    blob=git_blob(commit,row['published_path'])
    if hashlib.sha256(blob.replace(b'\r\n',b'\n')).hexdigest()!=row['sha256_lf']:
        raise ValueError('Published layerwise source SHA mismatch')
    source=blob.decode('utf8')
    tree=ast.parse(source)
    heavy={'torch','transformers'}
    for node in tree.body:
        if isinstance(node,ast.Import) and any(n.name.split('.')[0] in heavy for n in node.names):
            raise ValueError('Heavy import outside explicit execution mode')
        if isinstance(node,ast.ImportFrom) and (node.module or '').split('.')[0] in heavy:
            raise ValueError('Heavy import outside explicit execution mode')
    if 'sys.path' in source or 'E:/code' in source: raise ValueError('Old independent project dependency')
    checked=[]
    if archive:
        originals={}
        for item in row['sources']:
            raw=git_blob(row['source_ref'],item['path'])
            if hashlib.sha256(raw).hexdigest()!=item['sha256']: raise ValueError('Archive SHA mismatch')
            originals[item['path'].rsplit('/',1)[-1]]=raw.decode('utf8')
        for name in row['unchanged_functions_ast']:
            old=originals['four_image_flow.py' if name=='phase_planes' else 'abo_full_query_flow_snapshot_20260928.py']
            if ast.dump(function(old,name),include_attributes=False)!=ast.dump(function(source,name),include_attributes=False):
                raise ValueError('Original function AST changed: '+name)
            checked.append(name)
    return dict(source_sha256_verified=True,original_functions_ast_verified=checked,
                model_loaded=False,devices_opened=False,hardware_regression_completed=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',default='main')
    parser.add_argument('--audit-archive',action='store_true')
    args=parser.parse_args()
    print(json.dumps(audit(args.commit,args.audit_archive),indent=2))

if __name__=='__main__':main()
