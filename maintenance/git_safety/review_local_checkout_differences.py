"""Read-only structural review of protected local files versus published main.

Does not import experiment modules, merge code, run models, or resolve conflicts.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]

def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def declarations(source):
    module=ast.parse(source)
    result={}
    other_counts={}
    for node in module.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            key=node.name
        elif isinstance(node,(ast.Import,ast.ImportFrom)):
            key='import:'+ast.unparse(node)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            target=node.targets[0] if isinstance(node,ast.Assign) else node.target
            key='assignment:'+ast.unparse(target)
        elif isinstance(node,ast.If):key='if:'+ast.unparse(node.test)
        elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            key='module_docstring'
        else:
            kind=type(node).__name__
            other_counts[kind]=other_counts.get(kind,0)+1
            key='statement:'+kind+':'+str(other_counts[kind])
        if key in result:raise ValueError('Duplicate top-level name needs manual review: '+key)
        result[key]=ast.dump(node,include_attributes=False)
    return result

def compare(local,main):
    a=declarations(local);b=declarations(main)
    return dict(local_only=sorted(set(a)-set(b)),main_only=sorted(set(b)-set(a)),
        module_ast_identical=ast.dump(ast.parse(local),include_attributes=False)==ast.dump(ast.parse(main),include_attributes=False),
        different=sorted(name for name in set(a)&set(b) if a[name]!=b[name]),
        identical=sorted(name for name in set(a)&set(b) if a[name]==b[name]))

def main():
    before=git('status','--porcelain','--untracked-files=no')
    head=git('rev-parse','HEAD').decode().strip();target=git('rev-parse','main').decode().strip()
    index=Path(git('rev-parse','--path-format=absolute','--git-path','index').decode().strip())
    index_sha=hashlib.sha256(index.read_bytes()).hexdigest()
    plan=json.loads((ROOT/'.codex_tmp/main_checkout_preflight_20261005.json').read_text(encoding='utf8'))
    names={row['path'] for key in ('tracked_user_edits','existing_new_main_paths') for row in plan[key]
        if row['classification']=='different_from_main_preserve_and_review'}
    rows=[]
    for name in sorted(names):
        path=ROOT/name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):raise ValueError('Unsafe source')
        local=path.read_bytes();published=git('show',target+':'+name)
        row=dict(path=name,local_sha256=hashlib.sha256(local).hexdigest(),main_sha256=hashlib.sha256(published).hexdigest(),
            adopted=False,original_preserved=True)
        if name.endswith('.py'):
            try:row['structural_difference']=compare(local.decode('utf-8-sig'),published.decode('utf-8-sig'))
            except (SyntaxError,ValueError) as error:row['manual_review_reason']=str(error)
        else:row['manual_review_reason']='Documentation/configuration; AST equivalence does not apply'
        rows.append(row)
    if git('status','--porcelain','--untracked-files=no')!=before or git('rev-parse','HEAD').decode().strip()!=head or git('rev-parse','main').decode().strip()!=target or hashlib.sha256(index.read_bytes()).hexdigest()!=index_sha:
        raise RuntimeError('Repository changed during read-only review')
    report=ROOT/'.codex_tmp/local_checkout_structural_review_20261005.json'
    report.write_text(json.dumps(dict(read_only=True,head=head,main=target,rows=rows,conflicts_resolved=False),indent=2),encoding='utf8')
    print(json.dumps(dict(files_reviewed=len(rows),report=str(report),summary=[dict(path=row['path'],
        differing_declarations=row.get('structural_difference',{}).get('different'),
        local_only_declarations=row.get('structural_difference',{}).get('local_only'),
        manual_review_reason=row.get('manual_review_reason')) for row in rows]),indent=2))

if __name__=='__main__':main()
