"""Read-only static local-import audit of published Python source, not runtime proof."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

ROOTS = {'LightGenV2', 'experiments', 'opticalmoe', 'TransferFromElectricity', 'FixedFeedbackSFT'}


def module_exists(module, tracked):
    path = module.replace('.', '/')
    return (path+'.py' in tracked or path+'/__init__.py' in tracked
            or any(name.startswith(path+'/') for name in tracked))


def dependencies(path, text, tracked, include_present=False):
    """Report module references only, never mistake imported classes for modules."""
    package = Path(path).parent.parts
    rows = []
    for node in ast.walk(ast.parse(text, filename=path)):
        targets = []
        if isinstance(node, ast.Import):
            targets = [name.name for name in node.names if name.name.split('.')[0] in ROOTS]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package)-node.level+1
                if keep <= 0: continue
                base = '.'.join(package[:keep])
                if node.module:
                    targets = [base+'.'+node.module]
                else:
                    # A package may export an attribute; report only alias modules
                    # actually present on disk in the caller, via the CLI below.
                    targets = [base+'.'+name.name for name in node.names if name.name != '*']
            elif node.module and node.module.split('.')[0] in ROOTS:
                targets = [node.module]
            if include_present and targets:
                base = targets[0]
                targets += [base+'.'+name.name for name in node.names
                            if name.name != '*' and module_exists(base+'.'+name.name, tracked)]
        for module in targets:
            present = module_exists(module, tracked)
            if include_present or not present:
                rows.append(dict(source=path,line=node.lineno,module=module,
                                 published_module_present=present,
                                 alias_may_be_package_attribute=isinstance(node,ast.ImportFrom) and node.module is None))
    return rows


def module_source_paths(module, tracked):
    parts = module.split('.')
    candidates = ['/'.join(parts[:i])+'/__init__.py' for i in range(1,len(parts)+1)]
    candidates.append('/'.join(parts)+'.py')
    return [path for path in candidates if path in tracked]


def audit(root, reference, transitive=False):
    root = Path(root).resolve()
    def git(*args): return subprocess.check_output(['git','-C',str(root),*args])
    pin = git('rev-parse',reference+'^{commit}').decode().strip()
    tracked = set(git('ls-tree','-r','--name-only',pin).decode().splitlines())
    selected = sorted(p for p in tracked if p.startswith('LightGenV2/tasks/') and p.endswith('.py'))
    rows=[]; syntax=[]; checked=set(); pending=list(selected)
    while pending:
        path=pending.pop()
        if path in checked: continue
        checked.add(path)
        try: found = dependencies(path,git('show',pin+':'+path).decode('utf8'),tracked,include_present=transitive)
        except SyntaxError as error:
            syntax.append(dict(source=path,line=error.lineno)); continue
        for row in found:
            if row['published_module_present']:
                pending.extend(module_source_paths(row['module'],tracked))
                continue
            candidate = root/(row['module'].replace('.','/')+'.py')
            package = root/row['module'].replace('.','/')/'__init__.py'
            row['untracked_local_module_present'] = candidate.is_file() or package.is_file()
            if not row['alias_may_be_package_attribute'] or row['untracked_local_module_present']:
                rows.append(row)
    return dict(commit=pin,task_files_selected=len(selected),files_checked=len(checked),transitive=transitive,
                unpublished_local_imports=rows,syntax_errors=syntax,
                read_only=True,runtime_dependency_closure_proven=False,
                limitations=['Static references include optional/test imports and are not proof of active execution',
                             'External packages, dynamic imports, SDKs and data assets are outside this audit'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--reference',default='HEAD')
    p.add_argument('--transitive',action='store_true',help='Follow matched local modules and package initializers recursively')
    args=p.parse_args(); result=audit(args.repo,args.reference,args.transitive)
    print(json.dumps(result,indent=2))
    return bool(result['unpublished_local_imports'] or result['syntax_errors'])


if __name__=='__main__': raise SystemExit(main())
