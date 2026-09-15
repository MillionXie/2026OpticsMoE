"""Export six pinned baseline source snapshots, with local import/config closure.

Only reads Git objects. Never copies the dirty worktree, data, model weights or
credentials. Generated handoffs are source archives, not laboratory packages.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import posixpath
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'LightGenV2/reports/20260915_baseline_methods/code_handoff.json'
LOCAL = ('LightGenV2', 'experiments')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


class Snapshot:
    def __init__(self, commit):
        self.commit = git('rev-parse', commit + '^{commit}').decode().strip()
        self.files = {}
        for line in git('ls-tree', '-r', self.commit).decode().splitlines():
            meta, path = line.split('\t', 1)
            if meta.split()[1] == 'blob':
                self.files[path] = meta.split()[2]
        self.data = {}

    def read(self, path):
        if path not in self.data:
            self.data[path] = git('cat-file', 'blob', self.files[path])
        return self.data[path]

    def modules(self, name):
        base = name.replace('.', '/')
        return [p for p in (base + '.py', base + '/__init__.py') if p in self.files]

    def dependencies(self, path):
        if not path.endswith('.py'):
            return set()
        tree = ast.parse(self.read(path).decode('utf-8-sig'), filename=path)
        package = path.rsplit('/', 1)[0].replace('/', '.')
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split('.')
                    base = '.'.join(parts[:len(parts) - node.level + 1])
                    base += ('.' + node.module) if node.module else ''
                else:
                    base = node.module or ''
                names.add(base)
                names.update(base + '.' + a.name for a in node.names if a.name != '*')
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith(tuple(x + '.' for x in LOCAL)):
                    names.add(node.value)
        result = set()
        for name in names:
            if name.split('.')[0] in LOCAL:
                result.update(self.modules(name))
        # Package initializers can themselves import local modules.
        for parent in Path(path).parents:
            candidate = parent.as_posix() + '/__init__.py'
            if candidate in self.files:
                result.add(candidate)
        return result

    def closure(self, seeds):
        selected, pending = set(), list(seeds)
        while pending:
            path = pending.pop()
            if path in selected:
                continue
            if path not in self.files:
                raise FileNotFoundError(f'{self.commit}:{path}')
            selected.add(path)
            pending.extend(self.dependencies(path) - selected)
        # Shared settings load YAMLs indirectly. Include the configuration trees
        # of reached packages and their recursive base-config references.
        owners = set()
        for path in selected:
            bits = path.split('/')
            if bits[0] == 'experiments' and len(bits) > 2:
                owners.add('/'.join(bits[:2]))
            elif bits[:2] == ['LightGenV2', 'tasks'] and len(bits) > 3:
                owners.add('/'.join(bits[:3]))
        configs = {p for p in self.files if p.endswith(('.yaml', '.yml'))
                   and any(p.startswith(owner + '/configs/') for owner in owners)}
        pending = list(configs | {p for p in selected if p.endswith(('.yaml', '.yml'))})
        checked = set()
        while pending:
            path = pending.pop()
            if path in checked:
                continue
            checked.add(path)
            selected.add(path)
            text = self.read(path).decode('utf-8-sig')
            for value in re.findall(r'^\s*(?:base_config|optical_base_config):\s*[\"\']?([^\s\"\'#]+)', text, re.M):
                target = posixpath.normpath(posixpath.join(posixpath.dirname(path), value))
                if target in self.files:
                    pending.append(target)
                else:
                    # Legacy readers also resolve configurations by package suffix.
                    suffix = value.lstrip('./')
                    matches = [p for p in self.files if p.endswith(suffix)]
                    if not matches:
                        matches = [p for p in self.files if p.endswith('/' + posixpath.basename(value))]
                    if not matches:
                        raise FileNotFoundError(f'Unresolved config {path}: {value}')
                    pending.extend(matches)
        return sorted(selected)


RUNNER = r'''"""Baseline handoff launcher; stdlib-only checks, original task code at execution."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT/'TASK.json').read_text(encoding='utf-8'))

def verify():
    m=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for row in m['files']:
        p=ROOT/'source'/row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Source differs from snapshot: '+row['path'])
    print('Verified',len(m['files']),'original files at',m['source_commit'])

def initialize_git():
    source=ROOT/'source'
    if (source/'.git').exists(): return
    subprocess.run(['git','init','--quiet'],cwd=source,check=True)
    subprocess.run(['git','add','.'],cwd=source,check=True)
    subprocess.run(['git','-c','user.name=Baseline Handoff','-c','user.email=baseline@localhost',
                    'commit','--quiet','-m','Import pinned baseline source snapshot'],cwd=source,check=True)
    # This is a real local snapshot commit, not an impersonation of the original
    # repository commit. SOURCE_ORIGIN.json preserves the historical identity.

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','list','init-source']+list(SPEC['actions']))
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.action=='check': verify(); return
    if a.action=='list':
        for name,command in SPEC['actions'].items(): print(name,': python -m',' '.join(command))
        return
    if a.action=='init-source': verify(); initialize_git(); return
    if not (ROOT/'source/.git').is_dir():
        p.error('First run: python run_baseline.py init-source')
    tail=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    command=[sys.executable,'-m',*SPEC['actions'][a.action],*tail]
    result=subprocess.run(command,cwd=ROOT/'source')
    raise SystemExit(result.returncode)
if __name__=='__main__': main()
'''


def build(spec, output):
    # Windows source paths can exceed MAX_PATH while preserving the original
    # long experiment names. Extended paths affect only filesystem access.
    if os.name == 'nt' and not str(output).startswith('\\\\?\\'):
        output = Path('\\\\?\\' + str(output.resolve()))
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for task in spec['tasks']:
        dest = output / task['directory']
        source = dest / 'source'
        source.mkdir(parents=True)
        snapshot = Snapshot(task['commit'])
        seeds = set(task['files'])
        for command in task['actions'].values():
            seeds.update(snapshot.modules(command[0]))
        paths = snapshot.closure(seeds)
        records = []
        for path in paths:
            content = snapshot.read(path)
            if path.endswith('.py'):
                compile(content, path, 'exec')
            target = source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            records.append(dict(path=path, git_blob=snapshot.files[path], sha256=hashlib.sha256(content).hexdigest()))
        externals = set()
        for path in paths:
            if not path.endswith('.py'): continue
            for node in ast.walk(ast.parse(snapshot.read(path).decode('utf-8-sig'))):
                if isinstance(node, ast.Import): externals.update(a.name.split('.')[0] for a in node.names)
                if isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    externals.add(node.module.split('.')[0])
        externals -= set(sys.stdlib_module_names) | set(LOCAL) | {'__future__'}
        manifest = dict(source_commit=snapshot.commit, files=records, external_import_roots=sorted(externals),
                        verification='Original Git bytes and Python syntax checked; no new full-dataset inference or retraining.')
        write_json(dest/'SOURCE_MANIFEST.json', manifest)
        write_json(source/'SOURCE_ORIGIN.json', dict(source_commit=snapshot.commit, task=task['directory']))
        write_json(dest/'TASK.json', task)
        references = json.loads((SPEC.parent/'VERSION_MAP.json').read_text(encoding='utf-8'))
        selected = [r for r in references['rows'] if r['table_order'] == task['directory'][:2]]
        for row in selected:
            row.pop('checkpoint', None)
        write_json(dest/'REFERENCE.json', dict(model_snapshots=references['model_snapshots'], rows=selected))
        (dest/'run_baseline.py').write_text(RUNNER, encoding='utf-8')
        (dest/'requirements.txt').write_text(spec['requirements'], encoding='utf-8')
        (dest/'METHODS.md').write_bytes((ROOT/task['methods']).read_bytes())
        (dest/'README.md').write_text(readme(task), encoding='utf-8')
        # Every artifact is fixed to the source version and checked independently.
        subprocess.run([sys.executable, str(dest/'run_baseline.py'), 'check'], check=True)
        zip_path = output/(task['directory']+'.zip')
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(dest.rglob('*')):
                if path.is_file(): archive.write(path, path.relative_to(output).as_posix())
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        summaries.append(dict(task=task['directory'], source_files=len(records), zip_bytes=zip_path.stat().st_size,
                              sha256=digest, source_commit=snapshot.commit, external_import_roots=sorted(externals)))
        print(task['directory'],len(records),'source files',zip_path.stat().st_size,'ZIP bytes',flush=True)
    write_json(output/'BUILD_MANIFEST.json', dict(tasks=summaries))
    (output/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['task']+'.zip\n' for r in summaries),encoding='utf-8')
    (output/'README.md').write_text('# Baseline 代码交接包\n\n'+ '\n'.join(
        f"{i+1}. [{t['title']}]({t['directory']}/README.md) · [ZIP]({t['directory']}.zip)" for i,t in enumerate(spec['tasks']))+'\n',encoding='utf-8')
    return summaries


def write_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def readme(task):
    return f'''# {task['title']} baseline 代码

对应指标：{task['metric']}。源码版本：`{task['commit']}`。

`source/` 保存该版本的原始源码和依赖配置，`METHODS.md` 为技术说明，`SOURCE_MANIFEST.json` 为逐文件校验清单。公共模块保留原目录结构，训练时仅运行下方 baseline 入口。

## 准备

使用 Python 3.11、Git 和 CUDA 环境，安装 `requirements.txt`（PyTorch 2.6.0 对应 CUDA 12.4；其他依赖未作为完整锁文件固定）。
在当前目录执行：

```bash
python run_baseline.py check
python run_baseline.py init-source
```

第二条命令为导出的源码建立真实的本地 Git 快照，供原训练器记录运行版本；历史来源仍以 `SOURCE_ORIGIN.json` 为准。
配置路径均相对于 `source/`。数据、Qwen 预训练权重、训练后的任务头及特征缓存需另外提供，包内不含这些文件。
{task['assets']}

## 运行

{task['instructions']}

原入口的完整参数可通过 `python run_baseline.py <入口名> -- --help` 查看。首次修改配置前可先执行 `check`；修改后的配置和训练记录归属于新复现运行。
代码包已进行源码字节、依赖收集和语法校验；本次整理未重新训练或运行完整数据集评价。固定权重复评需使用下列权重身份，重新训练所得指标另行报告。

## 权重与数据身份

{task['identities']}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,default=SPEC)
    p.add_argument('--output',type=Path,required=True,help='Must be a new directory; existing handoffs are never overwritten')
    args=p.parse_args()
    spec=json.loads(args.spec.read_text(encoding='utf-8'))
    build(spec,args.output.resolve())


if __name__ == '__main__': main()
