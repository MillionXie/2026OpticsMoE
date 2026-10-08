"""Check committed package source closure without loading models or devices."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT/'LightGenV2/tasks/t06_video_quality_assessment'
ORIGINAL = '7093ec46082eed2fae127ec5028d3e2e8548b592'


def reverse_phase_import_migration(source):
    source = source.replace('import argparse,json,os,sys,time,subprocess', 'import argparse,os,sys,time,subprocess')
    current = (' from LightGenV2.hardware_common.shs.phase_hdmi import PhaseHDMI,load_native\n'
               ' from LightGenV2.hardware_common.shs.phase_owner import message_pump\n'
               ' from LightGenV2.hardware_common.shs.phase_display import DisplayOrigin\n')
    old = (' sys.path.insert(0,str(a.bench_root.resolve()))\n from guarded_workflow import read\n'
           ' from phase_hdmi import PhaseHDMI,load_native\n from phase_owner import message_pump\n'
           ' from phase_display import DisplayOrigin\n')
    return source.replace(current, old).replace("c=json.loads(Path(a.link_config).read_text(encoding='utf-8-sig'))", 'c=read(a.link_config)')


def test_required_phase_and_portable_entry_match_preserved_source():
    for relative in ('lab_phase.py', 'hardware/run_lab.py'):
        path = TASK/relative
        original = subprocess.check_output(['git', 'show', ORIGINAL+':'+path.relative_to(ROOT).as_posix()], cwd=ROOT)
        current = path.read_bytes().replace(b'\r\n', b'\n')
        if relative == 'lab_phase.py':
            current = reverse_phase_import_migration(current.decode()).encode()
        assert current == original
        compile(original, relative, 'exec')


def test_bundle_literal_runtime_paths_are_published():
    tree = ast.parse((TASK/'lab_bundle.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build')
    declaration = next(n for n in function.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'paths' for t in n.targets))
    required = ast.literal_eval(declaration.value)
    required += ['LightGenV2/tasks/t06_video_quality_assessment/hardware/run_lab.py',
                 'LightGenV2/tasks/t06_video_quality_assessment/hardware/COMMAND_SHS.md']
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    assert not [p for p in required if p not in tracked or not (ROOT/p).is_file()]


def test_phase_help_opens_no_sdk_and_creates_no_output(tmp_path):
    before = set(tmp_path.iterdir())
    result = subprocess.run([sys.executable, '-I', str(TASK/'lab_phase.py'), '--help'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '--bench-root' in result.stdout and '--link-config' in result.stdout
    assert set(tmp_path.iterdir()) == before


def test_phase_sdk_imports_are_inside_explicit_main_only():
    tree = ast.parse((TASK/'lab_phase.py').read_text())
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(isinstance(n, ast.ImportFrom) and n.module in
                   ('phase_hdmi', 'guarded_workflow', 'phase_owner', 'phase_display') for n in imports)
    assert not any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) for n in tree.body)


def test_optional_release_is_only_change_from_formal_server_package():
    path = TASK/'lab_phase.py'
    original = subprocess.check_output(['git', 'show', '8e869473787f4ffceb2a6a77f4430b94c206f459:'+path.relative_to(ROOT).as_posix()], cwd=ROOT).decode()
    current = reverse_phase_import_migration(path.read_text())
    current = current.replace(";p.add_argument('--release-file',type=Path)", '')
    current = current.replace(" if a.release_file and a.release_file.exists():raise ValueError('Release file already exists')\n", '')
    current = current.replace("while not (a.release_file and a.release_file.exists()):", "while True:")
    assert current == original


def test_shs_dispatch_requires_explicit_target_output_and_pinned_checkpoint(monkeypatch, tmp_path):
    import types
    from LightGenV2.tasks.t06_video_quality_assessment import build_lab_package as wrapper
    calls=[]
    module=types.ModuleType('LightGenV2.tasks.t06_video_quality_assessment.lab_bundle')
    module.build=lambda args:calls.append(args)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    monkeypatch.setattr(wrapper,'load_profile',lambda *_: (_ for _ in ()).throw(AssertionError('legacy dispatch')))
    for flags in (['--bench','shs'],['--bench','shs','--target','spatial'],
                  ['--bench','shs','--target','spatial','--output',str(tmp_path/'new'),'--checkpoint','other.pt']):
        monkeypatch.setattr(sys,'argv',['builder',*flags])
        with pytest.raises(SystemExit) as exc:
            wrapper.main()
        assert exc.value.code==2
    monkeypatch.setattr(sys,'argv',['builder','--bench','shs','--target','temporal','--output',str(tmp_path/'new'),'--device','cpu'])
    assert wrapper.main()==0
    assert len(calls)==1 and calls[0].target=='temporal' and calls[0].device=='cpu'
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('existing', ['directory','adjacent_zip'])
def test_shs_output_collision_rejected_before_model_load(tmp_path,existing):
    import argparse
    # Execute the pure function definition, not the module's Torch imports.
    function=next(n for n in ast.parse((TASK/'lab_bundle.py').read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='build')
    module=ast.Module(body=[function],type_ignores=[])
    scope={'Path':Path,'__file__':str(TASK/'lab_bundle.py'),
           'load_model':lambda *_: (_ for _ in ()).throw(AssertionError('model loaded before collision check'))}
    exec(compile(ast.fix_missing_locations(module),'fixture_build','exec'),scope)
    out=tmp_path/'release'
    if existing=='directory':out.mkdir()
    else:out.with_suffix('.zip').write_bytes(b'existing package')
    before={p.name:p.read_bytes() if p.is_file() else None for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        scope['build'](argparse.Namespace(output=str(out),source_root=str(tmp_path),target='temporal',device='cpu'))
    assert before=={p.name:p.read_bytes() if p.is_file() else None for p in tmp_path.iterdir()}


def test_legacy_output_collision_does_not_start_backend(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from LightGenV2.tasks.t06_video_quality_assessment import build_lab_package as wrapper
    checkpoint=tmp_path/'fixture.pt';checkpoint.write_bytes(b'not a model')
    output=tmp_path/'existing.zip';output.write_bytes(b'original package')
    monkeypatch.setattr(wrapper,'load_profile',lambda *_:{'backend':{},'artifacts':{}})
    monkeypatch.setattr(wrapper.subprocess,'run',lambda *_args,**_kw: (_ for _ in ()).throw(AssertionError('backend launched')))
    monkeypatch.setattr(sys,'argv',['builder','--checkpoint',str(checkpoint),'--output',str(output)])
    with pytest.raises(FileExistsError):wrapper.main()
    assert output.read_bytes()==b'original package'
