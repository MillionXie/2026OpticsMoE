"""Check committed package source closure without loading models or devices."""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT/'LightGenV2/tasks/t06_video_quality_assessment'
ORIGINAL = '7093ec46082eed2fae127ec5028d3e2e8548b592'


def test_required_phase_and_portable_entry_match_preserved_source():
    for relative in ('lab_phase.py', 'hardware/run_lab.py'):
        path = TASK/relative
        original = subprocess.check_output(['git', 'show', ORIGINAL+':'+path.relative_to(ROOT).as_posix()], cwd=ROOT)
        assert path.read_bytes().replace(b'\r\n', b'\n') == original
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
    current = path.read_text()
    current = current.replace(";p.add_argument('--release-file',type=Path)", '')
    current = current.replace(" if a.release_file and a.release_file.exists():raise ValueError('Release file already exists')\n", '')
    current = current.replace("while not (a.release_file and a.release_file.exists()):", "while True:")
    assert current == original
