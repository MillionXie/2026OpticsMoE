"""Exercise the real process bridge without importing Torch or vendor SDKs."""
import ast
from pathlib import Path
import pytest


@pytest.fixture
def owner_class(tmp_path):
    # Keep these subprocess transport tests usable on CPU-only packaging hosts.
    source=Path(__file__).parents[1]/'lab_control.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if
           (isinstance(n,ast.ClassDef) and n.name=='ProcessPhaseOwner') or
           (isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_PHASE_WORKER' for x in n.targets))]
    namespace={}
    exec('import sys,json,subprocess,queue,threading\nfrom types import SimpleNamespace',namespace)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    (tmp_path/'phase_owner.py').write_text('''from types import SimpleNamespace
class PhaseOwner:
    def __init__(self,c,flat,lens):self.c=c
    def __enter__(self):
        self.info={'mock':True};self.display=SimpleNamespace(audit=[]);return self
    def show(self,path,expected=None):
        if path=='bad':raise RuntimeError('mock write rejected')
        return {'phase_file':path,'phase_sha256':expected}
    def __exit__(self,*args):pass
''',encoding='utf-8')
    return namespace['ProcessPhaseOwner'],tmp_path


def test_phase_process_roundtrip_and_shutdown(owner_class):
    cls,folder=owner_class
    with cls({},'flat','lens',folder) as owner:
        assert owner.info['sdk_process_isolated']
        assert owner.show('mask.bmp','abc')['phase_sha256']=='abc'
    assert owner.process.poll()==0


def test_phase_process_failure_is_not_ack(owner_class):
    cls,folder=owner_class
    with cls({},'flat','lens',folder) as owner:
        with pytest.raises(RuntimeError,match='mock write rejected'):
            owner.show('bad')
    assert owner.process.poll()!=0


def test_phase_wait_drains_before_ack_and_after():
    import threading,time
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    source=Path(__file__).parents[1]/'lab_control.py'
    node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
              if isinstance(n,ast.FunctionDef) and n.name=='show_while_draining')
    ns=dict(ThreadPoolExecutor=ThreadPoolExecutor,time=time,sha=lambda p:'digest')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    drained=threading.Event();counter=[]
    def grab():counter.append(1);drained.set();time.sleep(.005)
    def show(path,digest):
        assert drained.wait(timeout=1),'Camera was not drained during phase wait'
        return {'test':True}
    result=ns['show_while_draining'](SimpleNamespace(show=show),SimpleNamespace(camera=SimpleNamespace(grab=grab)),'mask')
    assert result['camera_drain_during_phase_wait']['frames']==len(counter)>1
