"""Direct transport must dispatch explicit ledger schemas, including full70 TEST."""
import ast
import inspect
from LightGenV2.tasks.t12_text_to_image import scene_lab_direct_transfer as direct


def test_direct_transport_uses_same_schema_dispatch_as_data_stream():
    tree=ast.parse(inspect.getsource(direct))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert sum(n.func.id=='open_ledger' for n in calls)==1
    assert not any(n.func.id=='Ledger' for n in calls)
