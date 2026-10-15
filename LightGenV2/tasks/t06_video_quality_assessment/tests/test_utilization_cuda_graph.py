import ast
from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import audit_utilization_cuda_graph as graph


def test_bounded_native_graph_contract():
    assert graph.CASES==(("qwen",8),("clip",10),("yolo",3))
    assert graph.REPLAY_GROUP==16 and graph.WARMUP_CALLS==3
    source=Path(graph.__file__).read_text()
    assert 'with torch.cuda.graph(graph)' in source
    assert 'graph.replay()' in source
    assert '"historical_anchor_replaced":False' in source
    assert 'timer.close()' in source
    calls={n.func.attr for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not {'train','backward','step'}.intersection(calls)
    assert 'timeout=min(300,remaining)' in source
