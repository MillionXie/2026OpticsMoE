"""Development-only one-change graph allocator-pool hint.

Only graphs belonging to the SAME real slot and SAME explicit CUDA stream
share a pool. Frontend/body have different pools, as do the two independent
slots. The existing slot fence, full graph replay order and output clone remain.
Pool handles are first created inside the original timed capture, not warmed.
No graph/input/weight/math/event/clock/capture is removed or modified.

Reviewed pinned sources:
https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/docs/source/notes/cuda.rst
https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/cuda/graphs.py
"""
import ast
import inspect


def pool_hint_AST(source):
    import textwrap
    tree = ast.parse(textwrap.dedent(source))
    expected = ast.dump(ast.parse("graph.capture_begin(capture_error_mode='global')").body[0].value)
    count = 0
    class PoolHint(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal count
            if ast.dump(node) == expected:
                count += 1
                node.keywords.insert(0, ast.keyword(arg='pool', value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                        attr='pool_for_current_stream', ctx=ast.Load()), args=[], keywords=[])))
                return node
            return self.generic_visit(node)
    tree = PoolHint().visit(tree)
    if count != 1:raise ValueError('Exactly one pinned capture_begin pool hint required')
    return ast.fix_missing_locations(tree)


def tape_class(original):
    tree = pool_hint_AST(inspect.getsource(original.start))
    namespace = dict(original.start.__globals__)
    exec(compile(tree, inspect.getsourcefile(original)+'::same_slot_stream_pool', 'exec'), namespace)
    class SlotStreamPoolTape(original):
        def __init__(self, torch, stream):
            super().__init__(torch, stream)
            self.development_stream_pools = {}

        def pool_for_current_stream(self):
            key = int(self.stream.cuda_stream)
            if key not in self.development_stream_pools:
                self.development_stream_pools[key] = self.torch.cuda.graph_pool_handle()
            return self.development_stream_pools[key]

        start = namespace['start']

        def close(self):
            super().close()
            self.segments[-1].update(independent_default_pool=False,
                development_pool_scope='same_real_slot_same_stream_only',
                development_stream_pool_count=len(self.development_stream_pools))
    return SlotStreamPoolTape
