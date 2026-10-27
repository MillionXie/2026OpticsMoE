"""One explicit frontend-priority scheduling candidate on native host dispatch."""
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
import run_lgvq_a100_cpp_dispatch_20261027 as cpp

ENTRY=Path(__file__).resolve();CPP_ENTRY=cpp.ENTRY
SOURCES=cpp.source_map;WORKER_AST=cpp.dispatch.worker_AST;INSTALL=cpp.core.install


def source_map():
    result=SOURCES()
    result[CPP_ENTRY.relative_to(cpp.core.original.REPO).as_posix()]=hashlib.sha256(CPP_ENTRY.read_bytes()).hexdigest()
    return result


def worker_AST(source):
    tree=WORKER_AST(source);changed=0
    signature=ast.dump(ast.parse('fs,bs=torch.cuda.Stream(),torch.cuda.Stream()').body[0])
    class Priority(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal changed
            if ast.dump(node)==signature:
                changed+=1
                node.value.elts[0].keywords.append(ast.keyword(arg='priority',value=ast.Constant(-1)))
            return node
        def visit_Expr(self,node):
            if (isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='base.write'
                    and ast.unparse(node.value.args[0])=="output / 'a100_cpp_dispatch_identity.json'"):
                return [node]+ast.parse("base.write(output/'a100_front_priority_identity.json',dict(frontend_priority=fs.priority,body_priority=bs.priority,original_streams_and_dependencies_retained=True,no_ideal_overlap_or_queue_subtraction=True))").body
            return node
    tree=Priority().visit(tree)
    if changed!=1:raise ValueError('Exact original two-stream creation required')
    return ast.fix_missing_locations(tree)


def install(front):
    import torch
    previous=torch.cuda.Stream;created=[]
    def stream(*args,**kwargs):
        kwargs['priority']=-1
        value=previous(*args,**kwargs);created.append(value);return value
    torch.cuda.Stream=stream
    try:info=INSTALL(front,partitions=2)
    finally:torch.cuda.Stream=previous
    if len(created)!=1:raise ValueError('Exactly one unchanged auxiliary frontend stream required')
    info['auxiliary_stream_priorities']=[int(s.priority) for s in created]
    return info


@contextmanager
def configured():
    saved_entry=cpp.ENTRY;saved_sources=cpp.source_map
    saved_install=cpp.core.install;saved_AST=cpp.dispatch.worker_AST
    cpp.ENTRY=ENTRY;cpp.source_map=source_map;cpp.core.install=install;cpp.dispatch.worker_AST=worker_AST
    try:
        with cpp.configured():yield
    finally:
        cpp.ENTRY=saved_entry;cpp.source_map=saved_sources
        cpp.core.install=saved_install;cpp.dispatch.worker_AST=saved_AST


if __name__=='__main__':
    with configured():cpp.core.original.main()
