"""Load only the four exact original frontend tensors, without full Qwen layers.

The original LiveFront constructor/neural methods are retained. Three storage
operations are adapted: component directory, selected tensor container, and
the full-file size check replaced by an explicit component receipt. All four
original shape/dtype/content fingerprints and method/prompt gates still run.
"""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap

PACK=Path('/root/data/lgvq_20261029/lightgen_front_component_20261029_01')

def constructor_tree(source):
    tree=ast.parse(textwrap.dedent(source));counts={'path':0,'container':0,'size_gate':0}
    class Storage(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            if len(node.targets)==1 and ast.unparse(node.targets[0])=='self.model_path':
                if ast.unparse(node.value)!="Path(self.binding['model_candidate'])":raise ValueError('Original loader path changed')
                node.value=ast.Name(id='_component_directory',ctx=ast.Load());counts['path']+=1
            return node
        def visit_Constant(self,node):
            if node.value=='model.safetensors':counts['container']+=1;return ast.copy_location(ast.Constant('frontend.safetensors'),node)
            return node
        def visit_For(self,node):
            self.generic_visit(node)
            if ast.unparse(node.iter)=="self.binding['original_artifact_checks']":
                counts['size_gate']+=1
                node.body=ast.parse("if item['path']=='model.safetensors':\n _verify_component_origin(item)\n continue").body+node.body
            return node
    tree=Storage().visit(tree)
    if counts!={'path':1,'container':1,'size_gate':1}:raise ValueError('Exact three original storage boundaries required')
    return ast.fix_missing_locations(tree),counts

def install():
    import lgvq_live_frontend as live
    receipt_path=PACK/'frontend_component_receipt.json'
    expected=os.environ['LIGHTGEN_FRONT_COMPONENT_RECEIPT_SHA256']
    if hashlib.sha256(receipt_path.read_bytes()).hexdigest()!=expected:raise ValueError('Trusted component receipt required')
    receipt=json.loads(receipt_path.read_text())
    if (receipt['binding_sha256']!=live.BINDING_SHA or receipt['source_original_weight_bytes']!=4255140312
            or hashlib.sha256((PACK/'frontend.safetensors').read_bytes()).hexdigest()!=receipt['pack_sha256']):
        raise ValueError('Exact original component bytes required')
    def verify(item):
        if item['bytes']!=receipt['source_original_weight_bytes']:raise ValueError('Original full weight identity differs')
    original=live.LiveFront.__init__;source=inspect.getsource(original)
    tree,counts=constructor_tree(source)
    namespace=dict(original.__globals__,_component_directory=PACK,_verify_component_origin=verify)
    exec(compile(tree,'<same_original_frontend_component_storage>','exec'),namespace)
    live.LiveFront.__init__=namespace['__init__']
    return dict(component_receipt_sha256=expected,container_sha256=receipt['pack_sha256'],
                original_constructor_sha256=hashlib.sha256(source.encode()).hexdigest(),
                storage_only_changes=counts,all_original_four_tensor_and_prompt_checks_retained=True,
                full_Qwen_transformer_not_instantiated=True)
