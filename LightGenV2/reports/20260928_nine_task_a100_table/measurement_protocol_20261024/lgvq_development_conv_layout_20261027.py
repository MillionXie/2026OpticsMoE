"""One GPU-front candidate: original Conv3d in channels-last-3d storage.

Original learned convolution, bias, BF16, geometry and all arithmetic remain.
Only weight storage layout during native loading and input storage layout in
the actual timed forward change. No linear substitution, precision change,
dummy forward, tokenizer/feature/score cache or changed optical work.
The original same-card bitwise gate must still pass; layout equivalence alone
is not accepted as numerical evidence.
"""
import ast
import hashlib
import inspect
import textwrap
from types import MethodType

PINNED_CLASS_SHA256='b80abab0ebe915256e6c0630c70472670e4e6aa99d6a8d1f4c0cfc62cd7977e0'


def layout_AST(source):
    tree=ast.parse(textwrap.dedent(source));count=0
    signature=ast.dump(ast.parse('hidden_states.to(dtype=target_dtype)').body[0].value)
    class InputLayout(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            if ast.dump(node)==signature:
                count+=1;node.keywords.append(ast.keyword(arg='memory_format',
                    value=ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='channels_last_3d',ctx=ast.Load())))
                return node
            return self.generic_visit(node)
    tree=InputLayout().visit(tree)
    if count!=1:raise ValueError('Exactly one original BF16 convolution input cast required')
    return ast.fix_missing_locations(tree)


def install(front):
    import torch
    patch=front.visual.patch_embed;conv=patch.proj
    digest=hashlib.sha256(inspect.getsource(type(patch)).encode()).hexdigest()
    if (digest!=PINNED_CLASS_SHA256 or not isinstance(conv,torch.nn.Conv3d)
            or tuple(conv.kernel_size)!=(2,14,14) or tuple(conv.stride)!=(2,14,14)
            or tuple(conv.padding)!=(0,0,0) or tuple(conv.dilation)!=(1,1,1) or conv.groups!=1
            or conv.weight.dtype!=torch.bfloat16 or conv.bias is None or conv.bias.dtype!=torch.bfloat16):
        raise ValueError('Original learned BF16 biased Qwen3 patch convolution required')
    original=type(patch).forward
    before=list(conv.weight.stride());shape=list(conv.weight.shape)
    conv.to(memory_format=torch.channels_last_3d)
    if not conv.weight.is_contiguous(memory_format=torch.channels_last_3d):raise ValueError('Requested layout not applied')
    ns=dict(original.__globals__);exec(compile(layout_AST(inspect.getsource(original)),
        inspect.getsourcefile(original)+'::development_channels_last_3d','exec'),ns)
    patch.forward=MethodType(ns['forward'],patch)
    return dict(profile='original_Conv3d_channels_last_3d_storage_only',class_source_sha256=digest,
        weight_shape=shape,original_weight_stride=before,new_weight_stride=list(conv.weight.stride()),
        weight_dtype=str(conv.weight.dtype),bias_dtype=str(conv.bias.dtype),
        original_convolution_and_bias_retained=True,weight_values_or_precision_changed=False,
        input_layout_conversion_inside_actual_forward=True,extra_forward_calls=0)
