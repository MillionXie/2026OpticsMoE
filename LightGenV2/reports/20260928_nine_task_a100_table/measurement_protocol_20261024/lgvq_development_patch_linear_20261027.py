"""Exact non-overlapping one-patch Conv3d as biased matrix multiplication.

Only the frozen Qwen3 patch class (one complete kernel per input patch) is
eligible. Original learned weight, bias, BF16 casts and row order remain. This
is an implementation candidate, not a claim of equal floating-point rounding:
the original 2048-score same-card byte gate must still pass after execution.
"""
import ast
import hashlib
import inspect
import textwrap
from types import MethodType
from lgvq_development_conv_layout_20261027 import PINNED_CLASS_SHA256,PINNED_KERNEL


def linear_AST(source):
    tree=ast.parse(textwrap.dedent(source));count=0
    signature=ast.dump(ast.parse('self.proj(hidden_states.to(dtype=target_dtype))').body[0].value)
    replacement=ast.parse('torch.nn.functional.linear('
        'hidden_states.to(dtype=target_dtype).reshape(-1, self.in_channels*self.temporal_patch_size*self.patch_size*self.patch_size),'
        'self.proj.weight.view(self.embed_dim,-1),self.proj.bias)').body[0].value
    class OnePatch(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            if ast.dump(node)==signature:
                count+=1
                return replacement
            return self.generic_visit(node)
    tree=OnePatch().visit(tree)
    if count!=1:raise ValueError('Exactly one original biased patch-convolution call required')
    return ast.fix_missing_locations(tree)


def install(front):
    import torch
    patch=front.visual.patch_embed;conv=patch.proj
    digest=hashlib.sha256(inspect.getsource(type(patch)).encode()).hexdigest()
    if (digest!=PINNED_CLASS_SHA256 or not isinstance(conv,torch.nn.Conv3d)
            or tuple(conv.kernel_size)!=PINNED_KERNEL or tuple(conv.stride)!=PINNED_KERNEL
            or tuple(conv.padding)!=(0,0,0) or tuple(conv.dilation)!=(1,1,1) or conv.groups!=1
            or conv.weight.dtype!=torch.bfloat16 or conv.bias is None or conv.bias.dtype!=torch.bfloat16
            or not conv.weight.is_contiguous() or tuple(conv.weight.shape)!=(1024,3,2,16,16)):
        raise ValueError('Frozen contiguous biased BF16 one-patch Qwen3 Conv3d required')
    original=type(patch).forward;ns=dict(original.__globals__)
    exec(compile(linear_AST(inspect.getsource(original)),
        inspect.getsourcefile(original)+'::development_one_patch_linear','exec'),ns)
    patch.forward=MethodType(ns['forward'],patch)
    return dict(profile='one_complete_nonoverlapping_patch_biased_linear',class_source_sha256=digest,
        actual_original_kernel=list(conv.kernel_size),actual_original_stride=list(conv.stride),
        original_weight_shape=list(conv.weight.shape),weight_dtype=str(conv.weight.dtype),bias_dtype=str(conv.bias.dtype),
        original_parameters_retained=True,weight_values_or_precision_changed=False,
        one_output_per_complete_patch=True,input_row_order_unchanged=True,
        flatten_and_linear_inside_actual_forward=True,extra_forward_calls=0)
