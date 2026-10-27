"""One complete patch: fold temporal/height axes for native Conv2d.

Unlike rejected GEMM candidates, this retains a native convolution and its
separate low-precision bias behavior. The exact frozen input is one full kernel
per patch, so folding T,H preserves every channel/temporal/spatial product and
row order. Backend floating-point equivalence is NOT assumed: the original
2048-score same-card byte gate must still pass. No extra forward or autotune.
"""
import ast
import hashlib
import inspect
import textwrap
from types import MethodType
from lgvq_development_conv_layout_20261027 import PINNED_CLASS_SHA256,PINNED_KERNEL


def conv2d_AST(source):
    tree=ast.parse(textwrap.dedent(source));count=0
    signature=ast.dump(ast.parse('self.proj(hidden_states.to(dtype=target_dtype))').body[0].value)
    replacement=ast.parse('torch.nn.functional.conv2d('
        'hidden_states.to(dtype=target_dtype).reshape(-1,self.in_channels,self.temporal_patch_size*self.patch_size,self.patch_size),'
        'self.proj.weight.view(self.embed_dim,self.in_channels,self.temporal_patch_size*self.patch_size,self.patch_size),'
        'self.proj.bias,stride=(self.temporal_patch_size*self.patch_size,self.patch_size))').body[0].value
    class CompletePatch(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal count
            if ast.dump(node)==signature:
                count+=1
                return replacement
            return self.generic_visit(node)
    tree=CompletePatch().visit(tree)
    if count!=1:raise ValueError('Exactly one original complete patch convolution required')
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
        raise ValueError('Frozen contiguous biased BF16 complete-patch Qwen3 convolution required')
    original=type(patch).forward;ns=dict(original.__globals__)
    exec(compile(conv2d_AST(inspect.getsource(original)),
        inspect.getsourcefile(original)+'::development_temporal_height_fold_conv2d','exec'),ns)
    patch.forward=MethodType(ns['forward'],patch)
    return dict(profile='complete_patch_temporal_height_fold_native_conv2d',class_source_sha256=digest,
        original_kernel=list(conv.kernel_size),original_stride=list(conv.stride),
        folded_kernel=[32,16],folded_stride=[32,16],original_weight_shape=list(conv.weight.shape),
        weight_dtype=str(conv.weight.dtype),bias_dtype=str(conv.bias.dtype),
        native_convolution_retained=True,original_parameter_objects_retained=True,
        weight_values_or_precision_changed=False,one_output_per_complete_patch=True,
        reduction_index_order_unchanged=True,folding_inside_actual_forward=True,
        extra_forward_calls=0,autotune_or_backend_policy_changed=False)
