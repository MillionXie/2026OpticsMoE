"""Untimed actual-shape NN MAC ledger, never a timing benchmark.

1 real MAC=2 OP is the adopted *algorithmic dense-MAC convention*, not a GPU
instruction counter. Padding/causal masking is not a sparsity discount. Non-MAC
operators, FFT simulation and unregistered operators are retained separately;
missing arithmetic is not silently declared zero. Physical dense-equivalent
optical propagation must be supplied separately, never inferred from GPU FFT.

No Torch import at module import: pure shape rules are testable without CUDA.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from math import prod,isfinite
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from stage_meter import StageMeter


def shape(value):
    result=tuple(int(x) for x in value)
    if any(x<0 for x in result):raise ValueError('Negative tensor dimension')
    return result


def broadcast(a,b):
    a,b=list(a),list(b);n=max(len(a),len(b));a=[1]*(n-len(a))+a;b=[1]*(n-len(b))+b
    if any(x!=y and x!=1 and y!=1 for x,y in zip(a,b)):
        raise ValueError('Not broadcast-compatible')
    # A0 dimension broadcast against1 is0, not max(0,1)=1.
    return tuple(y if x==1 else x for x,y in zip(a,b))


def matmul_mac(left,right):
    a,b=shape(left),shape(right)
    if not a or not b:raise ValueError('Matmul needs nonscalar tensors')
    if len(a)==1 and len(b)==1:
        if a[0]!=b[0]:raise ValueError('Dot dimensions differ')
        return a[0]
    av,bv=len(a)==1,len(b)==1
    if av:a=(1,*a)
    if bv:b=(*b,1)
    if a[-1]!=b[-2]:raise ValueError('Matmul contraction differs')
    batch=broadcast(a[:-2],b[:-2])
    return prod(batch)*a[-2]*a[-1]*b[-1]


def linear_mac(input_shape,weight_shape):
    x,w=shape(input_shape),shape(weight_shape)
    if not x or len(w)!=2 or x[-1]!=w[1]:raise ValueError('Linear dimensions differ')
    return prod(x[:-1])*w[0]*w[1]


def convolution_mac(input_shape,weight_shape,output_shape,groups=1,transposed=False):
    x,w,y=map(shape,(input_shape,weight_shape,output_shape))
    if (len(x)<3 or len(x)!=len(w) or len(x)!=len(y) or x[0]!=y[0]
            or not isinstance(groups,int) or groups<1):raise ValueError('Convolution dimensions/groups differ')
    if transposed:
        if x[1]!=w[0] or y[1]!=w[1]*groups or x[1]%groups:
            raise ValueError('Transpose-convolution channels differ')
        # Conventional dense ConvTranspose count on its input lattice. Cropped
        # boundary contributions are not discounted; record this convention.
        return x[0]*prod(x[2:])*x[1]*w[1]*prod(w[2:])
    if x[1]!=w[1]*groups or y[1]!=w[0] or y[1]%groups:
        raise ValueError('Convolution channels differ')
    return y[0]*prod(y[2:])*w[0]*prod(w[1:])


def attention_mac(query,key,value):
    q,k,v=map(shape,(query,key,value))
    if min(len(q),len(k),len(v))<3 or q[-1]!=k[-1] or k[-2]!=v[-2]:
        raise ValueError('Rectangular attention dimensions differ')
    # Q,K,V leading dimensions include batch and head; head replication/GQA
    # must already be reflected in the actual executed tensors. Do not guess.
    lead=broadcast(broadcast(q[:-2],k[:-2]),v[:-2])
    return prod(lead)*q[-2]*k[-2]*(q[-1]+v[-1])


def tensor_metadata(value):
    if hasattr(value,'shape') and hasattr(value,'dtype') and hasattr(value,'device'):
        return dict(shape=list(value.shape),dtype=str(value.dtype),device=str(value.device))
    if isinstance(value,(list,tuple)):return [tensor_metadata(v) for v in value]
    if isinstance(value,dict):return {k:tensor_metadata(v) for k,v in value.items()}
    if isinstance(value,float) and not isfinite(value):return dict(nonfinite_scalar=str(value))
    if isinstance(value,(int,float,bool,str)) or value is None:return value
    return {'type':type(value).__name__}


def tensor_shapes(arguments):
    return [tuple(v.shape) for v in arguments if hasattr(v,'shape')]


def classify(operator,args,kwargs,output):
    """Exact shape rules for dense MAC kernels; everything else explicit."""
    root=operator.split('.')[1] if operator.startswith('aten.') else operator
    tensors=[v for v in args if hasattr(v,'shape')]
    if any('complex' in str(v.dtype) for v in tensors):
        return dict(kind='complex_or_optical_simulation_not_real_MAC',mac=None)
    if root in ('mm','bmm','matmul','mv','dot'):
        return dict(kind='dense_matrix_MAC',mac=matmul_mac(args[0].shape,args[1].shape))
    if root in ('addmm','baddbmm','addmv'):
        return dict(kind='dense_matrix_MAC_with_separate_non_MAC_bias',mac=matmul_mac(args[1].shape,args[2].shape))
    if root=='linear':return dict(kind='linear_MAC',mac=linear_mac(args[0].shape,args[1].shape))
    if root in ('convolution','_convolution'):
        groups=int(args[8] if len(args)>8 else kwargs.get('groups',1))
        transposed=bool(args[6] if len(args)>6 else kwargs.get('transposed',False))
        return dict(kind='dense_convolution_MAC',mac=convolution_mac(args[0].shape,args[1].shape,output.shape,groups,transposed),
            convolution_padding='dense convention, no padded-zero or cropped-boundary discount',transposed=transposed)
    if root in ('scaled_dot_product_attention','_scaled_dot_product_flash_attention',
                '_scaled_dot_product_flash_attention_for_cpu','_scaled_dot_product_efficient_attention',
                '_scaled_dot_product_cudnn_attention'):
        return dict(kind='dense_attention_QK_and_AV_MAC',mac=attention_mac(*(a.shape for a in args[:3])),
            attention_mask='full rectangular algorithmic count; causal/padding zeros not subtracted',
            softmax='separate non-MAC, not another matrix product')
    if root.startswith('_fft'):
        return dict(kind='numerical_FFT_simulation_separate_from_physical_optical_OP',mac=None)
    if root in ('embedding','index','index_select','gather','scatter','scatter_','masked_scatter',
                'view','reshape','_unsafe_view','permute','transpose','t','slice','select','squeeze','unsqueeze',
                'expand','as_strided','contiguous','clone','detach','alias','cat','stack','split','split_with_sizes',
                'copy_','_to_copy','to','empty','empty_like','zeros','zeros_like','ones','ones_like','arange',
                'lift_fresh','new_empty','new_zeros','new_ones','scalar_tensor','full','full_like','unbind'):
        return dict(kind='lookup_index_layout_allocation_transfer_no_dense_MAC_rule',mac=None,
            warning='No dense MAC does NOT mean zero time or zero energy')
    if root in ('add','add_','sub','sub_','rsub','mul','mul_','div','div_','true_divide','floor_divide',
                'pow','square','sqrt','rsqrt','abs','absolute','neg','exp','log','log1p','reciprocal',
                'sin','cos','tanh','sigmoid','silu','gelu','relu','relu_','softplus','clamp','clamp_min','clamp_max',
                'sum','mean','var','std','amax','amin','max','min','linalg_vector_norm','norm','native_layer_norm',
                'layer_norm','native_group_norm','group_norm','_softmax','softmax','where','eq','ne','ge','gt',
                'le','lt','isfinite','all','any','logical_not','logical_and','logical_or','bitwise_not',
                'avg_pool2d','adaptive_avg_pool2d','_adaptive_avg_pool2d','max_pool2d_with_indices',
                'upsample_bilinear2d','upsample_nearest2d','upsample_bicubic2d','constant_pad_nd'):
        return dict(kind='non_MAC_arithmetic_or_nonlinearity_not_converted_to_MAC',mac=None)
    return dict(kind='UNREGISTERED_OPERATOR_REVIEW_REQUIRED',mac=None)


class OperationStageMeter(StageMeter):
    """Same existing AST/wrap points, no timing/event creation in OP replay."""
    def __init__(self):
        super().__init__(None);self.current=None;self.stage_calls=Counter()

    @contextmanager
    def scope(self,name,family='neural',charged=True,gpu=True):
        if self.current is not None:
            yield;return
        previous=self.current;self.current=(name,family,bool(charged));self.stage_calls[(self.sample_index,name)]+=1
        try:yield
        finally:self.current=previous


def make_dispatch_counter(meter):
    """Lazy factory; __torch_dispatch__ observes actual, untimed execution."""
    import json
    from torch.utils._python_dispatch import TorchDispatchMode
    class ActualShapeCounter(TorchDispatchMode):
        def __init__(self):
            super().__init__();self.records={}

        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {}
            result=func(*args,**kwargs)
            stage=meter.current or ('UNSCOPED_REVIEW_REQUIRED','unclassified',None)
            name=str(func)
            try:rule=classify(name,args,kwargs,result)
            except ValueError as error:
                # A valid native op can use a shape not covered by this rule
                # (for example ragged attention). Keep its actual output and
                # mark the arithmetic missing, rather than guessing or failing
                # the unchanged model forward after it has already executed.
                rule=dict(kind='UNSUPPORTED_MAC_SHAPE_REVIEW_REQUIRED',mac=None,reason=str(error))
            metadata=dict(sample_index=meter.sample_index,valid_samples=meter.valid_samples,
                stage=stage[0],family=stage[1],charged=stage[2],operator=name,
                inputs=tensor_metadata(args),keywords=tensor_metadata(kwargs),outputs=tensor_metadata(result),**rule)
            key=json.dumps(metadata,sort_keys=True)
            if key not in self.records:self.records[key]=dict(metadata,calls=0)
            self.records[key]['calls']+=1
            return result

        def report(self):
            rows=list(self.records.values())
            actual=sum(r['calls']*r['mac'] for r in rows if r['mac'] is not None)
            serial=sum(r['calls']*r['mac'] for r in rows if r['mac'] is not None and r['charged'] is True)
            unscoped=sum(r['calls']*r['mac'] for r in rows if r['mac'] is not None and r['charged'] is None)
            unsupported=sorted({r['operator'] for r in rows if r['kind'].startswith(('UNREGISTERED_','UNSUPPORTED_'))})
            return dict(status='untimed_actual_dense_MAC_subtotal_nonMAC_and_unsupported_separate',
                actual_executed_real_mac=actual,actual_executed_mac_op=2*actual,
                charged_serial_real_mac=serial,charged_serial_mac_op=2*serial,
                unscoped_review_required_mac=unscoped,unsupported_operators=unsupported,
                complete_total_scalar_OP_claim=False,physical_optical_OP_included=False,
                numeric_FFT_counted_as_physical_light=False,
                convention='1 real MAC=2 OP; dense masked attention and padded convolution, not hardware instructions',
                stage_calls=[dict(sample_index=k[0],stage=k[1],calls=v) for k,v in meter.stage_calls.items()],
                records=rows)
    return ActualShapeCounter()


def torch_cpu_contract():
    """Small actual CPU tensors only; never a formal sample or CUDA warmup."""
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1);torch.manual_seed(42)
    meter=OperationStageMeter();counter=make_dispatch_counter(meter)
    x=torch.randn(2,3);w=torch.randn(4,3);bias=torch.randn(4)
    expected=F.linear(x,w,bias)
    with counter, meter.scope('test.linear'):
        actual=F.linear(x,w,bias)
    if not torch.equal(expected,actual) or counter.report()['actual_executed_real_mac']!=24:
        raise ValueError('Dispatch changes actual Linear values/count')
    q,k,v=(torch.randn(1,2,3,4) for _ in range(3))
    expected=F.scaled_dot_product_attention(q,k,v,is_causal=True)
    previous=counter.report()['actual_executed_real_mac']
    with counter,meter.scope('test.attention'):
        actual=F.scaled_dot_product_attention(q,k,v,is_causal=True)
    if not torch.equal(expected,actual) or counter.report()['actual_executed_real_mac']-previous!=144:
        raise ValueError('Dispatch changes actual attention values/count or double-counts fusion')
    image=torch.randn(1,2,5,5);kernel=torch.randn(3,2,3,3)
    expected=F.conv2d(image,kernel,padding=1)
    previous=counter.report()['actual_executed_real_mac']
    with counter,meter.scope('test.conv'):
        actual=F.conv2d(image,kernel,padding=1)
    if not torch.equal(expected,actual) or counter.report()['actual_executed_real_mac']-previous!=1350:
        raise ValueError('Dispatch changes actual Conv values/count')
    return dict(status='passed',device='CPU only',actual_tensor_bitwise_contracts=['Linear','causal SDPA','Conv2d'],
        torch=torch.__version__,operators=sorted({r['operator'] for r in counter.report()['records']}),
        no_model_loaded=True,no_formal_replay_performed=True,
        **{k:v for k,v in counter.report().items() if k not in ('records','status')})


if __name__=='__main__':
    import json
    print(json.dumps(torch_cpu_contract(),indent=2))
