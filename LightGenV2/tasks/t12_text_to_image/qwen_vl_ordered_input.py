"""Native text-first Qwen input with a parameter-free canonical readout order."""
from collections import defaultdict,deque
import torch


ORDER='text_first_canonical_readout'


def canonical_permutation(native,canonical):
    """Move readout states only; Qwen attention/RoPE always use native order.

    Stable occurrences preserve the ordering within repeated image/text tokens.
    Reject any tokenizer change rather than guess a visual/text boundary.
    """
    if native.ndim!=1 or canonical.ndim!=1 or len(native)!=len(canonical):raise ValueError('Equal B1 token sequences required')
    positions=defaultdict(deque)
    for i,v in enumerate(native.tolist()):positions[v].append(i)
    result=[]
    for v in canonical.tolist():
        if not positions[v]:raise ValueError('Native/canonical token multisets differ')
        result.append(positions[v].popleft())
    if any(positions.values()):raise ValueError('Native tokens omitted')
    permutation=torch.tensor(result,dtype=torch.long)
    if not torch.equal(native[permutation],canonical):raise ValueError('Canonical token alignment failed')
    return permutation


def ordered_inputs(processor,image,prompt):
    values=[]
    for first in ('image','text'):
        content=[{'type':'image','image':image},{'type':'text','text':prompt}]
        if first=='text':content.reverse()
        rendered=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        values.append(processor(text=[rendered],images=[image],return_tensors='pt'))
    canonical,native=values
    if not native['attention_mask'].all() or not canonical['attention_mask'].all():raise ValueError('Actual B1 without token padding required')
    if native['image_grid_thw'].tolist()!=[[1,16,16]] or not torch.equal(native['pixel_values'],canonical['pixel_values']):
        raise ValueError('Native image processor/grid changed')
    permutation=canonical_permutation(native['input_ids'][0],canonical['input_ids'][0])
    return native,permutation,canonical
