"""Lazy native Qwen adapter. Importing this module does not load Torch or an SDK."""
from pathlib import Path
from .scene_lab_protocol import sha

F9='f9fbdb7fd36d1c862686616a624e7c18f4e9149cef61b0704bbf97767de6eaec'
F6='6f34f2d720060c9193cda0e19ffb8fb4d157bc03e3fd5e5c1433c60ad884471e'
E662='662ff90f9a46670b28bf86d61629c59e4fb619ed69a62e4ae52ed8527ea76f76'


def load_head(checkpoint,*,expected_sha256):
    """Restore the checkpoint's flag and exact existing state, never default E head."""
    if expected_sha256 not in(F9,E662)or sha(checkpoint)!=expected_sha256:raise ValueError('Wrong frozen head')
    import torch
    from .qwen3_scene_conditioning import QwenSDXLConditions
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    expected_joint=expected_sha256==F9
    if state.get('joint_image_context',False)!=expected_joint:raise ValueError('joint_image_context differs from frozen identity')
    if state['variant']!='qwen' or state['Qwen_input_side']!=256 or state['resolution']!=512:raise ValueError('Wrong native graph')
    head=QwenSDXLConditions(joint_image_context=expected_joint)
    head.load_state_dict(state['head'],strict=True);head.eval().requires_grad_(False)
    return head,state


class NativeRunner:
    """Use an ALREADY constructed frozen F6 Qwen and F9 head, without device setup.

    Caller retains the original frontend loader, LoRA/sampler/generator contract.
    This returns conditions only, not a replacement SDXL renderer or device driver.
    """
    def __init__(self,qwen,processor,head,reference_root,*,frontend_sha256,head_sha256):
        if frontend_sha256!=F6 or head_sha256!=F9 or not head.joint_image_context:
            raise ValueError('Require F6 and the F9 joint-image head')
        if any(p.requires_grad for m in(qwen,head)for p in m.parameters()):raise ValueError('Frozen modules required')
        if any(m.pairs!=1 for m in qwen.model.hybrid.values()):raise ValueError('One E/G pair per modality')
        self.qwen,self.processor,self.head=qwen,processor,head
        self.reference_root=Path(reference_root);self.native_forward_attempts=0;self.completed_forwards=0

    def __call__(self,request,provider):
        import torch
        from PIL import Image
        from .scene_optical_boundary import attach
        path=(self.reference_root/request['reference']).resolve()
        if not path.is_relative_to(self.reference_root.resolve())or sha(path)!=request['reference_sha256']:raise ValueError('Reference identity/path mismatch')
        with Image.open(path)as image:image=image.convert('RGB').copy()
        content=[{'type':'image','image':image},{'type':'text','text':request['prompt']}]
        text=self.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        device=next(self.qwen.parameters()).device
        inputs=self.processor(text=[text],images=[image],return_tensors='pt').to(device)
        if inputs['image_grid_thw'].tolist()!=[[1,16,16]]or not inputs['attention_mask'].all():raise ValueError('Require original256 input without padding')
        native=[];hook=self.qwen.model.visual.register_forward_hook(lambda m,a,out:native.append(out[0]))
        def bridge(stage,amplitude,phase,ideal):
            if ideal is not None:raise ValueError('No ideal CCD allowed')
            value=provider(stage,amplitude.detach().cpu().float().numpy(),phase.detach().cpu().float().numpy(),None)
            return torch.from_numpy(value.copy()).to(amplitude.device)
        restore=attach(self.qwen.model.hybrid,bridge,simulation_diagnostic=False)
        self.native_forward_attempts+=1
        try:
            with torch.inference_mode():
                hidden=self.qwen.model(**inputs,use_cache=False,return_dict=True).last_hidden_state
                if len(native)!=1 or native[0].shape!=(64,2048):raise ValueError('Native visual hook differs')
                # Exactly the original renderer's bf16-Qwen -> fp32-head cast.
                result=self.head(hidden.float(),native[0][None].float())
        finally:restore();hook.remove()
        self.completed_forwards+=1
        return dict(conditions={k:v.detach().cpu()for k,v in result.items()},input_ids=inputs['input_ids'].cpu(),
                    Qwen_forwards=1,SDK_used=False,all_boundaries_measured=True)
