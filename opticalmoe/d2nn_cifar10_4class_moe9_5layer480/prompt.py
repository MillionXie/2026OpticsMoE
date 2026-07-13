import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputTopKRouter(nn.Module):
    """Standard input-dependent sparse MoE gate: pooled image -> Linear -> top-k."""

    def __init__(self, num_experts=9, top_k=3, pool_size=10, temperature=1.0):
        super().__init__()
        self.num_experts=int(num_experts);self.top_k=int(top_k);self.pool_size=int(pool_size);self.temperature=float(temperature)
        if not 1<=self.top_k<=self.num_experts:raise ValueError("prompt.top_k must be between 1 and 9.")
        if self.temperature<=0:raise ValueError("prompt.temperature must be positive.")
        self.gate=nn.Linear(self.pool_size*self.pool_size,self.num_experts)
        nn.init.normal_(self.gate.weight,mean=0.0,std=0.01);nn.init.zeros_(self.gate.bias)

    def forward(self,images):
        if images.ndim==3:images=images.unsqueeze(1)
        if images.shape[1]!=1:images=images.mean(1,keepdim=True)
        pooled=F.adaptive_avg_pool2d(images.float(),(self.pool_size,self.pool_size)).flatten(1)
        logits=self.gate(pooled);probabilities=torch.softmax(logits/self.temperature,dim=-1)
        _,indices=torch.topk(probabilities,k=self.top_k,dim=-1)
        selected=torch.zeros_like(probabilities,dtype=torch.bool);selected.scatter_(1,indices,True)
        sparse=probabilities*selected.to(probabilities.dtype);weights=sparse/(sparse.sum(-1,keepdim=True)+1e-8)
        importance=probabilities.mean(0)
        load=selected.float().mean(0)/float(self.top_k)
        balance_loss=float(self.num_experts)*torch.sum(importance*load)
        return {"logits":logits,"probabilities":probabilities,"weights":weights,"selected_mask":selected,"selected_indices":indices,"balance_loss":balance_loss,"importance":importance,"load":load}


class GlobalRouterPrompt(nn.Module):
    """Input-dependent 450x450 complex grating router with no random phase."""

    def __init__(self,layout,wavelength_m,pixel_size_m,propagation_m,focal_length_m,top_k=3,pool_size=10,temperature=1.0):
        super().__init__();layout.validate();self.layout=layout
        self.router_network=InputTopKRouter(9,top_k,pool_size,temperature)
        center=layout.canvas_size//2;axis=(torch.arange(layout.canvas_size,dtype=torch.float64)-center)*float(pixel_size_m);y,x=torch.meshgrid(axis,axis,indexing="ij")
        k=2.0*math.pi/float(wavelength_m);lens=-k/(2.0*float(focal_length_m))*(x.square()+y.square())
        gratings=[];max_abs_frequency=0.0
        for cy,cx in layout.expert_centers:
            shift_y=(cy-center)*float(pixel_size_m);shift_x=(cx-center)*float(pixel_size_m)
            fx=shift_x/(float(wavelength_m)*float(propagation_m));fy=shift_y/(float(wavelength_m)*float(propagation_m));max_abs_frequency=max(max_abs_frequency,abs(fx),abs(fy))
            gratings.append(2.0*math.pi*(fx*x+fy*y))
        nyquist=1.0/(2.0*float(pixel_size_m))
        if max_abs_frequency>nyquist:
            minimum_distance=(layout.expert_pitch*float(pixel_size_m))/(float(wavelength_m)*nyquist)
            raise ValueError(f"Prompt grating requires {max_abs_frequency:.1f} cycles/m, above Nyquist {nyquist:.1f}. Increase prompt_to_expert to at least {minimum_distance:.4f} m.")
        self.max_abs_grating_frequency=max_abs_frequency;self.nyquist_frequency=nyquist
        self.register_buffer("lens_phase",lens.float(),persistent=False);self.register_buffer("grating_phases",torch.stack(gratings).float(),persistent=False);self.register_buffer("active_mask",layout.active_mask().float(),persistent=False);self.register_buffer("phase_biases",torch.zeros(9),persistent=True)

    def transmission(self,weights):
        phase=self.grating_phases.unsqueeze(0)+self.phase_biases.view(1,-1,1,1)
        grating_sum=torch.sum(weights[:,:,None,None]*torch.exp(1j*phase),dim=1)
        lens=torch.exp(1j*self.lens_phase).to(torch.complex64)
        return self.active_mask.to(torch.complex64).unsqueeze(0)*lens.unsqueeze(0)*grating_sum.to(torch.complex64)

    def forward(self,field,images):
        routing=self.router_network(images);transmission=self.transmission(routing["weights"])
        routing["transmission"]=transmission;routing["prompt_amplitude"]=transmission.abs();routing["prompt_phase"]=torch.remainder(torch.angle(transmission),2.0*math.pi)*self.active_mask.unsqueeze(0)
        return field.to(torch.complex64)*transmission,routing
