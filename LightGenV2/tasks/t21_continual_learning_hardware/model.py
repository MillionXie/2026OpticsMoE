"""Native8-um T21 front with explicit display quantization and CCD boundaries."""
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t20_reconfigurable_rgb_sar.hardware_aware import NativeOEOOptics


class HardwareOptics(NativeOEOOptics):
    STAGES=('router','first','final')

    def __init__(self,architecture):
        super().__init__(architecture,expert_grid=4,expert_size=230,gap=32,
                         border=20,routing_temperature=1.25,readout='linear')
        self.register_buffer('display_gains',torch.tensor([32.,128.,384.]))

    @staticmethod
    def phase_fraction(raw):
        value=raw.sigmoid()
        quant=torch.floor(value*256).clamp(0,255)/256
        return value+(quant-value).detach()

    def bounded_amplitude(self,raw,stage):
        value=(raw*self.display_gains[self.STAGES.index(stage)]).clamp(0,1)
        quant=torch.round(value*255)/255
        return value+(quant-value).detach()

    def route_from_ccd(self,ccd):
        b=self.border
        energy=self.detect(F.pad(ccd,(b,)*4),self.router_centers,self.router_side)
        indices=self.active_indices[:int(self.active_count)]
        scores=(energy.index_select(1,indices)+1e-12).pow(1/self.temperature)
        weights=scores/scores.sum(1,keepdim=True).clamp_min(1e-20)
        return torch.zeros_like(energy).scatter(1,indices[None].expand(len(ccd),-1),weights)

    def raw_payload(self,amplitude,stage,prefix):
        if stage not in self.STAGES or amplitude.shape[-2:]!=(224,224):
            raise ValueError('Unknown stage or changed original encoding')
        n=len(amplitude)
        if stage=='router':
            if self.architecture!='moe':raise ValueError('D2NN has no router')
            size=self.expert_size;s=(self.active-size)//2
            amp=self.normalize(F.interpolate(amplitude[:,None],(size,size),mode='bilinear',align_corners=False)[:,0])
            phase=self.phase_fraction(self.router_phase)
            return F.pad(amp,(s,)*4),F.pad(phase,(s,)*4),None
        if stage=='first':
            if self.architecture=='d2nn':
                amp=self.normalize(F.interpolate(amplitude[:,None],(self.active,self.active),mode='bilinear',align_corners=False)[:,0])
                return amp,self.phase_fraction(self.first_phase),None
            q=self.route_from_ccd(prefix['router'])
            amp=self.normalize(F.interpolate(amplitude[:,None],(self.expert_size,self.expert_size),mode='bilinear',align_corners=False)[:,0])
            raw=amp.new_zeros((n,self.active,self.active))
            phase=amp.new_zeros((self.active,self.active))
            for i in self.active_indices[:int(self.active_count)].tolist():
                y,x=self.slots[i];y-=self.border;x-=self.border;s=self.expert_size
                raw[:,y:y+s,x:x+s]=amp*q[:,i,None,None].sqrt()
                phase[y:y+s,x:x+s]=self.phase_fraction(self.first_phase[i])
            return raw,phase,q
        ccd=prefix['first']
        intensity=ccd/ccd.mean((-2,-1),keepdim=True).clamp_min(1e-20)
        return self.normalize(F.softsign(intensity)),self.phase_fraction(self.global_phase),None

    def payload(self,amplitude,stage,prefix):
        raw,phase,q=self.raw_payload(amplitude,stage,prefix)
        return self.bounded_amplitude(raw,stage),phase,q

    def propagate_payload(self,amplitude,phase):
        b=self.border
        field=F.pad(amplitude*torch.exp(2j*torch.pi*phase),(b,)*4)
        intensity=self.propagator(field).abs().square()
        return intensity[:,b:-b,b:-b]

    def features_from_ccd(self,ccd):
        features=F.adaptive_avg_pool2d(ccd[:,None],(28,28))[:,0].flatten(1)
        return features/features.mean(1,keepdim=True).clamp_min(1e-20)

    def forward(self,amplitude,*,measured=None,return_debug=False):
        measured={} if measured is None else measured
        order=self.STAGES if self.architecture=='moe' else self.STAGES[1:]
        if set(measured)-set(order):raise ValueError('Unknown measurement boundary')
        prefix={};route=None;stats={}
        for stage in order:
            if stage in measured:
                ccd=measured[stage]
                if ccd.shape!=(len(amplitude),self.active,self.active) or not torch.isfinite(ccd).all() or (ccd<0).any():
                    raise ValueError('Invalid admitted camera frame')
            else:
                raw,phase,q=self.raw_payload(amplitude,stage,prefix)
                bounded=self.bounded_amplitude(raw,stage)
                ccd=self.propagate_payload(bounded,phase)
                route=q if q is not None else route
                if return_debug:
                    stats[stage]=dict(clipped_fraction=float((raw*self.display_gains[self.STAGES.index(stage)]>1).float().mean().detach()),nonzero_fraction=float((bounded>0).float().mean().detach()))
            prefix[stage]=ccd
        features=self.features_from_ccd(prefix['final'])
        logits=self.shared_head(features)
        result=dict(logits=logits,probabilities=logits.softmax(1),ccd_features=features,route_power=route)
        if return_debug:result.update(ccd=prefix,display_stats=stats)
        return result

    @torch.no_grad()
    def calibrate_gains(self,training_amplitudes):
        """Fixed TRAIN-only gains; never per-image normalization or TEST fitting."""
        prefix={};report={}
        order=self.STAGES if self.architecture=='moe' else self.STAGES[1:]
        for stage in order:
            raw,phase,_=self.raw_payload(training_amplitudes,stage,prefix)
            maximum=float(raw.max())
            if maximum<=0:raise ValueError('Zero TRAIN amplitude')
            gain=min(4096.,max(1.,.8/maximum))
            self.display_gains[self.STAGES.index(stage)]=gain
            prefix[stage]=self.propagate_payload(self.bounded_amplitude(raw,stage),phase)
            report[stage]=dict(gain=gain,TRAIN_maximum=maximum,target_maximum=.8)
        return report


def build_model(architecture):
    # 4*230 + 3*32 = 1016 native pixels. One optical expert array,
    # one global phase, plus an independent router for MoE only.
    return HardwareOptics(architecture)
