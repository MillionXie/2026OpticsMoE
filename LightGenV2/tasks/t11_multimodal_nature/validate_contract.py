"""Validate the shared optical input contract on synthetic tensors."""
import argparse,json,torch
from LightGenV2.tasks.t09_multimodal_matching.model import OpticalOEO, normalize_power

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args();d=torch.device(a.device)
 field=torch.rand(4,224,224,device=d); field=field*(1.0/field.square().sum((-2,-1),keepdim=True)).sqrt()
 model=OpticalOEO('moe',17,input_layout='left_right',oeo_activation='centered_leaky_relu').to(d); y=model(field); assert y['probabilities'].shape==(4,2); assert torch.isfinite(y['probabilities']).all(); assert torch.allclose(y['probabilities'].sum(1),torch.ones(4,device=d),atol=1e-5)
 print(json.dumps({'input_shape':list(field.shape),'probability_shape':list(y['probabilities'].shape),'route_mean':y['route_power'].mean(0).tolist(),'status':'pass'}))
if __name__=='__main__':main()
