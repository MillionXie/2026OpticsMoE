from __future__ import annotations

from typing import Any,Sequence

import torch
from torch import nn


class GridClassRegionDetector(nn.Module):
    def __init__(self,field_size:int,class_names:Sequence[str],region_size:int,temperature:float=1.0,rows:int=2,columns:int=5,eps:float=1e-8)->None:
        super().__init__();self.field_size=field_size;self.class_names=list(class_names);self.region_size=region_size;self.temperature=temperature;self.eps=eps
        if rows*columns!=len(class_names):raise ValueError("Detector grid must contain one region per class")
        masks=torch.zeros(len(class_names),field_size,field_size);boxes=[]
        for index,name in enumerate(class_names):
            row=index//columns;column=index%columns;center_x=field_size*(column+1)/(columns+1);center_y=field_size*(row+1)/(rows+1);x0=int(round(center_x-region_size/2));y0=int(round(center_y-region_size/2));x0=max(0,min(x0,field_size-region_size));y0=max(0,min(y0,field_size-region_size));x1=x0+region_size;y1=y0+region_size;masks[index,y0:y1,x0:x1]=1;boxes.append({"class_index":index,"class_name":name,"row":row,"column":column,"x0":x0,"y0":y0,"x1":x1,"y1":y1,"width":region_size,"height":region_size})
        if torch.any(masks.sum(0)>1):raise ValueError("Detector regions overlap")
        self.register_buffer("region_masks",masks,persistent=True);self.boxes=boxes
    def forward(self,intensity:torch.Tensor)->dict[str,torch.Tensor]:
        value=intensity.float().clamp_min(0);energy=torch.einsum("bhw,khw->bk",value,self.region_masks.float());total=value.sum((-2,-1)).clamp_min(self.eps);fractions=energy/total[:,None];detector_fraction=fractions.sum(1).clamp(max=1);distribution=energy/energy.sum(1,keepdim=True).clamp_min(self.eps);logits=torch.log(distribution.clamp_min(self.eps))/self.temperature;return {"region_energy":energy,"region_fractions":fractions,"region_distribution":distribution,"region_logits":logits,"detector_fraction":detector_fraction}
    def specification(self)->dict[str,Any]:return {"layout":"two_by_five","field_size":self.field_size,"region_size":self.region_size,"class_order":self.class_names,"boxes":self.boxes}

