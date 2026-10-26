"""Optional TRAIN-only semantic and adversarial losses, absent from inference.

Inspired by the loss families used in pix2pix-Turbo, not its unchanged trainer.
Object crops use TRAIN supervision regions; no mask is passed to the generator.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


def region_boxes(regions,padding=.10):
    boxes=[]
    for mask in regions[:,0]:
        height,width=mask.shape
        y,x=torch.where(mask>.05)
        if x.numel()==0:
            boxes.append([0.,0.,1.,1.]);continue
        x0,x1=float(x.min()),float(x.max());y0,y1=float(y.min()),float(y.max())
        side=min(max(x1-x0,y1-y0,1)*(1+2*padding),width-1,height-1)
        cx=min(max((x0+x1)*.5,side*.5),width-1-side*.5)
        cy=min(max((y0+y1)*.5,side*.5),height-1-side*.5)
        boxes.append([(cx-side*.5)/(width-1),(cy-side*.5)/(height-1),
                      (cx+side*.5)/(width-1),(cy+side*.5)/(height-1)])
    return torch.tensor(boxes,device=regions.device,dtype=torch.float32)


def crop_regions(images,boxes,size=128):
    """Differentiable fixed-label crops without torchvision compiled operators."""
    with torch.autocast(images.device.type,enabled=False):
        axis=torch.linspace(0,1,size,device=images.device)
        x=boxes[:,0,None]+(boxes[:,2]-boxes[:,0])[:,None]*axis
        y=boxes[:,1,None]+(boxes[:,3]-boxes[:,1])[:,None]*axis
        grid=torch.stack([x[:,None,:].expand(-1,size,-1),
                          y[:,:,None].expand(-1,-1,size)],-1)*2-1
        return F.grid_sample(images.float(),grid,mode='bilinear',padding_mode='border',align_corners=True)


def discriminator_views(images,boxes):
    return torch.cat([F.interpolate(images.float(),size=(128,128),mode='bilinear',align_corners=False),
                      crop_regions(images,boxes,128)])


class PatchDiscriminator(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        blocks=[];incoming=3
        for outgoing in (width,width*2,width*4,width*8):
            blocks.extend([spectral_norm(nn.Conv2d(incoming,outgoing,4,2,1)),nn.LeakyReLU(.2)])
            incoming=outgoing
        blocks.append(spectral_norm(nn.Conv2d(incoming,1,3,1,1)))
        self.net=nn.Sequential(*blocks)

    def forward(self,image):return self.net(image.float())


def hinge_discriminator_loss(real_logits,fake_logits):
    return .5*(F.relu(1-real_logits).mean()+F.relu(1+fake_logits).mean())


class FrozenCLIPSemantic(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.clip=model.float().eval().requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.48145466,.4578275,.40821073])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.26862954,.26130258,.27577711])[None,:,None,None])

    def image_features(self,images):
        with torch.autocast(images.device.type,enabled=False):
            rgb=F.interpolate(images.float().clamp(-1,1)*.5+.5,size=(224,224),mode='bicubic',align_corners=False)
            encoded=self.clip.encode_image((rgb-self.mean)/self.std).float()
            return F.normalize(encoded,dim=-1)

    def forward(self,images,text_features):
        # Positive PAIRS only; averaging a BxB similarity matrix would teach
        # every generated image to match unrelated captions in the same batch.
        visual=self.image_features(images)
        text=F.normalize(text_features.float(),dim=-1)
        return (1-(visual*text).sum(-1)).mean()
