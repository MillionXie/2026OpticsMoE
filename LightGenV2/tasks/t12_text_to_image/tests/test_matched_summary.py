import math
import torch
from LightGenV2.tasks.t12_text_to_image.export_matched_summary import per_image_metrics,noise_batch,summarize
from LightGenV2.tasks.t12_text_to_image.train_matched_qwen_baseline import CachedPairs


def test_rgb_range_and_per_image_psnr():
    target=torch.full((2,3,16,16),-1.)
    prediction=torch.stack((torch.zeros((3,16,16)),torch.ones((3,16,16))))
    rows=per_image_metrics(prediction,target)
    assert abs(rows[0]['mse_0_1']-.25)<1e-8
    assert abs(rows[0]['psnr_db']-10*math.log10(4))<1e-5
    assert rows[1]['mse_0_1']==1 and rows[1]['psnr_db']==0
    combined=[{'mode':'background',**{'model_'+k:v for k,v in r.items()}} for r in rows]
    # This mean differs from -10log10(mean MSE), intentionally.
    assert abs(summarize(combined,'model')['overall']['psnr_db']-3.010299956)<1e-5


def test_identity_ssim_and_clamp():
    target=torch.zeros((1,3,16,16))
    assert abs(per_image_metrics(target,target)[0]['ssim']-1)<1e-6
    assert per_image_metrics(torch.full_like(target,8),target)==per_image_metrics(torch.ones_like(target),target)


def test_noise_seed_is_index_based_not_batch_based():
    full=noise_batch(3,torch.device('cpu'),17,1042)
    joined=torch.cat([noise_batch(1,torch.device('cpu'),17+i,1042) for i in range(3)])
    assert torch.equal(full,joined)


def test_training_uses_full_qwen_cache_not_old_two_layer_features():
    payload={'reference':torch.zeros(1,4,32,32),'target':torch.ones(1,4,32,32),
             'prompts':['prompt'],'qwen_text':torch.full((1,2048),-999.)}
    row=CachedPairs(payload,{'prompt':torch.full((1,2048),28.)})[0]
    assert row['condition'].shape==(2048,) and bool((row['condition']==28).all())
