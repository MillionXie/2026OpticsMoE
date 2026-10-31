"""Training-only modality-marginal optical load regularization; no GT routing."""
import torch


def modality_mean_power_KL(dense_power, domains):
    """KL(uniform || mean dense power), separately within each present modality.

    This encourages marginal usage, not four-way uniform routing per image.
    Actual hard Top2 usage must still be audited separately.
    """
    if dense_power.ndim!=2 or dense_power.shape[1]!=4 or domains.shape!=dense_power.shape[:1]:
        raise ValueError('Four ports and one acquisition-domain ID per sample required')
    if not bool(((domains==0)|(domains==1)).all()) or not len(domains):raise ValueError('RGB/SAR domains required')
    if not torch.isfinite(dense_power).all() or bool((dense_power<0).any()):raise ValueError('Invalid power')
    means=torch.stack([dense_power[domains==d].mean(0) for d in (0,1) if bool((domains==d).any())])
    return (-.25*(means.clamp_min(1e-20)*4).log()).sum(1).mean()
