"""Training-only two-domain router conflict projection in physical phase coordinates."""
import torch
from .alternating_readout import anchored_classification
from .f_cross_domain import soft_only_source_test_loss


def validate_projection_profile(cfg):
    mode=cfg.get('router_gradient_projection')
    rule=cfg.get('router_step_rule','adam')
    if rule not in ('adam','normalized_angle_sgd'):raise ValueError('Unknown router step rule')
    if mode is None:
        if rule!='adam':raise ValueError('Router angle SGD requires explicit projected joint training')
        return
    if mode!='two_domain_physical' or cfg.get('architecture')!='LG' or cfg.get('main_layers')!=2 or cfg.get('dataset')!='eurosat_original_f' or cfg.get('training_domain') is not None:
        raise ValueError('Router projection requires explicit joint two-layer F LG')
    if cfg.get('phase_optimizer')!='wrapped' or cfg.get('phase_update_mode','mini_batch')!='mini_batch' or cfg.get('training_scope')!='all' or not cfg.get('source_test_distillation') or not cfg.get('alternating_head') or cfg.get('domain_route_aux',0) or cfg.get('semantic_distillation'):
        raise ValueError('Router projection requires joint wrapped minibatch/head-block KD without other auxiliaries')
    if set(cfg.get('teacher',{}).get('domain_cache_runs',{}))!={'RGB','SAR'}:
        raise ValueError('Router projection requires both checked teachers')


def project_raw_pair(a_raw,b_raw,raw,weight_a):
    """Return a raw gradient whose angle-space components omit mutual conflicts."""
    if not 0<=weight_a<=1 or a_raw.shape!=raw.shape or b_raw.shape!=raw.shape:raise ValueError('Invalid gradient shape/weight')
    if not all(torch.isfinite(v).all() for v in (a_raw,b_raw,raw)):raise ValueError('Nonfinite projection inputs')
    fraction=raw.detach().double().sigmoid();jac=(2*torch.pi*fraction*(1-fraction)).clamp_min(1e-20)
    a=a_raw.detach().double()/jac;b=b_raw.detach().double()/jac
    aa=a.square().sum();bb=b.square().sum();dot=(a*b).sum();conflict=bool(dot<0 and aa>0 and bb>0)
    if conflict:
        # Both projections use the original vectors, never an updated operand.
        ap=a-dot/bb*b;bp=b-dot/aa*a;combined=weight_a*ap+(1-weight_a)*bp
        result=(combined*jac).to(a_raw)
    else:
        combined=weight_a*a+(1-weight_a)*b
        result=weight_a*a_raw+(1-weight_a)*b_raw
    record=dict(projected=conflict,cosine_before=float(dot/(aa*bb).sqrt()) if aa>0 and bb>0 else 0.,
        domain_weights=[weight_a,1-weight_a],physical_norms=[float(aa.sqrt()),float(bb.sqrt())],
        projected_direction_dot_original=[float((combined*a).sum()),float((combined*b).sum())],coordinate='physical router phase',task_gradient_evaluations=2)
    return result.detach(),record


def router_projection_for_batch(model,output,domains,labels,teacher_targets,anchor,cfg,soft_only=False):
    if cfg.get('router_gradient_projection') is None:return None,None
    validate_projection_profile(cfg)
    if soft_only and labels is not None:raise ValueError('TEST soft projection must not receive hard labels')
    if domains.ndim!=1 or len(domains)!=len(output['raw_logits']) or not torch.isin(domains,domains.new_tensor([0,1])).all():raise ValueError('Invalid training acquisition domains')
    gradients=[];counts=[]
    for number in (0,1):
        keep=domains==number;count=int(keep.sum())
        if not count:continue
        if soft_only:
            loss=soft_only_source_test_loss(output['raw_logits'][keep],teacher_targets[keep],model.shared_head.weight,anchor,cfg['alternating_head']['anchor_strength'],cfg['teacher']['weight'],cfg['teacher']['temperature'])
        else:
            loss=anchored_classification(output['raw_logits'][keep],labels[keep],model.shared_head.weight,anchor,cfg['alternating_head']['anchor_strength'],cfg['label_smoothing'],teacher_targets[keep],cfg['teacher']['weight'],cfg['teacher']['temperature'])
        gradients.append(torch.autograd.grad(loss,model.router_phase,retain_graph=True)[0]);counts.append(count)
    if len(gradients)==1:return gradients[0].detach(),dict(projected=False,cosine_before=None,task_gradient_evaluations=1,coordinate='physical router phase',single_domain_batch=True)
    return project_raw_pair(*gradients,model.router_phase,counts[0]/sum(counts))
