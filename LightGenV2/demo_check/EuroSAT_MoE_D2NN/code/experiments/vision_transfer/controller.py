"""Measured automatic routing controls training guidance, never inference."""
from dataclasses import asdict, dataclass
import math

from .layout import A_EXPERTS, B_EXPERTS
EXPERTS = {'A': A_EXPERTS, 'B': B_EXPERTS}


def routing_health(routes, gradients, task, policy, require_gradients=True):
    reasons = []
    domains = [('clean','A')] if task == 'A' else [('clean','A'),('corrupted','B')]
    for modality in ('vision',):
        for domain, group in domains:
            key = modality + '_' + domain
            info = routes.get(key)
            if not info:
                reasons.append(key + ': missing automatic observations')
                continue
            if info.get('samples',0) < policy['probe']['minimum_samples_per_domain']:
                reasons.append(key + ': insufficient automatic samples')
            ids = EXPERTS[group]
            limits = policy['probe'][group]
            selection, power = info['selection_share'], info['power_share']
            if len(selection)!=4 or len(power)!=4 or not all(math.isfinite(v) for v in selection+power):
                reasons.append(key + ': invalid observations')
                continue
            if sum(power[i] for i in ids) < limits['minimum_group_power']:
                reasons.append(key + ': insufficient target group power')
            for expert in ids:
                if power[expert] < limits['minimum_expert_power']:
                    reasons.append(f'{key}: E{expert} insufficient power')
                # Frozen A phases are not expected to have gradients in B.
                if require_gradients and group == task:
                    candidates = [value for name,value in gradients.items()
                                  if name.startswith(modality+'_optical.') and f'.experts.{expert}.' in name]
                    if not candidates or not all(math.isfinite(v) for v in candidates) or min(candidates) <= policy['probe']['minimum_ce_gradient_l2']:
                        reasons.append(f'{key}: E{expert} no measured target-domain CE phase gradient')
    return dict(ready=not reasons, reasons=reasons)


def validation_routes(clean, corrupted=None):
    result = {}
    for modality, info in clean.get('routes',{}).items():
        result[modality+'_clean'] = dict(samples=clean['samples'], **info)
    if corrupted:
        for modality in ('vision',):
            rows = [row for row in corrupted['conditions'] if modality in row.get('routes',{})]
            total = sum(row['samples'] for row in rows)
            if total:
                result[modality+'_corrupted'] = dict(samples=total, **{
                    key:[sum(row['routes'][modality][key][i]*row['samples'] for row in rows)/total for i in range(4)]
                    for key in ('selection_share','power_share')})
    return result


@dataclass
class GuidanceController:
    fraction: float = 1.0
    healthy_streak: int = 0
    unhealthy_streak: int = 0
    automatic_epochs: int = 0
    recoveries: int = 0

    def state_dict(self):
        return asdict(self)

    def update(self, epoch, task, ready, policy):
        if self.fraction!=0:raise RuntimeError('Dense routing never enables task guidance')
        self.automatic_epochs=self.automatic_epochs+1 if ready else 0
        self.healthy_streak=self.healthy_streak+1 if ready else 0
        self.unhealthy_streak=self.unhealthy_streak+1 if not ready else 0
        failure=None
        if epoch>=policy['guidance']['guidance_deadline_epoch'][task] and self.unhealthy_streak>=3:
            failure='dense_target_power_or_gradient_not_ready'
        eligible=ready and self.automatic_epochs>=policy['guidance']['automatic_epochs_before_selection']
        return dict(fraction_used=0.,fraction_next=0.,event='dense_automatic',failure=failure,
                    selection_eligible=eligible,state=self.state_dict())
