/** CPU-only presentation contract. Never changes or integrates original samples. */
export const USER_ZOH='user_authorized_ZOH_estimate_not_strict250ms';
const requireValue=(ok,message)=>{if(!ok)throw Error(message);};
export function energyScope(report,audit,{rental=false,derived=false}={}){
  const c=report.cost,p=c.power_integration;
  requireValue(audit.status==='passed','Accepted CPU audit required');
  requireValue(report.power_limit_w===(rental?400:250),'Exact source hardware cap');
  requireValue(Number.isFinite(p.max_power_sample_age_s)&&p.max_power_sample_age_s>=0,'Actual support age required');
  requireValue(Number.isFinite(c.continuous_wall_s)&&c.continuous_wall_s>0&&Number.isFinite(c.gpu_energy_j)&&c.gpu_energy_j>0,'Actual time and board integral required');
  requireValue(Math.abs(c.estimated_system_energy_j-c.gpu_energy_j-c.continuous_wall_s*338.2)<=Math.max(1e-5,c.estimated_system_energy_j*1e-9),'Fixed host proxy exactly once');
  if(rental&&report.model==='deepseek'){
    requireValue(audit.energy_qualified===true&&audit.qualification===USER_ZOH,'Explicit user ZOH qualification required');
    requireValue(p.no_synthetic_or_retimed_samples===true&&typeof p.strict250ms_qualified==='boolean','Actual-sample protocol metadata required');
    requireValue(p.strict250ms_qualified===(p.max_power_sample_age_s<=.25),'Separate strict age flag must match');
    requireValue(Number.isInteger(p.gap_over250ms_count)&&p.gap_over250ms_count>=0&&p.gap_over250ms_support_fraction>=0&&p.gap_over250ms_support_fraction<=1,'Actual gap metadata required');
    if(derived)requireValue(report.status==='complete_separate_CPU_derivation'&&report.original_failed_status_unchanged===true&&report.input_verification_was_late===true,'Preserved failed run and truthful late check required');
    return {qualification:USER_ZOH,strict250ms_qualified:p.strict250ms_qualified,max_power_sample_age_s:p.max_power_sample_age_s,gap_over250ms_count:p.gap_over250ms_count,gap_over250ms_support_fraction:p.gap_over250ms_support_fraction};
  }
  requireValue(!derived&&p.max_power_sample_age_s<=.25,'Original strict250ms qualification required');
  if(rental)requireValue(audit.energy_qualified===true&&audit.power_only===true,'Original Kimi power-only audit required');
  return {qualification:'original_strict250ms',strict250ms_qualified:true,max_power_sample_age_s:p.max_power_sample_age_s,gap_over250ms_count:0,gap_over250ms_support_fraction:0};
}
