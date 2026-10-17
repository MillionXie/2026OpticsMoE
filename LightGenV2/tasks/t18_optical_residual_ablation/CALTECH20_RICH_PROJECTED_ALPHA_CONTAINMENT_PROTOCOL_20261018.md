# Representable physical-alpha lower bound, caught BEFORE GPU launch

Original328 projected-coordinate CPU proof passed its stated algebra and other
contracts, but an additional prelaunch float32 check found nominal.4001 becomes
.4000999927520752 in physical float32, below the unchanged strict .4001 audit
guard. Original328 has NO GPU dispatch claim/out or trained model; its source
and CPU proof remain intact, not falsely called a training failure or fully
qualified bound proof. Do not launch its uncontained projection entrypoint.

New four leaves project into[nextafter(float32(.4001),+infinity),.95], i.e.
parameter minimum.4001000225543976. This TIGHTENS physical containment without
relaxing the original strict numerical bound and still satisfies human alpha>.4.
Same forward function/learned count/parameter names, alpha.45→.65 course10E then
direct learning, same augmentation/objective/rates/fresh50/resource/selection.
Only optimizer representable lower bound changes, no new inference network.

Independent new CPU actually forces a model below/above both bounds, runs real
model.project and physical alphas() and checks strict .4001<a<.95, verifies
analytic projected Adam and inward recovery using THIS actual minimum, exact
parent effective coefficients and full TRAIN/auditor global symbols plus actual
loader/augmentation/teacher/shape/seed/resource contract. Realmodel CPUforward/
optimizer/TEST0. Full actualGPU numeric proof still required separately.

Independent final Adam reconstruction uses the identical representable minimum,
then checks EMA, step counts, curriculum, all predictions and best selection.
Extra curriculum checks are hashed into audit.json before the external audit
process returns0; completed SHA is created only afterward by the controller.
No old failed audit or original328 entry replay. All source/PT/data/claims kept;
source only Git. This is an engineering preflight containment correction under
the human overnight authorization, not an extra strategy or extra TEST budget.
