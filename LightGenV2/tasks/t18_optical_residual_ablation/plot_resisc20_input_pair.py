"""Audited complete input100 pair and separately labeled stopped-control prefix."""
import argparse,hashlib,inspect,json,os,subprocess,textwrap
from pathlib import Path
import numpy as np
import resisc20 as s
import audit_resisc20_input_pair as audit
import plot_saved_validation_studies as render

KIND='resisc_input_pair'
SPEC=dict(render.STUDIES['resisc_input'],keys=audit.PAIR,labels=('Input100 rho0.3','Input100 rho0'),parameters=(908552,908552),title='RESISC fixed20: complete input100 rho pair; human-stopped Input50 shown separately')
RENDER_SHA='bbce2491d452389667339c4fef1ad53c8bc07b941a7dce7672845281c9489ebb'

def guarded(kind,base):
    assert kind==KIND and base.name==audit.RUN
    proof=s.read(base/audit.OUT/'audit.json');dispatch=s.read(base/'cpu_remaining_pair_audit_dispatch_completed_20261014.json')
    assert proof['passed'] and dispatch['passed'] and dispatch['audit_sha256']==s.sha(base/audit.OUT/'audit.json')
    assert proof['model_forward_calls']==proof['test_arrays_read']==proof['test_evaluations']==0 and not (base/audit.FAIL).exists()
    assert proof['audit_source_sha256']==dispatch['audit_tool_source_sha256']==s.sha(Path(audit.__file__))
    info=audit.barrier(base);assert info['ready'] and proof['sources']==info['meta']['sources'] and proof['training_source_commit']==SPEC['commit']
    assert proof['stop_claim_sha256']==s.sha(base/'human_stop_a100_claim_20261014.json') and proof['stop_receipt_sha256']==s.sha(base/'human_stop_a100_receipt_20261014.json')
    assert set(proof['fits'])==set(audit.PAIR) and len(proof['sources'])==151
    m=s.manifest(Path(audit.study.config(audit.PAIR[0])['data']))
    data={}
    for key in audit.PAIR:
        fit=s.read(base/key/'fit_result.json');hist=s.read(base/key/'history.json')
        assert fit==proof['fits'][key] and fit['config']==audit.expected_config(key,m) and fit['sources']==proof['sources']
        assert fit['epochs_completed']==len(hist)<=120 and fit['config']['maximum_seconds']==7200 and fit['config']['microbatch']==8
        assert fit['metrics']['train']['n']==11198 and fit['metrics']['val']['n']==1399 and fit['test_evaluations']==0 and not fit['test_used_for_selection']
        assert s.sha(base/key/'history.json')==fit['history_sha256'] and s.sha(base/key/'best_checkpoint.pt')==fit['checkpoint_sha256']
        assert min(hist,key=lambda h:h['val']['balanced_nll'])['epoch']==fit['selected_epoch']
        assert proof['gates'][key]==s.read(base/key/'validation_gate.json') and not (base/key/'failure.json').exists()
        assert proof['scheduler_budget_checks'][key]['passed'];data[key]=(fit,hist)
    partial=proof['human_stopped_partial'];assert partial['passed'] and partial['case']==audit.STOP and partial['epochs']==26 and not partial['eligible'] and not partial['complete_budget'] and not partial['selected_train_metrics_available']
    assert partial['validation_predictions_sha256']==s.sha(base/audit.STOP/'val_predictions.csv')
    assert len({f['initial_parameter_sha256'] for f,_ in data.values()}|{partial['initial_parameter_sha256']})==1
    common=min(f['epochs_completed'] for f,_ in data.values());assert common==proof['common_pair_epochs'] and proof['common_control_prefix_epochs']==min(26,common)
    assert proof['all_pair_eligible']==all(proof['gates'][k]['passed'] for k in audit.PAIR)
    assert len({tuple(f['orders'][:common]) for f,_ in data.values()})==len({tuple(f['transforms'][:common]) for f,_ in data.values()})==1
    return SPEC,proof,data

def renderer_entry():
    assert s.sha(Path(render.__file__))==RENDER_SHA,'Preserve the reviewed generic renderer version'
    source=textwrap.dedent(inspect.getsource(render.plot));token="base/'cpu_report/audit.json'";assert source.count(token)==1
    source=source.replace(token,"base/'cpu_remaining_pair_report/audit.json'")
    ns=dict(render.__dict__);ns.update(guarded=guarded,__file__=__file__)
    exec(compile(source,str(__file__)+':saved_pair_renderer','exec'),ns)
    return ns['plot']

def plot(base,out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and out==base/audit.OUT/'figures' and not out.exists()
    guarded(KIND,base)
    receipt=renderer_entry()(KIND,base,out)
    proof=s.read(base/audit.OUT/'audit.json');prefix=proof['common_control_prefix_epochs'];rows=[]
    fig,axes=render.plt.subplots(1,3,figsize=(15,4.5))
    for key,label,color in ((audit.STOP,'Input50 rho0.3: stopped control','#595959'),(audit.PAIR[0],'Input100 rho0.3: shared prefix','#2475ba'),(audit.PAIR[1],'Input100 rho0: shared prefix','#cc4739')):
        hist=s.read(base/key/'history.json')[:prefix];assert len(hist)==prefix and [h['epoch'] for h in hist]==list(range(1,prefix+1))
        for metric,ax,scale in (('accuracy',axes[0],100),('balanced_accuracy',axes[1],100),('balanced_nll',axes[2],1)):
            ax.plot([h['epoch'] for h in hist],[scale*h['val'][metric] for h in hist],color=color,label=label)
        for h in hist:rows.append(dict(case=key,epoch=h['epoch'],scope='shared_measured_prefix_only',human_stopped_control=key==audit.STOP,val_support=h['val']['n'],val_accuracy=h['val']['accuracy'],val_macro_recall=h['val']['balanced_accuracy'],val_balanced_nll=h['val']['balanced_nll']))
    for ax,label in zip(axes,('Validation accuracy (%)','Validation macro recall (%)','Validation balanced NLL')):
        ax.set_xlabel('Common completed epoch');ax.set_ylabel(label);ax.grid(alpha=.18);ax.set_xlim(.5,prefix+.5)
    axes[0].set_ylim(0,100);axes[1].set_ylim(0,100);axes[2].set_ylim(bottom=0);axes[0].legend(fontsize=7)
    fig.suptitle(f'First {prefix} measured epochs only: Input50 control stopped at26\nNo full-budget input50-vs100 conclusion, endpoint selection or test',fontsize=11);fig.tight_layout(rect=(0,0,1,.86))
    for ext in ('png','svg','pdf'):fig.savefig(out/('common_control_prefix.'+ext),dpi=180,bbox_inches='tight')
    svg=out/'common_control_prefix.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n',encoding='utf-8');render.plt.close(fig)
    render.csvwrite(out/'common_control_prefix.csv',rows)
    receipt.update(study_kind=KIND,source_sha256=s.sha(__file__),generic_renderer_sha256=RENDER_SHA,renderer_adaptation='One audited source substitution for the new saved-science audit path only; all generic metric-rendering math unchanged.',pdf_count=5,partial_control_epochs=26,common_control_prefix_epochs=prefix,full_budget_three_arm_comparison=False,scope='Audited completed input100 pair selected EMA metrics; separately labeled common measured control prefix up to26 only. Human-stopped control has no selected-train endpoint or eligibility. No model/forward/image/test/GPU activity.')
    receipt['files']={p.name:s.sha(p) for p in out.iterdir() if p.is_file() and p.name!='artifact_manifest.json'}
    assert len(list(out.glob('*.pdf')))==5
    (out/'artifact_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('base','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.out.exists(),'Existing/failed figures preserved; no re-entry'
    try:print(json.dumps(plot(a.base,a.out)))
    except BaseException as e:
        if a.out.exists():
            with (a.out/'failure.json').open('x',encoding='utf-8') as f:json.dump(dict(error=repr(e),automatic_retry=False),f)
        raise
