/** New immutable 36-trial timing workbook. No model imports or GPU work. */
import fs from 'node:fs/promises';
import fss from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {energyScope,USER_ZOH} from './energy_trial_contract_20261030.mjs';

const HERE=path.dirname(fileURLToPath(import.meta.url));
const REPO=path.resolve(HERE,'../../../..');
const runtime=process.env.ARTIFACT_RUNTIME_DIR;
if(!runtime)throw Error('ARTIFACT_RUNTIME_DIR must point to the bundled dependency junction');
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(createRequire(path.join(runtime,'loader.cjs')).resolve('@oai/artifact-tool')).href);
const energyRemeasured=process.argv.includes('--energy-remeasured');
const includeABO=energyRemeasured||process.argv.includes('--include-abo');
const OUT=path.join(HERE,'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b',energyRemeasured?'Lab8_Rental4_energy_LightGen_20261029_01':includeABO?'Lab8_Rental4_ABO_closeout_20261029_01':'Lab8_Rental4_timing_20261029_01');
const BOOK=energyRemeasured?'A100_LGVQ_energy_LightGen_20261029.xlsx':includeABO?'A100_T06_ABO_results_20261029.xlsx':'A100_Lab8_Rental4_full_input_20261029.xlsx';
if(fss.existsSync(path.join(OUT,'lab_delivery_receipt01.json')))throw Error('Preserve the SHA-backed delivered workbook');
const repairDraft=process.argv.includes('--repair-own-draft');
if(fss.existsSync(path.join(OUT,BOOK))){
  const receiptPath=path.join(OUT,'workbook_receipt.json');
  const ownReceipt=repairDraft&&fss.existsSync(receiptPath)?JSON.parse(fss.readFileSync(receiptPath,'utf8')):null;
  assertDraft(ownReceipt&&ownReceipt.file===BOOK&&crypto.createHash('sha256').update(fss.readFileSync(path.join(OUT,BOOK))).digest('hex')===ownReceipt.sha256);
}
function assertDraft(ok){if(!ok)throw Error('Refuse to overwrite delivered or modified workbook');}
const read=p=>JSON.parse(fss.readFileSync(p,'utf8'));
const sha=p=>crypto.createHash('sha256').update(fss.readFileSync(p)).digest('hex');
const rel=p=>path.relative(REPO,p).replaceAll('\\','/');
const assert=(ok,msg)=>{if(!ok)throw Error(msg);};
const close=(a,b)=>assert(Math.abs(a-b)<=Math.max(1e-8,Math.abs(b)*1e-10),`Mismatch ${a} != ${b}`);
const base=path.join(REPO,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation');
const source=path.join(HERE,'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b');
const labFile=path.join(source,'Lab_baselines_fullscope_20261029_01.json');
const lgFile=path.join(source,'LightGenII_resize_normalization_20261029_01/LightGenII_timing_summary.json');
assert(sha(labFile)==='4edd1231fd94329f2f227c936d83c4bb2d05d567c1e5874c51dd6e546f45237e','Pinned lab summary');
assert(sha(lgFile)==='08b4682698b2f67f16435444d35737677d61728c1ada5cc7a564eb9fcfb227ab','Pinned resize-inclusive LG summary');
const lab=read(labFile),lg=read(lgFile);
const rentalRoot=path.join(base,energyRemeasured?'rental_energy_userZOH_20261030_04':'rental_timing_only_20261029_02');
const priorEnergyRoot=path.join(base,'rental_energy_remeasure_20261029_01');
const secondEnergyRoot=path.join(base,'rental_energy_remeasure_20261029_02');
const rootReceipt=path.join(rentalRoot,'whole_root_backup_receipt_complete01.json');
if(energyRemeasured){
  const pin=process.argv.find(x=>x.startsWith('--rental-root-receipt-sha256='))?.split('=')[1];
  assert(pin&&/^[a-f0-9]{64}$/.test(pin)&&sha(rootReceipt)===pin,'Explicit SHA pin for new sealed rental root');
  const state=read(path.join(rentalRoot,'suite_status.json'));
  const expected=['deepseek_cap50_repeat02','deepseek_cap50_repeat03',...['01','02','03'].map(n=>'deepseek_cap100_repeat'+n)];
  assert(state.status==='complete'&&state.mps_released&&JSON.stringify(state.completed)===JSON.stringify(expected),'All five final native points complete and MPS released');
  assert(sha(path.join(priorEnergyRoot,'whole_root_backup_receipt_complete01.json'))==='81c24a2587832c8c1d4abc9e2e5456eb477127612a45126417da796d8737a057','Five original qualified points and failed sixth remain preserved');
  assert(sha(path.join(secondEnergyRoot,'whole_root_backup_receipt_complete01.json'))==='1b9465a7a6a9b72973c47810ef95b6123fa29dd77f6884580827b19a036936ca','Sixth Kimi continuation preserved');
  const audit=read(path.join(rentalRoot,'local_independent_pooled_audit01.json'));
  assert(audit.status==='passed'&&audit.original_root_receipt_sha256===pin&&audit.round_audits.length===12&&new Set(audit.round_audits.map(a=>a.report_sha256)).size===12,'Exact mixed-source twelve-point independent audit');
}else assert(sha(rootReceipt)==='9394417869da90fd83f24348a03b62bcb2d19631b6a2eee975dd06274774bac4','Sealed whole rental root');
const candidates=new Map();
for(const stem of ['qwen_fullscope_20261029_02','baselines_fullscope_20261029_02','baselines_fullscope_20261029_03']){
  const root=path.join(base,stem);
  for(const d of fss.readdirSync(root,{withFileTypes:true})){
    const p=path.join(root,d.name,'report.json');
    if(d.isDirectory()&&fss.existsSync(p))candidates.set(sha(p),p);
  }
}
const trials=[],configs=[];
const display={qwen:'Qwen3-VL-2B',glm:'GLM-4.6V-Flash',clip:'CLIP ViT-B/32',yolo:'YOLO11s',kimi:'Kimi-VL-A3B',deepseek:'DeepSeek-VL2 Full'};
function addTrial(file,energy,rentalNew=false){
  const r=read(file),p=path.dirname(file),c=r.cost;
  assert(r.status==='complete'&&r.measured_videos===512&&r.batch===1&&r.explicit_warmup_forwards===0,'Exact cold B1 512 trial');
  assert(r.visible_multiprocessors===(r.cap===50?54:108),'Actual MPS SM count');
  const parentAudit=jsonAudit(p,'cpu_audit.json');assert(parentAudit.status==='passed','Server CPU audit');
  let auditFile=path.join(p,'local_independent_fullscope_audit01.json');
  if(fss.existsSync(auditFile))assert(read(auditFile).status==='passed','Existing independent audit');
  else if(rentalNew){
    auditFile=path.join(rentalRoot,'local_independent_pooled_audit01.json');
    const audit=read(auditFile);assert(audit.status==='passed'&&audit.round_audits.some(a=>a.status==='passed'&&a.energy_qualified===true&&a.report_sha256===sha(file)),'Exact new independent round energy audit');
  }else{
    assert(r.model==='qwen'&&r.cap===50&&path.basename(path.dirname(p))==='qwen_fullscope_20261029_02','Only sealed Qwen50 uses pooled independent audit');
    auditFile=path.join(path.dirname(p),'local_independent_pooled_audit01.json');
    const audit=read(auditFile);assert(audit.status==='passed'&&audit.round_audits.some(a=>a.status==='passed'&&a.report_sha256===sha(file)),'Exact existing pooled round audit');
  }
  const scope=energy?energyScope(r,parentAudit,{rental:rentalNew}):{};
  if(energy&&rentalNew){
    const accepted=read(path.join(rentalRoot,'local_independent_pooled_audit01.json')).round_audits.find(a=>a.report_sha256===sha(file));
    assert(accepted?.status==='passed'&&accepted.energy_qualified===true,'Exact mixed-source independent acceptance');
    if(r.model==='deepseek')assert(accepted.qualification===USER_ZOH,'Independent ZOH qualification');
  }
  if(!energy){assert(r.energy_qualified===false&&c.energy_qualified===false&&r.power_limit_w===400,'Rental timing-only, not energy');assert(fss.existsSync(path.join(p,'lab_backup_receipt_complete01.json')),'Lab receipt');}
  const key=r.model+'_'+r.cap;
  const t={key,model:r.model,cap:r.cap,hardware:rentalNew||!energy?'A100 80GB / 400W':'A100 40GB / 250W',repeat:r.repeat,videos:512,groups:32,seconds:c.continuous_wall_s,tokens:r.effective_input_tokens512,gpu_j:energy?c.gpu_energy_j:null,system_j:energy?c.estimated_system_energy_j:null,energy,report_sha256:sha(file),source:rel(file),audit_sha256:sha(auditFile),...scope};
  assert(t.seconds>0&&Number.isInteger(t.tokens)&&t.tokens>=0,'Native numerator');trials.push(t);return t;
}
function jsonAudit(p,n){return read(path.join(p,n));}
function addDerivedTrial(){
  const p=path.join(source,'DeepSeek50_repeat01_userZOH_20261030_01'),file=path.join(p,'derived_report01.json');
  assert(sha(file)==='07c9c12eb1f24f45f695c673c5ad4d32783721377d9dbc5feac47ce28a9a382e'&&sha(path.join(p,'late_input_verification01.json'))==='a8404f961d2a07b1d1b05635ecd01d4b80ac83cf8b672b7d0397baaecddc76f9'&&sha(path.join(p,'lab_derived_data_receipt01.json'))==='ac06ed4dc855dc5cb1f7d8f2b801bfd8e93e4bbbe3a30e04b4da45095ff26926','Accepted immutable separate derivation and dual-site data receipt');
  const r=read(file),auditFile=path.join(rentalRoot,'local_independent_pooled_audit01.json');
  const a=read(auditFile).round_audits.find(a=>a.report_sha256===sha(file));
  assert(r.model==='deepseek'&&r.cap===50&&r.repeat===1&&r.batch===1&&r.measured_videos===512&&r.explicit_warmup_forwards===0&&r.visible_multiprocessors===54&&a?.original_failed_status_unchanged===true,'Exact separately derived full DS50 first round');
  const scope=energyScope(r,a,{rental:true,derived:true}),c=r.cost;
  const t={key:'deepseek_50',model:r.model,cap:r.cap,hardware:'A100 80GB / 400W',repeat:1,videos:512,groups:32,seconds:c.continuous_wall_s,tokens:r.effective_input_tokens512,gpu_j:c.gpu_energy_j,system_j:c.estimated_system_energy_j,energy:true,report_sha256:sha(file),source:rel(file),audit_sha256:sha(auditFile),original_raw_source:r.raw_source,input_verification_was_late:true,...scope};
  assert(Number.isInteger(t.tokens)&&t.tokens>0,'Actual derived native numerator');trials.push(t);return t;
}
for(const c of lab.records){
  const a=c.round_report_SHA256.map(pin=>{assert(candidates.has(pin),'Pinned lab report is installed');return addTrial(candidates.get(pin),true);});
  close(a.reduce((s,t)=>s+t.seconds,0)/96,c.time16_s);
  close(a.reduce((s,t)=>s+t.tokens,0),c.native_effective_input_tokens);
  configs.push({key:a[0].key,model:c.model,cap:c.MPS_percent,hardware:a[0].hardware,energy:true,trials:a});
}
for(const model of ['kimi','deepseek'])for(const cap of [50,100]){
  const a=[1,2,3].map(n=>{
    if(energyRemeasured&&model==='deepseek'&&cap===50&&n===1)return addDerivedTrial();
    const prior=energyRemeasured&&model==='kimi'&&(cap===50||n<3);
    return addTrial(path.join(prior?priorEnergyRoot:energyRemeasured&&model==='kimi'?secondEnergyRoot:rentalRoot,`${model}_cap${cap}_repeat0${n}/report.json`),energyRemeasured,energyRemeasured);
  });
  configs.push({key:a[0].key,model,cap,hardware:a[0].hardware,energy:energyRemeasured,trials:a});
}
assert(trials.length===36&&new Set(trials.map(t=>t.report_sha256)).size===36,'Exactly 36 distinct accepted trials');
const aggregate=c=>{const sec=c.trials.reduce((s,t)=>s+t.seconds,0),tok=c.trials.reduce((s,t)=>s+t.tokens,0);return {key:c.key,model:c.model,cap:c.cap,hardware:c.hardware,time16_s:sec/96,native_tokens:tok,native_tokens_per_s:tok?tok/sec:null,energy16_j:c.energy?c.trials.reduce((s,t)=>s+t.system_j,0)/96:null,gpu_mean_w:c.energy?c.trials.reduce((s,t)=>s+t.gpu_j,0)/sec:null,energy_qualified:c.energy,report_sha256:c.trials.map(t=>t.report_sha256)};};
const summary=configs.map(aggregate);
await fs.mkdir(OUT,{recursive:true});
const wb=Workbook.create(),s=wb.worksheets.add('Comparison');
const lgRentalSheet=energyRemeasured?wb.worksheets.add('LightGen rental'):null;
const aboSheet=includeABO?wb.worksheets.add('ABO retrieval'):null;
const raw=wb.worksheets.add('Trials');
let aboResult=null;
let lgRental=null;
if(energyRemeasured){
  const root=path.join(base,'lightgen_rental_20261029_01'),leaf=path.join(root,'lg_repeat01');
  const state=read(path.join(root,'suite_status.json'));
  assert(state.status==='stopped'&&state.mps_released&&state.completed.length===0,'Preserved LightGen numerical-gate failure');
  const reportFile=path.join(leaf,'numeric_audit.json'),numeric=read(reportFile);
  const auditFile=path.join(root,'local_independent_cpu_audit01.json'),audit=read(auditFile);
  assert(audit.status==='diagnostic_only_numeric_gate_failed'&&audit.numeric_audit_sha256===sha(reportFile),'Independent saved LightGen diagnostic replay');
  assert(!numeric.all_formal_scores_bitwise_equal&&numeric.equal_scores===92,'Do not relabel failed numerical gate as passed');
  const wall=audit;
  assert(wall.measured_videos===512&&wall.formal_fields===32&&wall.extra_warm_fields===3&&wall.physical_photonic_hardware_measured===false,'Original LightGen population and scope');
  const e=audit.board_energy_replayed;
  assert(e.energy_j>0&&e.elapsed_s>0&&e.host_proxy_w===0&&e.max_power_sample_age_s<=.25,'Actual software board-only energy, unchanged250ms gate');
  lgRental={...wall,gpu_board_energy_j:e.energy_j,gpu_board_mean_w:e.energy_j/e.elapsed_s,report_sha256:sha(reportFile),source:rel(reportFile),audit_sha256:sha(auditFile),physical_optical_energy_qualified:false};
  const a=lgRentalSheet;a.showGridLines=false;a.tabColor='#526983';
  a.getRange('A1:C24').format={font:{name:'Arial',size:10},verticalAlignment:'center',rowHeight:27};
  a.getRange('A2').values=[['LightGen-II on the same A100 80GB']];a.getRange('A2:C2').format.font={name:'Arial',size:15,bold:true};
  a.getRange('A4:C4').values=[['Measurement','Value','Unit and scope']];
  a.getRange('A5:C14').values=[
    ['Full fresh process','Not recorded','No independent enclosing monotonic stopwatch saved'],
    ['Formal software window',wall.actual_FFT_software_window_s,'s, 512 videos / 32 fields, with profiler overhead'],
    ['Software time / 16 videos',null,'s, formal window divided by32'],
    ['GPU board energy',e.energy_j,'J, formal FFT software window only'],
    ['GPU mean power',null,'W, board integral divided by its elapsed window'],
    ['GPU board energy / 16 videos',null,'J, formal FFT software energy divided by32'],
    ['Numerical qualification','FAILED','Actual software runtime is diagnostic only'],
    ['Videos',512,'one original TEST subset; no training'],
    ['Numerically equal scores',numeric.equal_scores,'of512 bitwise equal to the same-checkpoint reference'],
    ['Energy integration window',e.elapsed_s,'s, original monotonic-clock interval']
  ];
  a.getRange('B7').formulas=[['=B6/32']];a.getRange('B9').formulas=[['=B8/B14']];a.getRange('B10').formulas=[['=B8/32']];
  a.getRange('A15:C15').values=[['Maximum score difference',numeric.maximum_absolute_difference,'absolute difference from original same-checkpoint reference']];
  a.getRange('B5:B11').setNumberFormat('#,##0.000000');a.getRange('B12:B13').setNumberFormat('0');a.getRange('B5:B13').format.horizontalAlignment='right';
  a.getRange('A4:C4').format={fill:'#263C5A',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:30};
  [37,22,83].forEach((w,i)=>a.getRangeByIndexes(0,i,24,1).format.columnWidth=w);
  a.getRange('A16:A19').values=[['All32 formal fields executed. The failed numerical gate is retained, not relaxed or retried.'],['Software time and board energy are diagnostic, not a qualified same-reference performance result.'],['This is not physical optical latency, optical deployment energy or wall-socket power.'],['Loading, file decoding, RGB/crop and quality-pixel preparation are outside the software window.']];
  a.getRange('A16:C19').format.font.italic=true;
  a.getRange('A21:B24').values=[['Original report',rel(reportFile)],['Report SHA256',sha(reportFile)],['Independent audit',rel(auditFile)],['Audit SHA256',sha(auditFile)]];
}
if(includeABO){
  const task=path.join(REPO,'LightGenV2/tasks/t08_abo_image_text_retrieval');
  const run=path.join(task,'runs/simulation/abo_t2i_a100_20261029_01');
  const pack=path.join(task,'releases/abo_text_to_image_FULL_TEST_20261029_01');
  const report=path.join(run,'inference/report.json');
  const audit=path.join(run,'inference/local_independent_cpu_audit01.json');
  const delivery=path.join(run,'lab_full_TEST_delivery_receipt01.json');
  assert(sha(report)==='1d0677338c3a9ee0217ef9c92779e50f757fe8d04f266aff0c0e579059cc59e3','Pinned completed FULL report');
  assert(sha(audit)==='337b0035280c4d85ff2c9fe04f5a98b97b88f463dbfafaac375d0c273baf8fd3','Existing CPU score replay audit');
  assert(sha(delivery)==='8a6fb592c1014670f0d4b3527e0e27600f53e1dd1966998588898d217711dcc7','Existing local and lab ZIP delivery receipt');
  assert(sha(pack+'.zip')==='fe8a1d9764a43e42df152104c2f965a2ab975c9c9e82d00c4f47e8c6ca1316d9','Delivered ZIP unchanged');
  const manifest=read(path.join(pack,'manifest.json'));
  const sourceRead=name=>{
    const file=path.join(pack,'source',name),pin=manifest.files.find(x=>x.path==='source/'+name);
    assert(pin&&sha(file)===pin.sha256,'Pinned package source '+name);return read(file);
  };
  const ds=read(report);assert(ds.status==='complete'&&ds.query_count===100&&ds.gallery_count===2400&&ds.model_metadata.backbone_total_parameters===27480134248,'Full model, complete population');
  const records=[
    {model:'DeepSeek-VL2 Full',provenance:'New A10080 frozen inference',metrics:ds.metrics,source:rel(report),sha256:sha(report)},
    {model:'LightGen-II simulation',provenance:'Historical, TEST-selected',metrics:sourceRead('historical_ours_10cm_final_report.json').bidirectional.text_to_image,source:'source/historical_ours_10cm_final_report.json'},
    {model:'Qwen dynamic 2048D',provenance:'Historical frozen baseline',metrics:sourceRead('historical_qwen_report.json').metrics.dynamic_2048,source:'source/historical_qwen_report.json'},
    {model:'CLIP ViT-B/32',provenance:'Historical frozen baseline',metrics:sourceRead('historical_clip_report.json').results.easy100.text_to_image,source:'source/historical_clip_report.json'}
  ];
  for(const row of records){assert(row.metrics.query_count===100&&row.metrics.gallery_count===2400,'All 100 queries, same gallery');if(!row.sha256)row.sha256=sha(path.join(pack,row.source));}
  aboResult={records,query_count:100,gallery_count:2400,positives_per_query:24,zip_sha256:sha(pack+'.zip'),audit_sha256:sha(audit),lab_delivery_receipt_sha256:sha(delivery),visual_QA:'pending_manual_not_waived',rental_work_complete:true};
  const a=aboSheet;a.showGridLines=false;a.tabColor='#263C5A';
  a.getRange('A1:H25').format={font:{name:'Arial',size:10},verticalAlignment:'center',rowHeight:25};
  a.getRange('A2').values=[['ABO official-title text-to-image retrieval']];a.getRange('A2:H2').format.font={name:'Arial',size:15,bold:true};
  a.getRange('A3').values=[['100 queries, 2400 original TEST images, 24 exact-SKU positives per query.']];
  a.getRange('A5:H5').values=[['Model','Result origin','Hit@1','Hit@5','Hit@10','MRR','mAP','Queries']];
  a.getRange('A6:H9').values=records.map(r=>[r.model,r.provenance,r.metrics.hit_at_1,r.metrics.hit_at_5,r.metrics.hit_at_10,r.metrics.mrr,r.metrics.map,r.metrics.query_count]);
  a.getRange('A5:H5').format={fill:'#263C5A',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:32};
  a.getRange('C6:E9').setNumberFormat('0.0%');a.getRange('F6:G9').setNumberFormat('0.000000');a.getRange('H6:H9').setNumberFormat('0');a.getRange('C6:H9').format.horizontalAlignment='right';
  [31,34,12,12,12,14,14,10].forEach((w,i)=>a.getRangeByIndexes(0,i,25,1).format.columnWidth=w);
  const notes=[
    'Only DeepSeek-VL2 Full is newly measured here. The other three models reuse saved historical results.',
    'DeepSeek Full: 27,480,134,248 parameters, 2560D, zero trainable parameters. No quality head or tuning.',
    'Hit@K means at least one exact-SKU positive in the first K images. It is not category accuracy or Recall@K.',
    'LightGen-II is historical simulation (86%), not physical 85%. Historical TEST selection limits generalization.',
    'All 100 queries are retained. The older 10 examples were deliberately selected in favor of LightGen-II.',
    'DeepSeek audit replays saved scores, full order and Top10. It does not recompute all embedding cosines.',
    'ABO latency and energy were not formally measured. Do not reuse T06 timings as ABO latency.',
    'Original results and the complete figure ZIP are SHA-preserved locally and on lab. No rental work remains.',
    'Offline HTML visual review is still pending. It is not marked passed or waived and can be done locally.'
  ];
  a.getRange('A12:A20').values=notes.map(n=>[n]);a.getRange('A12:H20').format.rowHeight=24;
  a.getRange('A12:H18').format.font.italic=true;
  a.getRange('A22').values=[['Source identity (SHA256)']];a.getRange('A22:H22').format.font.bold=true;
  a.getRange('A23:B25').values=[['New FULL DS report',sha(report)],['Independent CPU replay',sha(audit)],['Complete TEST ZIP',aboResult.zip_sha256]];
}
for(const sheet of [s,raw]){sheet.showGridLines=false;sheet.getRange(sheet===s?'A1:I31':'A1:O42').format.font={name:'Arial',size:10};}
s.tabColor='#263C5A';
s.getRange('A2').values=[[energyRemeasured?'A100 full-input time and energy':'A100 full-input timing comparison']];s.getRange('A2:I2').format.font={name:'Arial',size:15,bold:true};
s.getRange('A3').values=[['Three fresh rounds per configuration. All rounds retained. 512 videos per round.']];
const headers=energyRemeasured?['Model','MPS (%)','Hardware','Time / 16 videos (ms)','Native tokens / s','GPU board / 16 videos (J)','GPU mean (W)','Board + host proxy / 16 (J)','Timing scope']:['Model','MPS (%)','Hardware','Time / 16 videos (ms)','Native tokens / s','Energy / 16 videos (J)','GPU mean (W)','Time / LG equivalent','LG rate / native rate'];
s.getRange('A5:I5').values=[headers];
raw.getRange('A1:O1').values=[['Config','Model','MPS (%)','Hardware','Repeat','Videos','Groups of 16','Continuous wall (s)','Native input tokens','GPU energy (J)','System proxy energy (J)','Energy qualified','Report SHA256','Original source','Independent audit SHA256']];
raw.getRange('A2:O37').values=trials.map(t=>[t.key,display[t.model],t.cap,t.hardware,t.repeat,t.videos,t.groups,t.seconds,t.tokens,t.gpu_j,t.system_j,t.energy,t.report_sha256,t.source,t.audit_sha256]);
if(energyRemeasured){
  raw.getRange('P1:T1').values=[['Energy protocol','Strict250ms qualified','Maximum support age (s)','Gaps over250ms','Gap support fraction']];
  raw.getRange('P2:T37').values=trials.map(t=>[t.qualification,t.strict250ms_qualified,t.max_power_sample_age_s,t.gap_over250ms_count,t.gap_over250ms_support_fraction]);
  raw.getRange('P1:T37').format={font:{name:'Arial',size:10},rowHeight:28,columnWidth:24};
  raw.getRange('P1:T1').format={fill:'#263C5A',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:38,wrapText:true};
  raw.getRange('P2:P37').format.columnWidth=64;raw.getRange('R2:R37').setNumberFormat('0.000000');raw.getRange('T2:T37').setNumberFormat('0.0000%');
}
for(const [i,c] of configs.entries()){
  const row=6+i+(i>=8?2:0),key=JSON.stringify(c.key),sum=col=>`SUMIFS('Trials'!$${col}$2:$${col}$37,'Trials'!$A$2:$A$37,${key})`;
  s.getRange(`A${row}:C${row}`).values=[[display[c.model],c.cap,c.hardware]];
  s.getRange(`D${row}`).formulas=[[`=${sum('H')}/${sum('G')}*1000`]];
  s.getRange(`E${row}`).formulas=[[`=IF(${sum('I')}=0,"N/A",${sum('I')}/${sum('H')})`]];
  s.getRange(`F${row}`).formulas=[[c.energy?`=${sum(energyRemeasured?'J':'K')}/${sum('G')}`:'="N/A"']];
  s.getRange(`G${row}`).formulas=[[c.energy?`=${sum('J')}/${sum('H')}`:'="N/A"']];
  if(energyRemeasured){
    s.getRange(`H${row}`).formulas=[[`=${sum('K')}/${sum('G')}`]];
    s.getRange(`I${row}`).values=[['File to CPU score']];
  }else{
    s.getRange(`H${row}`).formulas=[[`=D${row}/$D$23`]];
    s.getRange(`I${row}`).formulas=[[`=IF(E${row}="N/A","N/A",$E$23/E${row})`]];
  }
}
s.getRange('A15').values=[[energyRemeasured?'New rental time + energy results':'New rental timing-only results']];s.getRange('A15:I15').format.font.bold=true;
s.getRange('A22:I22').values=[headers];s.getRange('A23:C23').values=[['LightGen-II equivalent',null,'Operator scheduling']];
s.getRange('D23:G23').values=[[lg.LG_time16_s*1000,lg.LG_tokens_per_s,'N/A','N/A']];
s.getRange('A25').values=[['LG includes resize and normalization, but excludes file decoding, RGB/crop and quality-pixel preparation.']];
s.getRange('A26').values=[['Ratios compare different scopes and hardware. They are not same-scope physical end-to-end speedups.']];
s.getRange('A27').values=[['Lab energy: NVML board integral plus the fixed 338.2 W host proxy once. Not a wattmeter total.']];
s.getRange('A28').values=[['Rental and LG energy are unqualified: N/A. Native tokens use each baseline’s own numerator.']];
s.getRange('A29').values=[['LG alone shares Qwen input-token counts. Model loading is outside each baseline timing window.']];
if(energyRemeasured){
  s.getRange('A22:I23').clear({applyTo:'contents'});
  s.getRange('A22').values=[['LightGen rental software runtime and energy are on the next tab.']];
  s.getRange('A25').values=[['36 baseline trials. DeepSeek uses authorized actual-sample ZOH. Per-trial protocol flags are in Trials.']];
  s.getRange('A26').values=[['Lab A10040GB/250W and rental A10080GB/400W are distinct hardware, not identical-device comparisons.']];
  s.getRange('A27').values=[['GPU board energy is measured. Column H adds the fixed338.2W host proxy once; it is not socket power.']];
  s.getRange('A28').values=[['No old timing-only rental result is used. Missing utilization/memory diagnostics do not mean missing power.']];
  s.getRange('A29').values=[['Baseline loading is excluded. All511 gaps from the first file preparation to final CPU score are included.']];
  s.getRange('A30').values=[['Rental: six Kimi, one separate DS CPU derivation and five new DS rounds. Original failed files retained.']];
}
raw.getRange('A40:B42').values=[['Source summary',rel(labFile)],['LG reference',rel(lgFile)],['Rental root receipt',rel(rootReceipt)]];
for(const sheet of [s,raw]){
  const h=sheet===s?'A5:I5':'A1:O1';sheet.getRange(h).format={fill:'#263C5A',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:36,wrapText:true};
}
s.getRange('A22:I22').format={fill:'#526983',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:36,wrapText:true};
s.getRange('A6:I23').format.rowHeight=25;s.getRange('A22:I22').format.rowHeight=36;
s.getRange('D6:I23').setNumberFormat('#,##0.000');s.getRange('B6:B23').setNumberFormat('0');
s.getRange('D6:I23').format.horizontalAlignment='right';
s.getRange('B6:B23').format.horizontalAlignment='center';
s.getRange('A22:I22').format.horizontalAlignment='center';
const widths=[25,10,24,18,18,18,16,18,20];widths.forEach((w,i)=>s.getRangeByIndexes(0,i,29,1).format.columnWidth=w);
if(energyRemeasured)s.getRange('F1:H29').format.columnWidth=22;
s.getRange('A25:I29').format.font.italic=true;s.getRange('A25:I29').format.rowHeight=22;
raw.getRange('A1:O37').format.rowHeight=28;raw.getRange('A1:O1').format.rowHeight=38;
raw.getRange('A1:B37').format.columnWidth=22;raw.getRange('C1:C37').format.columnWidth=10;raw.getRange('D1:D37').format.columnWidth=25;
raw.getRange('E1:G37').format.columnWidth=13;raw.getRange('H1:K37').format.columnWidth=20;raw.getRange('L1:L37').format.columnWidth=18;
raw.getRange('M1:M37').format.columnWidth=70;raw.getRange('N1:N37').format.columnWidth=110;raw.getRange('O1:O37').format.columnWidth=70;
raw.getRange('H2:K37').setNumberFormat('#,##0.000000');raw.getRange('I2:I37').setNumberFormat('#,##0');raw.freezePanes.freezeRows(1);raw.freezePanes.freezeColumns(2);
raw.getRange('C2:C37').format.horizontalAlignment='center';
wb.recalculate();
for(const [i,a] of summary.entries()){
  const row=6+i+(i>=8?2:0);close(s.getRange(`D${row}`).values[0][0],a.time16_s*1000);
  if(a.native_tokens_per_s)close(s.getRange(`E${row}`).values[0][0],a.native_tokens_per_s);
  if(energyRemeasured){close(s.getRange(`F${row}`).values[0][0],configs[i].trials.reduce((n,t)=>n+t.gpu_j,0)/96);close(s.getRange(`H${row}`).values[0][0],a.energy16_j);close(s.getRange(`G${row}`).values[0][0],a.gpu_mean_w);}
}
const original=raw.getRange('H2').values[0][0],old=s.getRange('D6').values[0][0];raw.getRange('H2').values=[[original+1]];wb.recalculate();
close(s.getRange('D6').values[0][0],old+1000/96);raw.getRange('H2').values=[[original]];wb.recalculate();close(s.getRange('D6').values[0][0],old);
if(energyRemeasured){
  const j=raw.getRange('J2').values[0][0],k=raw.getRange('K2').values[0][0],f=s.getRange('F6').values[0][0],h=s.getRange('H6').values[0][0];
  raw.getRange('J2:K2').values=[[j+10,k+10]];wb.recalculate();
  close(s.getRange('F6').values[0][0],f+10/96);close(s.getRange('H6').values[0][0],h+10/96);
  raw.getRange('J2:K2').values=[[j,k]];wb.recalculate();close(s.getRange('F6').values[0][0],f);close(s.getRange('H6').values[0][0],h);
}
console.log((await wb.inspect({kind:'table',range:'Comparison!A5:I19',include:'values,formulas',tableMaxRows:15,tableMaxCols:9,maxChars:3500})).ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},maxChars:1500});console.log(errors.ndjson);
assert(!/#REF!|#DIV\/0!|#VALUE!|#NAME\?|#NUM!|#NULL!|#SPILL!|#CALC!/.test(errors.ndjson),'Formula error scan');
for(const [sheetName,range,name] of [['Comparison',energyRemeasured?'A1:I30':'A1:I29','comparison_preview.png'],...(energyRemeasured?[['LightGen rental','A1:C19','lightgen_preview.png']]:[]),...(includeABO?[['ABO retrieval','A1:H25','abo_preview.png']]:[]),['Trials','A1:L13','trials_preview.png']]){
  const blob=await wb.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(OUT,name),new Uint8Array(await blob.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(OUT,BOOK));
const result={status:'36_ACCEPTED_TRIALS_12_CONFIGS_WITH_DISTINCT_HARDWARE_AND_ENERGY_SCOPE',new_energy_remeasurement:energyRemeasured,LG_reference:{file:rel(lgFile),sha256:sha(lgFile),time16_ms:lg.LG_time16_s*1000,native_tokens_per_s:lg.LG_tokens_per_s,energy_qualified:false},...(lgRental?{LightGen_rental:lgRental}:{}),records:summary,trials,...(includeABO?{ABO:aboResult}:{})};
await fs.writeFile(path.join(OUT,'combined_summary.json'),JSON.stringify(result,null,2),{flag:repairDraft?'w':'wx'});
await fs.writeFile(path.join(OUT,'workbook_receipt.json'),JSON.stringify({file:BOOK,sha256:sha(path.join(OUT,BOOK)),bytes:fss.statSync(path.join(OUT,BOOK)).size,tests:{trials:36,configs:12,formula_input_perturbation_and_restore:true,all_summary_time_formulas_match:true},source_pins:{lab:sha(labFile),lg:sha(lgFile),rental_root:sha(rootReceipt)}},null,2),{flag:repairDraft?'w':'wx'});
console.log(JSON.stringify({status:'exported_needs_visual_review_and_lab_backup',output:path.join(OUT,BOOK),sha256:sha(path.join(OUT,BOOK))}));
