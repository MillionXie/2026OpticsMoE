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
if(process.argv.includes('--lgvq-only-reorganized')){
  await buildLGVQOnly();
}else{
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
    s.getRange(`J${row}`).formulas=[[`=D${row}/$D$23`]];
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
  s.getRange('J5').values=[['Time / LG equivalent (x, different scope)']];
  s.getRange('J22').values=[['Time / LG equivalent (x, different scope)']];
  s.getRange('H23:J23').values=[['N/A','Equivalent scheduling',1]];
  s.getRange('A24').values=[['LG45.949ms includes resize/normalization, excludes decode/RGB/crop/quality preparation. Not same-scope E2E.']];
  s.getRange('A25').values=[['36 baseline trials. DeepSeek uses authorized actual-sample ZOH. Per-trial protocol flags are in Trials.']];
  s.getRange('A26').values=[['Lab A10040GB/250W and rental A10080GB/400W are distinct hardware, not identical-device comparisons.']];
  s.getRange('A27').values=[['GPU board energy is measured. Column H adds the fixed338.2W host proxy once; it is not socket power.']];
  s.getRange('A28').values=[['LG rental FFT software diagnostic685.421ms is separate on the next tab. It does not replace LG45.949ms.']];
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
if(energyRemeasured){
  s.getRange('J1:J30').format={font:{name:'Arial',size:10},columnWidth:25,verticalAlignment:'center'};
  for(const cell of ['J5','J22'])s.getRange(cell).format={fill:cell==='J5'?'#263C5A':'#526983',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true};
  s.getRange('A5:J5').format.rowHeight=46;s.getRange('A22:J22').format.rowHeight=46;
  s.getRange('J6:J23').setNumberFormat('0.0');s.getRange('J6:J23').format.horizontalAlignment='right';
  s.getRange('A24:J24').format={font:{name:'Arial',size:10,italic:true},rowHeight:24};
}
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
  if(energyRemeasured){close(s.getRange(`F${row}`).values[0][0],configs[i].trials.reduce((n,t)=>n+t.gpu_j,0)/96);close(s.getRange(`H${row}`).values[0][0],a.energy16_j);close(s.getRange(`G${row}`).values[0][0],a.gpu_mean_w);close(s.getRange(`J${row}`).values[0][0],a.time16_s/lg.LG_time16_s);}
}
const original=raw.getRange('H2').values[0][0],old=s.getRange('D6').values[0][0];raw.getRange('H2').values=[[original+1]];wb.recalculate();
close(s.getRange('D6').values[0][0],old+1000/96);raw.getRange('H2').values=[[original]];wb.recalculate();close(s.getRange('D6').values[0][0],old);
if(energyRemeasured){
  const denominator=s.getRange('D23').values[0][0],ratio=s.getRange('J6').values[0][0];
  s.getRange('D23').values=[[denominator*2]];wb.recalculate();close(s.getRange('J6').values[0][0],ratio/2);
  s.getRange('D23').values=[[denominator]];wb.recalculate();close(s.getRange('J6').values[0][0],ratio);
}
if(energyRemeasured){
  const j=raw.getRange('J2').values[0][0],k=raw.getRange('K2').values[0][0],f=s.getRange('F6').values[0][0],h=s.getRange('H6').values[0][0];
  raw.getRange('J2:K2').values=[[j+10,k+10]];wb.recalculate();
  close(s.getRange('F6').values[0][0],f+10/96);close(s.getRange('H6').values[0][0],h+10/96);
  raw.getRange('J2:K2').values=[[j,k]];wb.recalculate();close(s.getRange('F6').values[0][0],f);close(s.getRange('H6').values[0][0],h);
}
console.log((await wb.inspect({kind:'table',range:'Comparison!A5:I19',include:'values,formulas',tableMaxRows:15,tableMaxCols:9,maxChars:3500})).ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},maxChars:1500});console.log(errors.ndjson);
assert(!/#REF!|#DIV\/0!|#VALUE!|#NAME\?|#NUM!|#NULL!|#SPILL!|#CALC!/.test(errors.ndjson),'Formula error scan');
for(const [sheetName,range,name] of [['Comparison',energyRemeasured?'A1:J30':'A1:I29','comparison_preview.png'],...(energyRemeasured?[['LightGen rental','A1:C19','lightgen_preview.png']]:[]),...(includeABO?[['ABO retrieval','A1:H25','abo_preview.png']]:[]),['Trials','A1:L13','trials_preview.png']]){
  const blob=await wb.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(OUT,name),new Uint8Array(await blob.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(OUT,BOOK));
const result={status:'36_ACCEPTED_TRIALS_12_CONFIGS_WITH_DISTINCT_HARDWARE_AND_ENERGY_SCOPE',new_energy_remeasurement:energyRemeasured,LG_reference:{file:rel(lgFile),sha256:sha(lgFile),time16_ms:lg.LG_time16_s*1000,native_tokens_per_s:lg.LG_tokens_per_s,energy_qualified:false},...(lgRental?{LightGen_rental:lgRental}:{}),records:summary,trials,...(includeABO?{ABO:aboResult}:{})};
await fs.writeFile(path.join(OUT,'combined_summary.json'),JSON.stringify(result,null,2),{flag:repairDraft?'w':'wx'});
await fs.writeFile(path.join(OUT,'workbook_receipt.json'),JSON.stringify({file:BOOK,sha256:sha(path.join(OUT,BOOK)),bytes:fss.statSync(path.join(OUT,BOOK)).size,tests:{trials:36,configs:12,formula_input_perturbation_and_restore:true,all_summary_time_formulas_match:true},source_pins:{lab:sha(labFile),lg:sha(lgFile),rental_root:sha(rootReceipt)}},null,2),{flag:repairDraft?'w':'wx'});
console.log(JSON.stringify({status:'exported_needs_visual_review_and_lab_backup',output:path.join(OUT,BOOK),sha256:sha(path.join(OUT,BOOK))}));
}

async function buildLGVQOnly(){
  const out=path.join(HERE,'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b/LGVQ_LG_first_parallel_20261030_01');
  const book=path.join(out,'LGVQ_LightGen_first_parallel_20261030.xlsx'),input=path.join(out,'lgvq_reorganized_sources01.json');
  const sha=p=>crypto.createHash('sha256').update(fss.readFileSync(p)).digest('hex');
  const ok=(v,m)=>{if(!v)throw Error(m);};
  const close=(a,b)=>ok(typeof a==='number'&&Math.abs(a-b)<=Math.max(1e-7,Math.abs(b)*1e-9),`Calculation differs ${a} ${b}`);
  ok(!fss.existsSync(book),'Preserve existing workbook');
  ok(sha(input)==='9873a8a7c021a977dc67682b1dcb618fbbaabf7f4944eb4ed2e137bcc1decc1d','Pinned new saved-data calculation');
  const d=JSON.parse(fss.readFileSync(input,'utf8'));
  ok(d.trials.length===36&&d.baselines.length===12&&!d.torch_imported,'Exact accepted source population');
  for(const f of d.source_files)ok(sha(path.join(REPO,f.path))===f.sha256,'Original source changed');
  const wb=Workbook.create(),s=wb.worksheets.add('主表'),detail=wb.worksheets.add('LG时间明细'),raw=wb.worksheets.add('轮次原值');
  const baseStyle=sheet=>{sheet.showGridLines=false;sheet.getRange('A1:M70').format={font:{name:'Arial',size:10},verticalAlignment:'center',rowHeight:25};};
  for(const sheet of [s,detail,raw])baseStyle(sheet);
  const put=(sheet,row,values)=>sheet.getRangeByIndexes(row-1,0,values.length,values[0].length).values=values;
  const formula=(sheet,cell,value)=>sheet.getRange(cell).formulas=[[value]];
  const header=(sheet,row,cols)=>sheet.getRangeByIndexes(row-1,0,1,cols).format={fill:'#263C5A',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:43};
  const title=(sheet,cell,text)=>{sheet.getRange(cell).values=[[text]];sheet.getRange(cell).format.font={name:'Arial',size:15,bold:true};};
  const labels={qwen:'Qwen3-VL-2B',glm:'GLM-4.6V-Flash',clip:'CLIP ViT-B/32',yolo:'YOLO11s',kimi:'Kimi-VL-A3B',deepseek:'DeepSeek-VL2 Full'};
  // All36 immutable accepted round values remain intact, including protocol flags.
  put(raw,1,[['Config','Model','MPS (%)','Hardware','Repeat','Videos','Groups of16','Continuous wall (s)','Native tokens','GPU energy (J)','Board+host proxy (J)','Energy accepted','Report SHA256','Original source','Independent audit SHA256','Energy protocol','Strict250ms qualified','Maximum support age (s)','Gaps over250ms','Gap support fraction']]);
  raw.getRange('A2:T37').values=d.trials.map(t=>[t.key,labels[t.model],t.cap,t.hardware,t.repeat,t.videos,t.groups,t.seconds,t.tokens,t.gpu_j,t.system_j,t.energy,t.report_sha256,t.source,t.audit_sha256,t.qualification,t.strict250ms_qualified,t.max_power_sample_age_s,t.gap_over250ms_count,t.gap_over250ms_support_fraction]);
  header(raw,1,20);raw.freezePanes.freezeRows(1);raw.freezePanes.freezeColumns(2);
  raw.getRange('A1:D37').format.columnWidth=23;raw.getRange('E1:G37').format.columnWidth=13;raw.getRange('H1:L37').format.columnWidth=21;raw.getRange('M1:O37').format.columnWidth=75;raw.getRange('N1:N37').format.columnWidth=110;raw.getRange('P1:P37').format.columnWidth=65;raw.getRange('Q1:T37').format.columnWidth=24;
  raw.getRange('H2:K37').setNumberFormat('#,##0.000000');raw.getRange('I2:I37').setNumberFormat('#,##0');raw.getRange('T2:T37').setNumberFormat('0.0000%');
  title(detail,'A2','LightGen阶段时间与并行计算');
  put(detail,3,[['每场16视频、64帧。关键路径贡献可相加，全部执行工作因并行重叠不能直接相加。']]);
  put(detail,5,[['阶段','实验室r1/ms','实验室r2/ms','实验室r3/ms','实验室合并/ms','80GB并行/ms','实验室执行工作/ms','80GB执行工作/ms']]);header(detail,5,8);
  const stages=[['resize','resize'],['normalization','归一化'],['patch','Patch与原BF16 cast'],['tokenizer','文字tokenizer'],['emb_text','文字embedding'],['emb_conv3d','视觉Conv3D embedding'],['emb_position','位置embedding'],['adapter_visual_pool','视觉pool'],['adapter_visual','视觉adapter'],['optics1','光学层1'],['oeo1','OEO1'],['optics2','光学层2'],['oeo2','OEO2'],['optics3','光学层3'],['oeo3','OEO3'],['merger','Merger'],['adapter_language','语言adapter'],['optics4','光学层4'],['oeo4','OEO4'],['optics5','光学层5'],['oeo5','OEO5'],['optics6','光学层6'],['oeo6','OEO6'],['decoder','Decoder与评分']];
  for(const [i,[key,label]] of stages.entries()){
    const r=6+i;put(detail,r,[[label,...d.LG_lab.source_rounds.map(x=>(x.equivalent_s_per_group[key]??0)*1000),null,(d.LG_rental.equivalent_s_per_group[key]??0)*1000,(d.LG_lab.work_s_per_group[key]??0)*1000,(d.LG_rental.work_s_per_group[key]??0)*1000]]);
    formula(detail,'E'+r,`=SUM(B${r}:D${r})/3`);
  }
  put(detail,30,[['每16视频并行时间合计']]);for(const c of ['B','C','D','E','F'])formula(detail,c+'30',`=SUM(${c}6:${c}29)`);
  put(detail,32,[['实验室：三轮，每轮512原TEST视频，32场，原输出逐位一致。']]);
  put(detail,33,[['80GB：原单轮512视频。重放保存trace构造同一两场流水资源调度，未重测模型。']]);
  put(detail,34,[['80GB仅92/512输出逐位一致，数值门未通过；并行时间和组件能量均为诊断估算。']]);
  put(detail,35,[['保留原CPU工作线程分配和共享GPU/光学资源依赖，不把串行工作总和直接除以16。']]);
  put(detail,36,[['固定每层光周期 = 1/1400 + 0.000300 + 0.0000307 秒，共6层。']]);
  put(detail,37,[['包含resize、归一化、patch、embedding及电子算子；不包含文件解码/RGB裁剪/quality准备与调度等待。']]);
  title(detail,'A40','原固定组件功率与时间并集能量');
  put(detail,42,[['组件','功率/W','计算约定']]);header(detail,42,3);
  put(detail,43,[['控制器',d.component_powers_w.controller,'全窗口时间并集只算一次'],['光器件',d.component_powers_w.optical_devices,'光学阶段时间并集'],['电子板卡',d.component_powers_w.electronic_board,'全部电子阶段时间并集，并行不重复']]);
  put(detail,48,[['每16视频时间并集','实验室r1','实验室r2','实验室r3','实验室合并','80GB单轮']]);header(detail,48,6);
  const kinds=[['controller_union_ms','控制器/s'],['optical_union_ms','光器件/s'],['electronic_union_ms','电子板卡/s'],['optical_electronic_overlap_ms','光电重叠/s']];
  for(const [i,[key,label]] of kinds.entries()){
    const r=49+i;put(detail,r,[[label,...d.LG_lab.source_rounds.map(x=>x.component_energy[key]/1000/x.fields),null,d.LG_rental.source_rounds[0].component_energy[key]/1000/32]]);formula(detail,'E'+r,`=SUM(B${r}:D${r})/3`);
  }
  put(detail,54,[['固定组件代理能量/J每16视频']]);
  for(const c of ['B','C','D','E','F'])formula(detail,c+'54',`=${c}49*$B$43+${c}50*$B$44+${c}51*$B$45`);
  put(detail,56,[['共同Qwen有效tokens/16视频',d.common_tokens16],['实验室LG自身tokens/16视频',d.LG_lab.native_tokens/d.LG_lab.fields],['80GB LG自身tokens/16视频',d.LG_rental.native_tokens/d.LG_rental.fields]]);
  put(detail,60,[['80GB实际FFT软件/ms每16视频',d.LG_rental_software.actual_FFT_s_per_16_videos*1000],['80GB实际FFT板卡/J每16视频',d.LG_rental_software.gpu_board_energy_j/32],['80GB实际FFT板卡均功率/W',d.LG_rental_software.gpu_board_mean_w]]);
  put(detail,64,[['组件能量是原功率模型的代理，不是光路部署电表实测，不能拿FFT板卡能量代替。']]);
  put(detail,65,[['LG自身token按每场共享prompt只计一次；共同tokens统一采用原Qwen分子，二者不可混称。']]);
  put(detail,67,[['来源汇总SHA256',sha(input)]]);
  detail.getRange('A1:A67').format.columnWidth=40;detail.getRange('B1:H67').format.columnWidth=21;detail.getRange('B6:H62').setNumberFormat('#,##0.000000');
  detail.getRange('A32:H37').format.font.italic=true;detail.getRange('A64:H65').format.font.italic=true;
  title(s,'A2','LGVQ时间、能耗与吞吐倍率');
  put(s,3,[['前两行是16视频光场并行的LG。基线为B1串行，统一折算每16视频/64帧，三轮合并不选最快。']]);
  put(s,5,[['模型','平台','资源/%','并行视频','时间/ms每16视频','等效单帧/ms','代理能量/J每16视频','共同Qwen tokens/s','自身tokens/s','实测板卡均W','代理均功率/W','结果口径','实测板卡/J每16视频']]);header(s,5,13);
  const L=[d.LG_lab,d.LG_rental];
  for(const [i,x] of L.entries()){
    const row=6+i,col=i?'F':'E';put(s,row,[[i?'LightGen-II（80GB）':'LightGen-II（实验室）',i?'A100 80GB / 400W':'A100 40GB / 250W',100,16,null,null,null,null,null,'N/A',null,i?'单轮诊断，并行估算':'三轮通过，并行估算','N/A']]);
    formula(s,'E'+row,`='LG时间明细'!${col}30`);formula(s,'F'+row,`=E${row}/64`);formula(s,'G'+row,`='LG时间明细'!${col}54`);formula(s,'H'+row,`='LG时间明细'!$B$56/E${row}*1000`);formula(s,'I'+row,`='LG时间明细'!$B$${i?58:57}/E${row}*1000`);formula(s,'K'+row,`=G${row}/E${row}*1000`);
  }
  for(const [i,c] of d.baselines.entries()){
    const row=8+i,key=JSON.stringify(c.key),sum=k=>`SUMIFS('轮次原值'!$${k}$2:$${k}$37,'轮次原值'!$A$2:$A$37,${key})`;
    put(s,row,[[labels[c.model],c.hardware,c.cap,1,null,null,null,null,null,null,null,'三轮文件至CPU评分',null]]);
    formula(s,'E'+row,`=${sum('H')}/${sum('G')}*1000`);formula(s,'F'+row,`=E${row}/64`);formula(s,'G'+row,`=${sum('K')}/${sum('G')}`);formula(s,'H'+row,`='LG时间明细'!$B$56/E${row}*1000`);formula(s,'I'+row,`=IF(${sum('I')}=0,"N/A",${sum('I')}/${sum('H')})`);formula(s,'J'+row,`=${sum('J')}/${sum('H')}`);formula(s,'K'+row,`=G${row}/E${row}*1000`);formula(s,'M'+row,`=${sum('J')}/${sum('G')}`);
  }
  put(s,21,[['LG为电子实测+固定光周期的并行调度估算；基线含文件解码至CPU评分。不是同范围物理端到端倍率。']]);
  put(s,22,[['能量：LG使用原固定组件功率并集代理；基线为板卡积分+338.2W主机代理，均非插座电表整机实测。']]);
  put(s,23,[['80GB LG数值门未通过，估算仅作诊断。实际685.421ms FFT软件窗口与板卡能量在LG时间明细中单列。']]);
  title(s,'A25','相对两种LightGen的倍率');
  put(s,27,[['模型','资源/%','时间比/LG实验室','时间比/LG80GB','能量比/LG实验室','能量比/LG80GB','共同速率增益/LG实验室','共同速率增益/LG80GB','自身速率比/LG实验室','自身速率比/LG80GB','代理功率比/LG实验室','代理功率比/LG80GB']]);header(s,27,12);
  for(let i=0;i<14;i++){
    const r=28+i,v=6+i;formula(s,'A'+r,`=A${v}`);formula(s,'B'+r,`=C${v}`);
    for(const [col,f] of [['C',`=E${v}/$E$6`],['D',`=E${v}/$E$7`],['E',`=G${v}/$G$6`],['F',`=G${v}/$G$7`],['G',`=$H$6/H${v}`],['H',`=$H$7/H${v}`],['I',`=IF(I${v}="N/A","N/A",$I$6/I${v})`],['J',`=IF(I${v}="N/A","N/A",$I$7/I${v})`],['K',`=K${v}/$K$6`],['L',`=K${v}/$K$7`]])formula(s,col+r,f);
  }
  put(s,43,[['时间/能量/功率比 = 基线÷所选LG；共同/自身速率比 = 所选LG÷基线。LG参照行同样按此公式。']]);
  put(s,44,[['共同tokens/s采用相同Qwen分子，因此共同速率增益应等于时间比；自身tokens定义不同，不应当作公平跨模型吞吐。']]);
  put(s,45,[['详细阶段的关键路径贡献合计等于并行时间，执行工作列保留重叠，不误相加、不重复除以16。']]);
  const widths=[29,27,15,15,23,20,24,25,22,21,23,29,24];widths.forEach((x,i)=>s.getRangeByIndexes(0,i,45,1).format.columnWidth=x);
  s.getRange('E6:M19').setNumberFormat('#,##0.000');s.getRange('C28:L41').setNumberFormat('0.000');s.getRange('C6:D19').setNumberFormat('0');
  s.getRange('E6:M19').format.horizontalAlignment='right';s.getRange('C28:L41').format.horizontalAlignment='right';s.getRange('A21:M23').format.font.italic=true;s.getRange('A43:M45').format.font.italic=true;s.getRange('A6:M7').format.fill='#EDF3F9';
  wb.recalculate();
  for(const [i,x] of L.entries()){close(s.getRange('E'+(6+i)).values[0][0],x.time16_s*1000);close(s.getRange('G'+(6+i)).values[0][0],x.component_energy16_j);}
  for(let i=0;i<14;i++){const r=28+i;close(s.getRange('C'+r).values[0][0],s.getRange('G'+r).values[0][0]);close(s.getRange('D'+r).values[0][0],s.getRange('H'+r).values[0][0]);}
  // Verify denominator, energy-power input, and baseline source dependencies.
  const change=(sheet,cell,delta,target,expected)=>{const old=sheet.getRange(cell).values[0][0],before=target.values[0][0];sheet.getRange(cell).values=[[old+delta]];wb.recalculate();close(target.values[0][0],expected(before,old,delta));sheet.getRange(cell).values=[[old]];wb.recalculate();close(target.values[0][0],before);};
  change(detail,'B43',1,s.getRange('G6'),(b)=>b+detail.getRange('E49').values[0][0]);
  change(raw,'H2',1,s.getRange('E8'),b=>b+1000/96);
  change(detail,'B6',3,s.getRange('C30'),(b,old,delta)=>s.getRange('E8').values[0][0]/(d.LG_lab.time16_s*1000+delta/3));
  const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:1500});ok(!errors.ndjson.includes('"kind":"match"'),'Formula error scan');
  const previews=[['主表','A1:M23','main_preview.png'],['主表','A25:L45','ratios_preview.png'],['LG时间明细','A1:H37','lg_stages_preview.png'],['LG时间明细','A40:F65','lg_energy_preview.png'],['轮次原值','A25:L37','trials_preview.png']];
  for(const [sheetName,range,name] of previews){const p=await wb.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(out,name),new Uint8Array(await p.arrayBuffer()));}
  await (await SpreadsheetFile.exportXlsx(wb)).save(book);
  await fs.writeFile(path.join(out,'builder_checks.json'),JSON.stringify({status:'exported_needs_saved_file_and_visual_QA',file:path.basename(book),sha256:sha(book),input_sha256:sha(input),source_files_intact:true,baseline_trials:36,LG_rows_first:true,ABO_included:false,formula_perturbations_restored:true,common_token_ratio_equals_time_ratio:true,component_energy_is_proxy_not_metered:true,collector_source_sha256:sha(path.join(HERE,'collect_lgvq_reorganized_20261030.py'))},null,2),{flag:'wx'});
  console.log(JSON.stringify({file:book,sha256:sha(book),LG_lab_ms:d.LG_lab.time16_s*1000,LG80_parallel_ms:d.LG_rental.time16_s*1000,ABO_included:false}));
}
