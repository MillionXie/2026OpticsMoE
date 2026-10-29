/** New immutable 36-trial timing workbook. No model imports or GPU work. */
import fs from 'node:fs/promises';
import fss from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';

const HERE=path.dirname(fileURLToPath(import.meta.url));
const REPO=path.resolve(HERE,'../../../..');
const runtime=process.env.ARTIFACT_RUNTIME_DIR;
if(!runtime)throw Error('ARTIFACT_RUNTIME_DIR must point to the bundled dependency junction');
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(createRequire(path.join(runtime,'loader.cjs')).resolve('@oai/artifact-tool')).href);
const OUT=path.join(HERE,'outputs/01a183c0-ff52-76d3-bd5b-5763b23bfb5b/Lab8_Rental4_timing_20261029_01');
const BOOK='A100_Lab8_Rental4_full_input_20261029.xlsx';
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
const rentalRoot=path.join(base,'rental_timing_only_20261029_02');
const rootReceipt=path.join(rentalRoot,'whole_root_backup_receipt_complete01.json');
assert(sha(rootReceipt)==='9394417869da90fd83f24348a03b62bcb2d19631b6a2eee975dd06274774bac4','Sealed whole rental root');
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
function addTrial(file,energy){
  const r=read(file),p=path.dirname(file),c=r.cost;
  assert(r.status==='complete'&&r.measured_videos===512&&r.batch===1&&r.explicit_warmup_forwards===0,'Exact cold B1 512 trial');
  assert(r.visible_multiprocessors===(r.cap===50?54:108),'Actual MPS SM count');
  assert(jsonAudit(p,'cpu_audit.json').status==='passed','Server CPU audit');
  let auditFile=path.join(p,'local_independent_fullscope_audit01.json');
  if(fss.existsSync(auditFile))assert(read(auditFile).status==='passed','Existing independent audit');
  else{
    assert(r.model==='qwen'&&r.cap===50&&path.basename(path.dirname(p))==='qwen_fullscope_20261029_02','Only sealed Qwen50 uses pooled independent audit');
    auditFile=path.join(path.dirname(p),'local_independent_pooled_audit01.json');
    const audit=read(auditFile);assert(audit.status==='passed'&&audit.round_audits.some(a=>a.status==='passed'&&a.report_sha256===sha(file)),'Exact existing pooled round audit');
  }
  if(energy){assert(c.power_integration.max_power_sample_age_s<=0.25,'Unchanged 250ms age gate');assert(r.power_limit_w===250,'Lab cap');}
  else{assert(r.energy_qualified===false&&c.energy_qualified===false&&r.power_limit_w===400,'Rental timing-only, not energy');assert(fss.existsSync(path.join(p,'lab_backup_receipt_complete01.json')),'Lab receipt');}
  const key=r.model+'_'+r.cap;
  const t={key,model:r.model,cap:r.cap,hardware:energy?'A100 40GB / 250W':'A100 80GB / 400W',repeat:r.repeat,videos:512,groups:32,seconds:c.continuous_wall_s,tokens:r.effective_input_tokens512,gpu_j:energy?c.gpu_energy_j:null,system_j:energy?c.estimated_system_energy_j:null,energy,report_sha256:sha(file),source:rel(file),audit_sha256:sha(auditFile)};
  assert(t.seconds>0&&Number.isInteger(t.tokens)&&t.tokens>=0,'Native numerator');trials.push(t);return t;
}
function jsonAudit(p,n){return read(path.join(p,n));}
for(const c of lab.records){
  const a=c.round_report_SHA256.map(pin=>{assert(candidates.has(pin),'Pinned lab report is installed');return addTrial(candidates.get(pin),true);});
  close(a.reduce((s,t)=>s+t.seconds,0)/96,c.time16_s);
  close(a.reduce((s,t)=>s+t.tokens,0),c.native_effective_input_tokens);
  configs.push({key:a[0].key,model:c.model,cap:c.MPS_percent,hardware:a[0].hardware,energy:true,trials:a});
}
for(const model of ['kimi','deepseek'])for(const cap of [50,100]){
  const a=[1,2,3].map(n=>addTrial(path.join(rentalRoot,`${model}_cap${cap}_repeat0${n}/report.json`),false));
  configs.push({key:a[0].key,model,cap,hardware:a[0].hardware,energy:false,trials:a});
}
assert(trials.length===36&&new Set(trials.map(t=>t.report_sha256)).size===36,'Exactly 36 distinct accepted trials');
const aggregate=c=>{const sec=c.trials.reduce((s,t)=>s+t.seconds,0),tok=c.trials.reduce((s,t)=>s+t.tokens,0);return {key:c.key,model:c.model,cap:c.cap,hardware:c.hardware,time16_s:sec/96,native_tokens:tok,native_tokens_per_s:tok?tok/sec:null,energy16_j:c.energy?c.trials.reduce((s,t)=>s+t.system_j,0)/96:null,gpu_mean_w:c.energy?c.trials.reduce((s,t)=>s+t.gpu_j,0)/sec:null,energy_qualified:c.energy,report_sha256:c.trials.map(t=>t.report_sha256)};};
const summary=configs.map(aggregate);
await fs.mkdir(OUT,{recursive:true});
const wb=Workbook.create(),s=wb.worksheets.add('Comparison'),raw=wb.worksheets.add('Trials');
for(const sheet of [s,raw]){sheet.showGridLines=false;sheet.getRange(sheet===s?'A1:I31':'A1:O42').format.font={name:'Arial',size:10};}
s.tabColor='#263C5A';
s.getRange('A2').values=[['A100 full-input timing comparison']];s.getRange('A2:I2').format.font={name:'Arial',size:15,bold:true};
s.getRange('A3').values=[['Three fresh rounds per configuration. All rounds retained. 512 videos per round.']];
const headers=['Model','MPS (%)','Hardware','Time / 16 videos (ms)','Native tokens / s','Energy / 16 videos (J)','GPU mean (W)','Time / LG equivalent','LG rate / native rate'];
s.getRange('A5:I5').values=[headers];
raw.getRange('A1:O1').values=[['Config','Model','MPS (%)','Hardware','Repeat','Videos','Groups of 16','Continuous wall (s)','Native input tokens','GPU energy (J)','System proxy energy (J)','Energy qualified','Report SHA256','Original source','Independent audit SHA256']];
raw.getRange('A2:O37').values=trials.map(t=>[t.key,display[t.model],t.cap,t.hardware,t.repeat,t.videos,t.groups,t.seconds,t.tokens,t.gpu_j,t.system_j,t.energy,t.report_sha256,t.source,t.audit_sha256]);
for(const [i,c] of configs.entries()){
  const row=6+i+(i>=8?2:0),key=JSON.stringify(c.key),sum=col=>`SUMIFS('Trials'!$${col}$2:$${col}$37,'Trials'!$A$2:$A$37,${key})`;
  s.getRange(`A${row}:C${row}`).values=[[display[c.model],c.cap,c.hardware]];
  s.getRange(`D${row}`).formulas=[[`=${sum('H')}/${sum('G')}*1000`]];
  s.getRange(`E${row}`).formulas=[[`=IF(${sum('I')}=0,"N/A",${sum('I')}/${sum('H')})`]];
  s.getRange(`F${row}`).formulas=[[c.energy?`=${sum('K')}/${sum('G')}`:'="N/A"']];
  s.getRange(`G${row}`).formulas=[[c.energy?`=${sum('J')}/${sum('H')}`:'="N/A"']];
  s.getRange(`H${row}`).formulas=[[`=D${row}/$D$23`]];
  s.getRange(`I${row}`).formulas=[[`=IF(E${row}="N/A","N/A",$E$23/E${row})`]];
}
s.getRange('A15').values=[['New rental timing-only results']];s.getRange('A15:I15').format.font.bold=true;
s.getRange('A22:I22').values=[headers];s.getRange('A23:C23').values=[['LightGen-II equivalent',null,'Operator scheduling']];
s.getRange('D23:G23').values=[[lg.LG_time16_s*1000,lg.LG_tokens_per_s,'N/A','N/A']];
s.getRange('A25').values=[['LG includes resize and normalization, but excludes file decoding, RGB/crop and quality-pixel preparation.']];
s.getRange('A26').values=[['Ratios compare different scopes and hardware. They are not same-scope physical end-to-end speedups.']];
s.getRange('A27').values=[['Lab energy: NVML board integral plus the fixed 338.2 W host proxy once. Not a wattmeter total.']];
s.getRange('A28').values=[['Rental and LG energy are unqualified: N/A. Native tokens use each baseline’s own numerator.']];
s.getRange('A29').values=[['LG alone shares Qwen input-token counts. Model loading is outside each baseline timing window.']];
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
}
const original=raw.getRange('H2').values[0][0],old=s.getRange('D6').values[0][0];raw.getRange('H2').values=[[original+1]];wb.recalculate();
close(s.getRange('D6').values[0][0],old+1000/96);raw.getRange('H2').values=[[original]];wb.recalculate();close(s.getRange('D6').values[0][0],old);
console.log((await wb.inspect({kind:'table',range:'Comparison!A5:I19',include:'values,formulas',tableMaxRows:15,tableMaxCols:9,maxChars:3500})).ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},maxChars:1500});console.log(errors.ndjson);
assert(!/#REF!|#DIV\/0!|#VALUE!|#NAME\?|#NUM!|#NULL!|#SPILL!|#CALC!/.test(errors.ndjson),'Formula error scan');
for(const [sheetName,range,name] of [['Comparison','A1:I29','comparison_preview.png'],['Trials','A1:L13','trials_preview.png']]){
  const blob=await wb.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(OUT,name),new Uint8Array(await blob.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(OUT,BOOK));
const result={status:'36_ACCEPTED_TRIALS_12_CONFIGS_WITH_DISTINCT_HARDWARE_AND_ENERGY_SCOPE',LG_reference:{file:rel(lgFile),sha256:sha(lgFile),time16_ms:lg.LG_time16_s*1000,native_tokens_per_s:lg.LG_tokens_per_s,energy_qualified:false},records:summary,trials};
await fs.writeFile(path.join(OUT,'combined_summary.json'),JSON.stringify(result,null,2),{flag:repairDraft?'w':'wx'});
await fs.writeFile(path.join(OUT,'workbook_receipt.json'),JSON.stringify({file:BOOK,sha256:sha(path.join(OUT,BOOK)),bytes:fss.statSync(path.join(OUT,BOOK)).size,tests:{trials:36,configs:12,formula_input_perturbation_and_restore:true,all_summary_time_formulas_match:true},source_pins:{lab:sha(labFile),lg:sha(lgFile),rental_root:sha(rootReceipt)}},null,2),{flag:repairDraft?'w':'wx'});
console.log(JSON.stringify({status:'exported_needs_visual_review_and_lab_backup',output:path.join(OUT,BOOK),sha256:sha(path.join(OUT,BOOK))}));
