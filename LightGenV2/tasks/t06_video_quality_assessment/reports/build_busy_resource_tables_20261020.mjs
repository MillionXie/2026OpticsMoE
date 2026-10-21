import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require=createRequire(path.join(process.cwd(),'.codex_tmp/artifact-runtime.cjs'));
const {Workbook,SpreadsheetFile,FileBlob}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
if(process.argv.includes('--temporal-fp32-shared-accuracy')){await editTemporalFp32SharedAccuracy();process.exit(0);}
if(process.argv.includes('--temporal-fp32-reference')){await buildTemporalFp32Reference();process.exit(0);}
if(process.argv.includes('--dataset-once')){await buildDatasetOnce();process.exit(0);}
if(process.argv.includes('--mps-deployed')){await buildMpsDeployed();process.exit(0);}
const repo=process.cwd(),originalRoot=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_busy_resource_20261020/presentation');
const root=process.env.LGVQ_STRICT_REPORT_ROOT?path.resolve(process.env.LGVQ_STRICT_REPORT_ROOT):originalRoot;
const data=JSON.parse(await fs.readFile(path.join(root,'eight_tables.json'),'utf8'));
const strict=data.timing_protocol==='per_forward_first_block_plus_internal_wait_v1';
const complete=strict&&data.status==='all48_complete';
if(strict&&data.accounting_protocol!=='per_forward_tf_plus_output_to_next_input_v2')throw Error('Latest intercall-gap accounting was not audited');
const preview=process.env.LGVQ_RESOURCE_PREVIEW==='1';
const out=preview?path.join(repo,'.codex_tmp/strict_resource_preview.xlsx'):path.join(repo,'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013/LGVQ_busy_resource_precision_comparison.xlsx');
const receiptName=preview?'workbook_preview_receipt.json':'workbook_receipt.json';
try{
 const existing=await fs.readFile(out);let receipt;
 try{receipt=JSON.parse(await fs.readFile(path.join(root,receiptName),'utf8'));}
 catch(e){if(e.code!=='ENOENT')throw e;receipt=JSON.parse(await fs.readFile(path.join(originalRoot,'workbook_receipt.json'),'utf8'));}
 const digest=crypto.createHash('sha256').update(existing).digest('hex');
 if(digest!==receipt.sha256)throw Error('Workbook changed outside this build; preserve user edits');
 if(strict){const archive=path.join(root,`prior_workbook_${digest}.xlsx`);try{await fs.copyFile(out,archive,1);}catch(e){if(e.code!=='EEXIST')throw e;if(crypto.createHash('sha256').update(await fs.readFile(archive)).digest('hex')!==digest)throw Error('Archive differs');}}
}catch(e){if(e.code!=='ENOENT')throw e;}
const book=Workbook.create(),views=[],models=['deepseek','kimi','qwen','glm','clip','yolo'];
const colors={deepseek:'#79502C',kimi:'#D14F64',qwen:'#0072B2',glm:'#9B567C',clip:'#008B70',yolo:'#B17D00'};
const parameterLabel=p=>!p?.model_parameters?'':`${p.parameter_count_is_nominal?'nominal ':''}${p.model_parameters>=1e9?(p.model_parameters/1e9).toFixed(2)+'B':(p.model_parameters/1e6).toFixed(2)+'M'} parameters`;
const unavailable=strict?'No qualified laboratory measurement':data.laboratory_status==='waiting_expired'?'original A100 unavailable':'pending measurement';
const missingFor=(model,precision)=>strict?(['deepseek','kimi'].includes(model)?'pending rental condition':unavailable):model==='deepseek'&&precision==='fp32'?'capacity exceeds80GB':unavailable;
for(const method of ['temporal','mps'])for(const precision of ['native','fp32'])for(const metric of ['time','energy']){
 const name=`${method==='temporal'?'Duty':'MPS'}_${precision==='native'?'Native':'FP32'}_${metric==='time'?'Time':'Energy'}`;
 views.push({method,precision,metric,sheet:book.worksheets.add(name)});
}

async function buildDatasetOnce(){
 const repo=process.cwd();
 const combined=process.argv.includes('--combined');
 const paper=process.argv.includes('--finalized-qwen');if(paper&&!combined)throw Error('Use combined data with finalized Qwen');
 const labRoot=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_mps_dataset_once_a100_20261020/presentation');
 const root=combined?path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_mps_dataset_once_rental_20261021_replacement02/presentation'):labRoot;
 const source=path.join(root,paper?'dataset_once_paper.json':combined?'dataset_once_combined.json':'dataset_once_audited.json'),data=JSON.parse(await fs.readFile(source,'utf8'));
 const points=combined?12:8,trials=points*3,last=8+points,lastTrial=1+trials,opticalRow=lastTrial+4,notesRow=last+4;
 if(data.status!=='complete'||data.points.length!==points||data.repeats.length!==trials)throw Error('All audited distinct-video repeats are required');
 const outdir=path.join(repo,'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013');
 const output=path.join(outdir,'LGVQ_MPS_dataset_once_comparison.xlsx');
 const opticalSource=path.join(root,'../../lgvq_strict_tf_rental_dual_20261020/presentation/eight_tables.json');
 const optical=JSON.parse(await fs.readFile(opticalSource,'utf8')).optical;
 if(optical.time_ms!==10.245569591370225||optical.energy_j!==1.1776895097085185||optical.srcc!==.7977138739)throw Error('Prior optical reference changed');
 const receiptPath=path.join(root,'dataset_once_workbook_receipt.json');
 if(!paper){try{if(JSON.parse(await fs.readFile(receiptPath,'utf8')).finalized_qwen_cost)throw Error('Do not undo the finalized Qwen presentation');}catch(e){if(e.code!=='ENOENT')throw e;}}
 try{const prior=await fs.readFile(output);let receipt;try{receipt=JSON.parse(await fs.readFile(receiptPath,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;receipt=JSON.parse(await fs.readFile(path.join(labRoot,'dataset_once_workbook_receipt.json'),'utf8'));}const digest=crypto.createHash('sha256').update(prior).digest('hex');if(digest!==receipt.sha256)throw Error('Workbook changed externally; preserve edits');try{await fs.copyFile(output,path.join(root,'prior_workbook_'+digest+'.xlsx'),1);}catch(e){if(e.code!=='EEXIST')throw e;}}
 catch(e){if(e.code!=='ENOENT')throw e;try{await fs.access(output);throw Error('Unidentified existing workbook');}catch(x){if(x.code!=='ENOENT')throw x;}}
 let book,imported=true;try{book=await SpreadsheetFile.importXlsx(await FileBlob.load(output));}catch(e){if(e.code!=='ENOENT')throw e;book=Workbook.create();imported=false;}
 const s=imported?book.worksheets.getItem('Plot data'):book.worksheets.add('Plot data'),t=imported?book.worksheets.getItem('Trial data'):book.worksheets.add('Trial data');
 if(imported)for(const sh of [s,t])sh.getUsedRange().clear({applyTo:'contents'});
 if(combined){t.getRange('A28:D29').clear({applyTo:'formats'});t.getRange('P28:Q29').clear({applyTo:'formats'});}
 const names={qwen:'Qwen3-VL-2B',glm:'GLM-4.6V-Flash',clip:'CLIP ViT-B/32',yolo:'YOLO11s',deepseek:'DeepSeek-VL2',kimi:'Kimi-VL-A3B'};
 const colors={qwen:'#0072B2',glm:'#92527F',clip:'#007F63',yolo:'#9B7000',deepseek:'#79502C',kimi:'#D14F64'};
 for(const sh of [s,t]){sh.showGridLines=false;sh.getRange('A1:R40').format.font={name:'Arial',size:10,color:'#263747'};sh.getRange('A1:R40').format.verticalAlignment='center';}
 s.tabColor='#214D68';s.getRange('A2').values=[['LGVQ temporal quality: cold, distinct-video MPS comparison']];s.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#214D68'};
 s.getRange('A4').values=[['B=1. Three fresh processes. No inference warmup. First 200 distinct videos for costs; 558 for SRCC.']];
 s.getRange('A5').values=[['First native block→GPU MOS plus output→next preprocessing start. Each preprocessing interval excluded.']];
 s.getRange('A7:K7').values=[['Model','MPS allocation','Mean TEST SRCC','Time /16 (ms)','Energy /16 (J)','Time ratio vs optical','Energy ratio vs optical','Block only /16 (ms)','Internal gap /16 (ms)','Repeats','Deployment precision']];
 s.getRange('A7:K7').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:42,wrapText:true,horizontalAlignment:'center'};
 s.getRange('A8:K8').values=[['LightGen-II',null,.7977138739,10.245569591370225,1.1776895097085185,1,1,null,null,null,'Prior optical component reference']];
 s.getRange('A8:K8').format.fill='#EEE6F4';s.getRange('A8').format.font={name:'Arial',size:10,bold:true,color:'#7828A8'};
 const headers=['Model','MPS allocation','Repeat','Video batch','Native-block seconds','Nonpreprocessing gap seconds','Integrated GPU energy (J)','Host proxy (W)','Cost videos','558TEST SRCC','558TEST PLCC','Time /16 (ms)','Energy /16 (J)','Visible SMs','Wall-window busy (%)','Report source','Report SHA256','Identity SHA256','Parameter dtypes','Readout dtypes','Checkpoint SHA256','Completed (+08:00)','Source commit'];
 t.getRange('A1:W1').values=[headers];t.getRange('A1:W1').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:42,wrapText:true,horizontalAlignment:'center'};
 let tr=2;const bindings=[];
 for(const [i,p]of data.points.entries()){
  const indices=[];
  for(const r of p.repeats){
   // Excel dates have no timezone. Preserve the Beijing wall-clock tuple.
   const completedBeijing=new Date(r.completed_at_beijing.replace('+08:00','Z'));
   t.getRange(`A${tr}:W${tr}`).values=[[names[p.model],p.resource_cap/100,r.repeat,1,r.compute_only_time16_ms/80,r.nonpreprocessing_gap16_ms/80,r.gpu_energy_j,338.2,200,r.srcc,r.plcc,null,null,r.visible_sms,r.actual_busy_percent/100,r.source,r.source_sha256,r.identity_sha256,r.parameter_dtypes.join(', '),r.readout_dtypes.join(', '),r.checkpoint_sha256,completedBeijing,r.source_commit]];
   t.getRange(`L${tr}:M${tr}`).formulas=[[`=(E${tr}+F${tr})*16000/I${tr}`,`=(H${tr}*(E${tr}+F${tr})+G${tr})*16/I${tr}`]];
   t.getRange(`A${tr}:U${tr}`).format.rowHeight=25;t.getRange(`A${tr}`).format.font={name:'Arial',size:10,bold:true,color:colors[p.model]};indices.push(tr++);
  }
  const r=9+i,cols=c=>indices.map(n=>`'Trial data'!${c}${n}`).join(',');
  s.getRange(`A${r}:K${r}`).values=[[names[p.model],p.resource_cap/100,null,null,null,null,null,null,null,3,['qwen','glm','deepseek','kimi'].includes(p.model)?'BF16; FP32 readout':p.model==='clip'?'FP16 autocast; FP32 readout (mixed parameters)':'FP16 autocast; FP32 parameters/readout']];
  s.getRange(`C${r}:I${r}`).formulas=[[`=AVERAGE(${cols('J')})`,`=(SUM(${cols('E')})+SUM(${cols('F')}))*16000/SUM(${cols('I')})`,`=SUMPRODUCT('Trial data'!M${indices[0]}:M${indices[2]},'Trial data'!I${indices[0]}:I${indices[2]})/SUM(${cols('I')})`,`=D${r}/$D$8`,`=E${r}/$E$8`,`=SUM(${cols('E')})*16000/SUM(${cols('I')})`,`=SUM(${cols('F')})*16000/SUM(${cols('I')})`]];
  s.getRange(`A${r}:K${r}`).format={fill:Math.floor(i/2)%2===0?'#F1F5F8':'#FFFFFF',rowHeight:34};s.getRange(`A${r}`).format.font={name:'Arial',size:10,bold:true,color:colors[p.model]};bindings.push({p,r,indices});
 }
 if(data.accuracy_reference_policy){
  s.getRange('C7').values=[['Reference SRCC (MPS100)']];
  for(const b of bindings){const ref=bindings.find(x=>x.p.model===b.p.model&&x.p.resource_cap===100);s.getRange('C'+b.r).formulas=[['=AVERAGE('+ref.indices.map(i=>`'Trial data'!J${i}`).join(',')+')']];}
 }
 const notes=[
  'MPS 50/100% are configured resource allocations, not measured 50/100% GPU busy time.',
  paper?'Displayed SRCC uses each model MPS100 full558 reference for both settings. Trial data retains measured50 scores.':'Each cap has three full 558-video evaluations. No copied accuracy score or highest-repeat selection.',
  'All 200 cold videos and 199 internal gaps are retained. Terminal post-output handling is excluded.',
  'Energy integrates raw board power over the same charged intervals. The 338.2 W host proxy is added once.',
  'Legacy first-block hook is unchanged; pending asynchronous frontend work is not newly drained.',
  combined?'DS/Kimi: rented A100-SXM4-80GB /400W. Other models: laboratory A100-PCIE-40GB /250W.':'DS/Kimi resident-repeat observations are excluded from this new distinct-video comparison.',
  'Optical values retain the prior component-cost reference; they are not a new matched wall-clock run.',
  paper?'† Qwen100 uses finalized historical cost. New three-trial costs remain in Trial data; no same-run claim.':'Historical Qwen B1: 1188.386146 ms, 462.113401 J, SRCC 0.7663428037. Preserved; not plotted as this run.',
  'Raw wall-window busy includes excluded preprocessing. Board energy uses driver-averaged NVML samples.',
  'The timer starts at the first native visual Transformer block, except YOLO (first convolution).',
  'Frozen preselected heads. TEST development metrics, no validation split, no new checkpoint selection.'
 ];
 notes.forEach((text,i)=>{s.getRange('A'+(notesRow+i)).values=[[text]];s.getRange(`A${notesRow+i}:K${notesRow+i}`).format.rowHeight=23;});
 s.getRange('A:A').format.columnWidth=26;s.getRange('B:J').format.columnWidth=18;s.getRange('K:K').format.columnWidth=43;
 s.getRange(`K9:K${last}`).format.wrapText=true;
 s.getRange(`J9:J${last}`).format.horizontalAlignment='center';
 s.getRange(`B9:B${last}`).setNumberFormat('0%');s.getRange(`C8:C${last}`).setNumberFormat('0.000000');s.getRange(`D8:E${last}`).setNumberFormat('#,##0.000');s.getRange(`F8:G${last}`).setNumberFormat('0.00"×"');s.getRange(`H9:I${last}`).setNumberFormat('0.000');s.getRange(`J9:J${last}`).setNumberFormat('0');
 t.getRange('A:A').format.columnWidth=25;t.getRange('B:O').format.columnWidth=20;t.getRange('P:W').format.columnWidth=78;t.getRange(`P2:W${lastTrial}`).format.wrapText=true;t.getRange(`P2:W${lastTrial}`).format.rowHeight=54;
 t.getRange(`S2:W${opticalRow}`).format.font={name:'Arial',size:10,color:'#263747'};
 t.getRange(`A1:W${opticalRow}`).format.verticalAlignment='center';t.getRange(`V2:V${lastTrial}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
 t.getRange(`E2:M${lastTrial}`).setNumberFormat('0.000000');for(const c of ['C','D','I','N'])t.getRange(`${c}2:${c}${lastTrial}`).setNumberFormat('0');for(const c of ['B','O'])t.getRange(`${c}2:${c}${lastTrial}`).setNumberFormat('0.00%');t.freezePanes.freezeRows(1);t.freezePanes.freezeColumns(2);
 t.getRange(`A${opticalRow-1}:D${opticalRow}`).values=[['Reference','Time /16 (ms)','Energy /16 (J)','SRCC'],['LightGen-II',optical.time_ms,optical.energy_j,optical.srcc]];
 t.getRange(`P${opticalRow-1}:Q${opticalRow}`).values=[['Optical reference source','Optical reference SHA256'],[path.relative(repo,opticalSource).replaceAll('\\','/'),crypto.createHash('sha256').update(await fs.readFile(opticalSource)).digest('hex')]];
 for(const range of [`A${opticalRow-1}:D${opticalRow-1}`,`P${opticalRow-1}:Q${opticalRow-1}`])t.getRange(range).format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:32,wrapText:true};
 t.getRange(`P${opticalRow}:Q${opticalRow}`).format={wrapText:true,rowHeight:54};t.getRange(`B${opticalRow}:D${opticalRow}`).setNumberFormat('0.000000');
 s.getRange('C8:E8').formulas=[[`='Trial data'!D${opticalRow}`,`='Trial data'!B${opticalRow}`,`='Trial data'!C${opticalRow}`]];
 if(paper){
  const p=data.points.find(p=>p.model==='qwen'&&p.resource_cap===100),ar=opticalRow+4;
  if(!data.presentation_override||p.time16_ms!==1188.3861464262009||p.energy16_j!==462.1134010767563)throw Error('Historical Qwen cost is not pinned');
  if(crypto.createHash('sha256').update(await fs.readFile(path.join(repo,p.cost_source))).digest('hex')!==p.cost_source_sha256)throw Error('Qwen historical source changed');
  t.getRange(`A${ar-1}:D${ar}`).values=[['Historical cost reference','Time /16 (ms)','Energy /16 (J)','SRCC reference'],['Qwen3-VL-2B finalized',p.time16_ms,p.energy16_j,.7663428036522781]];
  t.getRange(`P${ar-1}:R${ar}`).values=[['Historical source','Historical source SHA256','Source selector'],[p.cost_source,p.cost_source_sha256,p.cost_source_selector]];
  for(const rg of [`A${ar-1}:D${ar-1}`,`P${ar-1}:R${ar-1}`])t.getRange(rg).format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:42,wrapText:true};
  t.getRange(`P${ar}:R${ar}`).format={wrapText:true,rowHeight:58};t.getRange(`B${ar}:D${ar}`).setNumberFormat('0.000000');
  for(const rg of [`A${ar}:D${ar}`,`P${ar}:R${ar}`])t.getRange(rg).format.font={name:'Arial',size:10,color:'#263747'};
  s.getRange('A9').values=[['Qwen3-VL-2B †']];s.getRange('J7').values=[['Accuracy repeats']];
  s.getRange('D9:E9').formulas=[[`='Trial data'!B${ar}`,`='Trial data'!C${ar}`]];
  s.getRange('H9:I9').values=[[null,null]];
 }
 function verify(w){const ss=w.worksheets.getItem('Plot data');for(const {p,r}of bindings)for(const [c,k]of [['C','srcc'],['D','time16_ms'],['E','energy16_j']])if(Math.abs(ss.getRange(c+r).values[0][0]-p[k])>1e-7)throw Error('Mismatch '+c+r);}
 book.recalculate();verify(book);const probe=t.getRange(paper?'E5':'E2'),probeRow=paper?'D10':'D9',baseline=s.getRange(probeRow).values[0][0],before=probe.values[0][0];probe.values=[[before*1.01]];book.recalculate();if(s.getRange(probeRow).values[0][0]===baseline)throw Error('Recalculation failed');probe.values=[[before]];book.recalculate();verify(book);
 const errors=(await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1200})).ndjson;if(!errors.includes('Cell search matched 0 entries.'))throw Error(errors);
 console.log((await book.inspect({kind:'table',range:'Plot data!A7:K10',include:'values,formulas',tableMaxRows:4,tableMaxCols:11,maxChars:2300})).ndjson);
 for(const [sheetName,range,name]of [['Plot data',`A1:K${notesRow+10}`,'dataset_once_summary'],['Trial data','A1:O8','dataset_once_trials'],['Trial data','P1:Q4','dataset_once_sources'],['Trial data','R1:U4','dataset_once_identities'],['Trial data','V1:W4','dataset_once_dates'],['Trial data',`A${opticalRow-1}:D${opticalRow}`,'dataset_once_optical'],['Trial data',`P${opticalRow-1}:Q${opticalRow}`,'dataset_once_optical_source'],...(combined?[['Trial data','A26:O37','dataset_once_rental_trials'],['Trial data','P26:W28','dataset_once_rental_sources']]:[])]){if(process.argv.includes('--format-fix')&&!['dataset_once_summary','dataset_once_identities','dataset_once_dates'].includes(name))continue;const blob=await book.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(repo,'.codex_tmp',name+'.png'),new Uint8Array(await blob.arrayBuffer()));}
 if(paper){for(const [range,name]of [[`A${opticalRow+3}:D${opticalRow+4}`,'dataset_once_qwen_anchor'],[`P${opticalRow+3}:R${opticalRow+4}`,'dataset_once_qwen_anchor_source']]){const blob=await book.render({sheetName:'Trial data',range,scale:1,format:'png'});await fs.writeFile(path.join(repo,'.codex_tmp',name+'.png'),new Uint8Array(await blob.arrayBuffer()));}}
 await(await SpreadsheetFile.exportXlsx(book)).save(output);verify(await SpreadsheetFile.importXlsx(await FileBlob.load(output)));
 await fs.writeFile(receiptPath,JSON.stringify({file:output,sha256:crypto.createHash('sha256').update(await fs.readFile(output)).digest('hex'),source_sha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex'),conditions:points,repeats:trials,finalized_qwen_cost:paper,recalculation:true,reimport:true,formula_errors:0},null,2)+'\n');
 try{await fs.rename(output+'.inspect.ndjson',path.join(root,'dataset_once_inspect.ndjson'));}catch(e){if(e.code!=='ENOENT')throw e;}
 console.log(JSON.stringify({file:output,conditions:points,repeats:trials}));
}
const raw=book.worksheets.add('Measurements');
raw.showGridLines=false;
const windows=book.worksheets.add('Window repeats');windows.showGridLines=false;
const windowHeaders=['Model','Method','Precision','Condition','Video batch','MPS cap (%)','Visible SMs','Repeat','Elapsed seconds','Video evaluations','Mean board power W','GPU busy (%)','Telemetry samples','First TF monotonic s','Final output monotonic s','Unique videos','Reported time /16 ms','Reported energy /16 J','Report source','Report SHA256','Point acquired ISO8601 (+08:00)','558TEST SRCC development reference','Accuracy reference source','Accuracy report SHA256'];
if(strict){windowHeaders[8]='Charged block + intercall seconds';windowHeaders[10]='Integrated used-GPU energy J';windowHeaders[11]='Fixed hot-card busy (%)';windowHeaders[13]='First block monotonic s';windowHeaders[16]='Recomputed time /16 ms';windowHeaders[17]='Recomputed energy /16 J';}
windows.getRange('A1:X1').values=[windowHeaders];windows.getRange('A1:X1').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:45,horizontalAlignment:'center'};
windows.getRange('A:X').format.columnWidth=18;windows.getRange('A:D').format.columnWidth=25;windows.getRange('S:U').format.columnWidth=65;windows.getRange('W:X').format.columnWidth=65;
windows.freezePanes.freezeRows(1);windows.freezePanes.freezeColumns(4);
if(strict){windows.getRange('Y1:AK1').values=[['GPU0 busy','GPU1 busy','Full-wall seconds','Excluded preblock seconds','Internal wait seconds','TF compute seconds','Hot-card index','Prior TF+wait time /16 ms','Prior TF+wait energy /16 J','Output to next model input s','Output to next TF s','In-cycle preblock s','Full-cycle span s']];windows.getRange('Y1:AK1').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:45,horizontalAlignment:'center'};windows.getRange('Y:AK').format.columnWidth=20;}
if(strict){windows.getRange('AL1:AO1').values=[['GPU0 mean board memory (MiB)','GPU0 peak board memory (MiB)','GPU1 mean board memory (MiB)','GPU1 peak board memory (MiB)']];windows.getRange('AL1:AO1').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:45,horizontalAlignment:'center'};windows.getRange('AL:AO').format.columnWidth=25;}
const windowRows=new Map();let wr=2;
const key=p=>[p.model,p.method,p.precision,p.condition].join('/');
for(const point of data.values){
 const rr=[];for(const r of point.repeats){
  windows.getRange(`A${wr}:X${wr}`).values=[[point.model_name,point.method,point.precision,point.condition,point.batch,point.resource_cap,point.visible_sms,r.repeat,r.elapsed_s,r.video_evaluations,strict?r.gpu_energy_j:r.measurement.mean_board_power_w,r.measurement.mean_gpu_utilization_percent,r.measurement.samples,r.first_tf_start,r.final_gpu_output,r.unique_videos,r.time16_ms,r.energy16_j,point.source,point.source_sha256,point.measured_at,point.srcc_reference??null,point.accuracy_source??'',point.accuracy_source_sha256??'']];
  if(strict){windows.getRange(`Y${wr}:AK${wr}`).values=[[r.gpu_infos[0].mean_gpu_utilization_percent/100,r.gpu_infos[1]?r.gpu_infos[1].mean_gpu_utilization_percent/100:null,r.wall_elapsed_s,r.excluded_preblock_s,r.internal_wait_s,r.compute_s,point.hot_gpu_index,r.recorded_time16_ms,r.recorded_energy16_j,r.output_to_next_model_input_s,r.output_to_next_tf_s,r.in_cycle_preblock_s,r.full_cycle_span_s]];windows.getRange(`Y${wr}:Z${wr}`).setNumberFormat('0.00%');}
  if(strict){windows.getRange(`AL${wr}:AO${wr}`).values=[[r.gpu_infos[0].mean_memory_used_mib,r.gpu_infos[0].peak_memory_used_mib,r.gpu_infos[1]?.mean_memory_used_mib??null,r.gpu_infos[1]?.peak_memory_used_mib??null]];windows.getRange(`AL${wr}:AO${wr}`).format={font:{name:'Arial',size:10,color:'#243447'},verticalAlignment:'center'};windows.getRange(`AL${wr}:AO${wr}`).setNumberFormat('#,##0.0');}
  windows.getRange(`A${wr}:X${wr}`).format={font:{name:'Arial',size:10,color:'#243447'},rowHeight:point.accuracy_reference_count>1?120:52,verticalAlignment:'center'};windows.getRange(`S${wr}:X${wr}`).format.wrapText=true;rr.push(wr++);
 }windowRows.set(key(point),rr);
}
if(wr>2){windows.getRange(`I2:R${wr-1}`).setNumberFormat('0.000000');for(const col of ['E','F','G','H','J','M','P'])windows.getRange(`${col}2:${col}${wr-1}`).setNumberFormat('#,##0');windows.getRange(`E2:H${wr-1}`).format.horizontalAlignment='center';}
raw.getRange('A1:AJ1').merge();raw.getRange('A1').values=[['POOLED COST CALCULATIONS — repeat observations are linked from Window repeats']];
raw.getRange('A1:AJ1').format={font:{name:'Arial',size:13,bold:true,color:'#214D68'},rowHeight:30};
const headers=['Model','Method','Precision','Condition','Video batch','MPS cap (%)','Visible SMs','558TEST SRCC development reference',
 'Repeat1 seconds','Repeat1 video evaluations','Repeat1 boardW','Repeat1 busy%','Repeat1 samples',
 'Repeat2 seconds','Repeat2 video evaluations','Repeat2 boardW','Repeat2 busy%','Repeat2 samples',
 'Repeat3 seconds','Repeat3 video evaluations','Repeat3 boardW','Repeat3 busy%','Repeat3 samples',
 'Host proxy W','Total seconds','Total video evaluations','Estimated window energy J','Pooled busy%','Equivalent16-video ms','Equivalent16-video J','Repeats','Status','Platform','Report source','Report SHA256','Accuracy/protocol note'];
if(strict){headers[8]='Repeat1 charged seconds';headers[10]='Repeat1 GPU energy J';headers[13]='Repeat2 charged seconds';headers[15]='Repeat2 GPU energy J';headers[18]='Repeat3 charged seconds';headers[20]='Repeat3 GPU energy J';headers[24]='Total charged seconds';headers[26]='Estimated interval energy J';headers[27]='Pooled hot-card busy%';}
raw.getRange('A6:AJ6').values=[headers];raw.getRange('A6:AJ6').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:48};
raw.getRange('A7:AJ55').format={font:{name:'Arial',size:10,color:'#243447'},rowHeight:26};
raw.getRange('A:AJ').format.columnWidth=18;raw.getRange('A:A').format.columnWidth=26;raw.getRange('AH:AJ').format.columnWidth=65;
raw.getRange('AF:AG').format.columnWidth=30;raw.getRange('AF7:AJ55').format.wrapText=true;raw.getRange('A7:AJ55').format.rowHeight=52;
raw.getRange('H7:AD55').setNumberFormat('0.000000');raw.freezePanes.freezeRows(6);raw.freezePanes.freezeColumns(4);
for(const column of ['E','F','G','J','M','O','R','T','W','Z','AE'])raw.getRange(`${column}7:${column}55`).setNumberFormat('#,##0');
raw.getRange('A6:AJ6').format.horizontalAlignment='center';
if(strict){raw.getRange('AK6:AO6').values=[['Frozen duty','Model parameters (basis in notes)','First compute module','Pacing calibration','Required charged seconds']];raw.getRange('AK6:AO6').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:48,horizontalAlignment:'center'};raw.getRange('AK:AL').format.columnWidth=22;raw.getRange('AL7:AL55').format.horizontalAlignment='center';raw.getRange('AM:AM').format.columnWidth=45;raw.getRange('AN:AN').format.columnWidth=38;raw.getRange('AO:AO').format.columnWidth=24;}
if(strict){raw.getRange('AP6:AR6').values=[['Power telemetry receiver','Parameter count basis','Parameter count source']];raw.getRange('AP6:AR6').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:48,horizontalAlignment:'center'};raw.getRange('AP:AP').format.columnWidth=35;raw.getRange('AQ:AQ').format.columnWidth=70;raw.getRange('AR:AR').format.columnWidth=80;raw.getRange('AP7:AR55').format={font:{name:'Arial',size:10,color:'#243447'},wrapText:true,verticalAlignment:'center'};}
raw.getRange('A7:H7').values=[['LightGen-II','optical reference','not applicable','~50% spatial use',null,null,null,data.optical.srcc]];
raw.getRange('AC7:AD7').values=[[data.optical.time_ms,data.optical.energy_j]];raw.getRange('AF7:AJ7').values=[['unchanged optical reference','Optical platform',data.optical.cost_source,'','Spatial usage is not GPU busy time or a MPS resource cap. No FP32 designation.']];
let row=8;const entries=[];
for(const method of ['temporal','mps'])for(const precision of ['native','fp32'])for(const model of models)for(const condition of method==='temporal'?['busy50','unrestricted']:['mps_cap50','mps_cap100']){
 const point=data.values.find(x=>x.method===method&&x.precision===precision&&x.model===model&&x.condition===condition);
 const missing=missingFor(model,precision);
 const values=[data.models[model],method,precision,condition,point?.batch??(method==='mps'?1:null),point?.resource_cap??null,point?.visible_sms??null,point?.srcc_reference??null];
 for(let i=0;i<3;i++){const r=point?.repeats?.[i];values.push(r?.elapsed_s??null,r?.video_evaluations??null,(strict?r?.gpu_energy_j:r?.measurement.mean_board_power_w)??null,r?.measurement.mean_gpu_utilization_percent??null,r?.measurement.samples??null);}
 values.push(338.2,null,null,null,null,null,null,point?.repeats?.length??null,point?.status??missing,point?.platform??'',point?.source??'',point?.source_sha256??'',point?.accuracy_note??(missing==='capacity exceeds80GB'?data.capacity_missing:'No completed new-protocol point. Do not substitute historical pure-inference cost.'));
 raw.getRange(`A${row}:AJ${row}`).values=[values];
 if(strict&&point)raw.getRange(`AP${row}:AR${row}`).values=[[point.telemetry_receiver??'model_process_thread_receipt_v1',point.parameter_basis??'',point.parameter_source??'']];
 if(strict&&point){raw.getRange(`AK${row}:AO${row}`).values=[[point.frozen_duty,point.model_parameters,point.first_block_module,point.pacing_calibration,point.required_charged_seconds]];raw.getRange(`AK${row}`).setNumberFormat('0.00%');raw.getRange(`AL${row}`).setNumberFormat('#,##0');raw.getRange(`AN${row}:AO${row}`).format={font:{name:'Arial',size:10,color:'#243447'},wrapText:true,verticalAlignment:'center',horizontalAlignment:'center'};raw.getRange(`AO${row}`).setNumberFormat('0');}
 if(point){
  const rr=windowRows.get(key(point));
  if(point.srcc_reference!==undefined)raw.getRange('H'+row).formulas=[[`='Window repeats'!V${rr[0]}`]];
  raw.getRange(`I${row}:W${row}`).formulas=[[...rr.flatMap(w=>['I','J','K','L','M'].map(c=>`='Window repeats'!${c}${w}`))]];
  raw.getRange(`Y${row}:AE${row}`).formulas=[[
   `=SUM(I${row},N${row},S${row})`,`=SUM(J${row},O${row},T${row})`,
   strict?`=X${row}*SUM(I${row},N${row},S${row})+SUM(K${row},P${row},U${row})`:`=(X${row}+K${row})*I${row}+(X${row}+P${row})*N${row}+(X${row}+U${row})*S${row}`,
   `=(L${row}*M${row}+Q${row}*R${row}+V${row}*W${row})/SUM(M${row},R${row},W${row})`,
   `=Y${row}*16000/Z${row}`,`=AA${row}*16/Z${row}`,`=3`]];
 }
 entries.push({method,precision,model,condition,row,point});row++;
}
for(const {method,precision,metric,sheet} of views){
 sheet.showGridLines=false;
 sheet.getRange('A:M').format.columnWidth=16;sheet.getRange('A:A').format.columnWidth=25;sheet.getRange('B:B').format.columnWidth=25;sheet.getRange('J:J').format.columnWidth=25;sheet.getRange('L:M').format.columnWidth=27;
 sheet.getRange('A1:M1').merge();sheet.getRange('A1').values=[['LGVQ temporal quality · paired GPU cost and utilization']];
 sheet.getRange('A1:M1').format={font:{name:'Arial',size:16,bold:true,color:'#214D68'},rowHeight:34};
 sheet.getRange('A2:M2').merge();sheet.getRange('A2').values=[[`${method==='mps'?'MPS execution-resource provisioning':'Temporal duty control'} / ${precision==='fp32'?'FP32':'original deployed precision'} / ${metric==='time'?'processing time':'estimated energy'}`]];
 sheet.getRange('A2:M2').format={font:{name:'Arial',size:12,color:'#53687A'},rowHeight:28};
 sheet.getRange('A4:M4').merge();sheet.getRange('A4').values=[[strict?'First native block → output + output → NEXT model input. Pacing waits are already inside gaps, not added twice. Each frontend excluded; full-wall NVML busy separate. Three trials;16-video equivalents.':'First TF input → final GPU output; internal waits included, terminal wait excluded. Prepared inputs reused; three full windows.16-video equivalents are throughput-normalized, not a16-video batch.']];
 sheet.getRange('A4:M4').format={font:{name:'Arial',size:10,color:'#53687A'},rowHeight:36,wrapText:true};
 sheet.getRange('A5:M5').merge();sheet.getRange('A5').values=[[complete?'48 prescribed conditions / 144 retained trial windows. Matching-batch full558 accuracy references verified; accuracy is not re-evaluated separately at each cap.':strict?`${data.rental_count??data.values.length}/16 rental and ${data.laboratory_count??0}/32 laboratory conditions measured. Full558 accuracy references are independent of control windows; missing cases remain unavailable.`:'PARTIAL DATA — measured values only. Independent558-video accuracy references are not fresh restricted-condition accuracy tests.']];
 sheet.getRange('A5:M5').format={font:{name:'Arial',size:10,bold:!complete,color:complete?'#53687A':'#9C3D10'},rowHeight:28,wrapText:true};
 sheet.getRange('A6:M6').values=[['Model','Setting','Video batch','Resource cap',strict?'GPU busy (max card)':'Actual GPU busy',metric==='time'?'Equivalent time (ms)':'Estimated energy (J)','Baseline / optical','SRCC (development)','Repeats','Status','Video evaluations','Precision','Platform']];
 sheet.getRange('A6:M6').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:38,wrapText:true};
 sheet.getRange('A7:M19').format={font:{name:'Arial',size:10,color:'#243447'},rowHeight:42,wrapText:true,verticalAlignment:'center'};
 sheet.getRange('A7:M7').values=[['LightGen-II','Optical reference','n.a.','n.a.','~50% spatial',null,null,null,'n.a.','unchanged reference','n.a.','n.a.','Optical platform']];
 sheet.getRange('F7:H7').formulas=[[`=Measurements!${metric==='time'?'AC':'AD'}7`,'=1','=Measurements!H7']];
 sheet.getRange('A7:M7').format.fill='#EEE6F4';sheet.getRange('A7').format.font={name:'Arial',size:10,bold:true,color:'#7A3E9D'};
 let r=8;
 for(const e of entries.filter(e=>e.method===method&&e.precision===precision)){
  const setting=method==='mps'?`${e.condition==='mps_cap50'?'50%':'100%'} resource cap`:e.condition==='busy50'?(e.point?.gpu_count===2?'~50% max-card busy':'~50% time busy'):'Unrestricted duty';
  sheet.getRange(`A${r}:M${r}`).values=[[data.models[e.model],setting,e.point?.batch??(method==='mps'?1:'n.a.'),method==='mps'?(e.condition==='mps_cap50'?.5:1):'n.a.',null,null,null,null,null,e.point?.status??missingFor(e.model,precision),null,precision==='fp32'?'FP32':`${['clip','yolo'].includes(e.model)?'FP16':'BF16'} compute + FP32 head`,e.point?.platform??'']];
  const q=e.row;
  if(strict&&e.point)sheet.getRange('A'+r).values=[[data.models[e.model]+'\n'+parameterLabel(e.point)]];
  if(e.point){sheet.getRange(`E${r}:I${r}`).formulas=[[`=Measurements!AB${q}/100`,`=Measurements!${metric==='time'?'AC':'AD'}${q}`,`=F${r}/$F$7`,e.point.srcc_reference!==undefined?`=Measurements!H${q}`:'="n.a."',`=Measurements!AE${q}`]];sheet.getRange('K'+r).formulas=[[`=Measurements!Z${q}`]];}
  else sheet.getRange(`E${r}:I${r}`).values=[['n.a.','n.a.','n.a.','n.a.','n.a.']];
  sheet.getRange(`A${r}:M${r}`).format.fill=r%2===0?'#F2F5F7':'#FFFFFF';
  sheet.getRange('A'+r).format.font={name:'Arial',size:10,bold:true,color:colors[e.model]};r++;
 }
 sheet.getRange('D8:E19').setNumberFormat('0.0%');sheet.getRange('F7:F19').setNumberFormat('#,##0.000');sheet.getRange('G7:G19').setNumberFormat('0.00"×"');sheet.getRange('H7:H19').setNumberFormat('0.000000');sheet.freezePanes.freezeRows(6);sheet.freezePanes.freezeColumns(2);
 sheet.getRange('A6:M6').format.horizontalAlignment='center';
 sheet.getRange('C7:D19').format.horizontalAlignment='center';sheet.getRange('I7:I19').format.horizontalAlignment='center';
 sheet.getRange('K7:K19').setNumberFormat('#,##0');sheet.getRange('K7:L19').format.horizontalAlignment='center';
 const notes=[['Control',method==='mps'?'Resource cap is the nominal MPS thread limit; actual visible SM count is audited. GPU busy can still be high under50% MPS.':strict?'Batch is fixed between paced/unrestricted conditions. Full-wall busy retains ALL samples; for2 GPUs, show max pooled card busy. Both cards are in Window repeats.':'The predeclared near-saturation batch is fixed. Idle intervals occur only between actual inference groups. Actual busy is measured in the cost window.'],
 ['Energy',strict?'Integrate all used-GPU board power over TF + output-to-next-input intervals; add338.2W × charged seconds ONCE. Pacing waits included, not added twice. DS FP32 includes both cards.':'(338.2W retained host proxy + same-window sampled board power) × complete window duration, normalized by actual processed video evaluations.'],
 ['Accuracy','SRCC is a separate frozen full558 TEST development estimate: the original head was selected on TEST, with no VAL. No other precision is copied into FP32 cells.'],
 [strict?'Devices':'Capacity',data.capacity_missing],['Scope',complete?'Costs share the interval definition. Receiver versions and parameter-count sources are in Measurements. Historical QwenB1 is unchanged.':'80GB rental models prioritized. Missing original-platform measurements and unfinished precision cases remain blank. Historical QwenB1 not changed.']];
 for(const[i,[label,text]]of notes.entries()){const rr=22+i*2;sheet.getRange('A'+rr).values=[[label]];sheet.getRange(`B${rr}:M${rr}`).merge();sheet.getRange('B'+rr).values=[[text]];sheet.getRange(`A${rr}:M${rr}`).format={font:{name:'Arial',size:10,color:'#53687A'},wrapText:true,rowHeight:34};}
}
book.recalculate();
function verify(b){const s=b.worksheets.getItem('Measurements');for(const e of entries)if(e.point){for(const[c,k]of [['AC','time16_ms'],['AD','energy16_j'],['AB','busy_percent'],['H','srcc_reference']])if(e.point[k]!==undefined&&Math.abs(s.getRange(c+e.row).values[0][0]-e.point[k])>1e-8)throw Error('Raw arithmetic mismatch '+c+e.row);}}
verify(book);
const first=entries.find(e=>e.point);if(first){const cell=windows.getRange('I'+windowRows.get(key(first.point))[0]),v=cell.values[0][0];cell.values=[[v*1.01]];book.recalculate();if(book.worksheets.getItem('Measurements').getRange('AC'+first.row).values[0][0]===first.point.time16_ms)throw Error('Formula did not update');cell.values=[[v]];book.recalculate();verify(book);}
const errors=(await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:2000})).ndjson;
if(!errors.includes('Cell search matched 0 entries.'))throw Error(errors);
for(const {sheet}of views){const pic=await book.render({sheetName:sheet.name,range:'A1:M31',scale:1,format:'png'});await fs.writeFile(path.join(repo,'.codex_tmp','busy_resource_'+sheet.name+'.png'),new Uint8Array(await pic.arrayBuffer()));}
const rawPic=await book.render({sheetName:'Measurements',range:'A6:N12',scale:1,format:'png'});await fs.writeFile(path.join(repo,'.codex_tmp/busy_resource_raw.png'),new Uint8Array(await rawPic.arrayBuffer()));
const windowsPic=await book.render({sheetName:'Window repeats',range:'A1:M10',scale:1,format:'png'});await fs.writeFile(path.join(repo,'.codex_tmp/busy_resource_windows.png'),new Uint8Array(await windowsPic.arrayBuffer()));
if(strict){for(const [sheetName,range,file] of [['Measurements','AF6:AO12','busy_resource_metadata.png'],['Measurements','AK14:AO19','busy_resource_calibration_metadata.png'],['Window repeats','Y1:AK7','busy_resource_interval_audit.png'],['Window repeats','AL1:AO10','busy_resource_memory.png'],['Measurements','AP6:AR12','busy_resource_receiver.png']]){const pic=await book.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(repo,'.codex_tmp',file),new Uint8Array(await pic.arrayBuffer()));}}
await(await SpreadsheetFile.exportXlsx(book)).save(out);verify(await SpreadsheetFile.importXlsx(await FileBlob.load(out)));
await fs.writeFile(path.join(root,receiptName),JSON.stringify({file:out,sha256:crypto.createHash('sha256').update(await fs.readFile(out)).digest('hex'),source_sha256:crypto.createHash('sha256').update(await fs.readFile(path.join(root,'eight_tables.json'))).digest('hex'),views:8,raw_sheets:1,calculation_sheets:1,audited_points:data.values.length,recalculation:true,reimport:true,formula_errors:0},null,2));
try{await fs.rename(out+'.inspect.ndjson',path.join(root,'workbook_inspect_'+Date.now()+'.ndjson'));}catch(e){if(e.code!=='ENOENT')throw e;}
console.log(JSON.stringify({file:out,views:8,audited_points:data.values.length}));

async function buildMpsDeployed(){
 const repo=process.cwd();
 const root=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_strict_tf_rental_dual_20261020/presentation');
 const source=path.join(root,'mps_deployed.json'),data=JSON.parse(await fs.readFile(source,'utf8'));
 const folder=path.join(repo,'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013');
 const output=path.join(folder,'LGVQ_MPS_deployed_comparison.xlsx');
 if(data.values.length!==12||data.values.some(p=>p.batch!==1||p.precision!=='native'||p.method!=='mps'))throw Error('Only12 native MPS B1 conditions');
 try{
  const previous=await fs.readFile(output),digest=crypto.createHash('sha256').update(previous).digest('hex');
  const receipt=JSON.parse(await fs.readFile(path.join(root,'mps_deployed_workbook_receipt.json'),'utf8'));
  if(digest!==receipt.sha256)throw Error('Existing selected workbook has external edits');
  try{await fs.copyFile(output,path.join(root,`prior_mps_deployed_${digest}.xlsx`),1);}catch(e){if(e.code!=='EEXIST')throw e;}
 }catch(e){if(e.code!=='ENOENT')throw e;}
 const book=Workbook.create(),summary=book.worksheets.add('Plot data'),trials=book.worksheets.add('Trial data');
 for(const s of [summary,trials]){s.showGridLines=false;s.getRange('A1:U45').format.font={name:'Arial',size:10,color:'#243447'};}
 const colors={qwen:'#0072B2',glm:'#9B567C',clip:'#008B70',yolo:'#B17D00',deepseek:'#79502C',kimi:'#D14F64'};
 const head={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:42,horizontalAlignment:'center',verticalAlignment:'center'};
 summary.tabColor='#214D68';summary.getRange('A2').values=[['LGVQ temporal quality: MPS resource comparison']];
 summary.getRange('A2:K2').format={font:{name:'Arial',size:14,bold:true,color:'#214D68'},rowHeight:29};
 summary.getRange('A3').values=[['Original deployment precision; both MPS allocations use one video per inference.']];
 summary.getRange('A4').values=[['Time and energy are 16-video equivalents, not a simultaneous batch of 16.']];
 summary.getRange('A3:K4').format={font:{name:'Arial',size:10,color:'#53687A'},rowHeight:21};
 summary.getRange('A6:K6').values=[['Model','MPS allocation','SRCC','Equivalent time (ms)','Estimated energy (J)','Time / optical','Energy / optical','Inference precision','Parameters (B)','Video batch','Platform']];
 summary.getRange('A6:K6').format=head;
 summary.getRange('A7:K7').values=[['LightGen-II',null,data.optical.srcc,data.optical.time_ms,data.optical.energy_j,1,1,'Retained reference',null,null,'Optical system']];
 summary.getRange('A7:K7').format={fill:'#F4EDF8',font:{name:'Arial',size:10,color:'#7A3E9D'},rowHeight:30,verticalAlignment:'center'};
 trials.getRange('A1:U1').values=[['Model','MPS allocation','Video batch','Repeat','Charged seconds','First-block compute s','Output to next input s','Explicit pacing wait s','Video evaluations','Used-GPU energy J','Host proxy W','Estimated energy J','Actual GPU busy','Inference precision','Readout precision','Parameter dtypes','Report source','Report SHA256','Accuracy source','Accuracy SHA256','Identity SHA256']];
 trials.getRange('A1:U1').format=head;trials.freezePanes.freezeRows(1);trials.freezePanes.freezeColumns(2);
 const bound=[];let tr=2,sr=8;
 for(const p of data.values){
  const indices=[];
  for(const r of p.repeats){
   if(r.internal_wait_s!==0)throw Error('No pacing in MPS');
   const values=[p.model_name,p.resource_cap/100,p.batch,r.repeat,r.elapsed_s,r.compute_s,r.output_to_next_model_input_s,r.internal_wait_s,r.video_evaluations,r.gpu_energy_j,338.2,null,r.measurement.mean_gpu_utilization_percent/100,p.precision_detail,p.quality_readout_precision,p.backbone_parameter_dtypes.join(', '),p.source,p.source_sha256,p.accuracy_source,p.accuracy_source_sha256,p.identity_sha256];
   trials.getRange(`A${tr}:U${tr}`).values=[values];trials.getRange(`L${tr}`).formulas=[[`=E${tr}*K${tr}+J${tr}`]];
   trials.getRange(`A${tr}:U${tr}`).format.rowHeight=27;
   trials.getRange(`A${tr}`).format.font={name:'Arial',size:10,bold:true,color:colors[p.model]};
   indices.push(tr++);
  }
  const platform=['deepseek','kimi'].includes(p.model)?'A100 SXM4 80GB / 400W':'A100 PCIe 40GB / 250W';
  summary.getRange(`A${sr}:K${sr}`).values=[[p.model_name,p.resource_cap/100,p.srcc_reference,null,null,null,null,p.deployment_precision,p.model_parameters/1e9,p.batch,platform]];
  const sum=c=>indices.map(i=>`'Trial data'!${c}${i}`).join(',');
  summary.getRange(`D${sr}:G${sr}`).formulas=[[
   `=SUM(${sum('E')})*16000/SUM(${sum('I')})`,
   `=SUM(${sum('L')})*16/SUM(${sum('I')})`,
   `=D${sr}/$D$7`,`=E${sr}/$E$7`]];
  summary.getRange(`A${sr}:K${sr}`).format={fill:Math.floor((sr-8)/2)%2===0?'#F2F5F7':'#FFFFFF',rowHeight:30,verticalAlignment:'center'};
  summary.getRange(`A${sr}`).format.font={name:'Arial',size:10,bold:true,color:colors[p.model]};
  bound.push({point:p,row:sr,trials:indices});sr++;
 }
 summary.getRange('A:A').format.columnWidth=25;summary.getRange('B:G').format.columnWidth=17;
 summary.getRange('H:H').format.columnWidth=22;summary.getRange('I:J').format.columnWidth=17;summary.getRange('K:K').format.columnWidth=32;
 summary.getRange('B7:B19').setNumberFormat('0%');summary.getRange('C7:C19').setNumberFormat('0.000000');
 summary.getRange('D7:E19').setNumberFormat('#,##0.000');summary.getRange('F7:G19').setNumberFormat('0.00"×"');
 summary.getRange('I7:I19').setNumberFormat('0.000');summary.getRange('J7:J19').setNumberFormat('0');
 summary.getRange('B7:K19').format.horizontalAlignment='right';summary.getRange('K7:K19').format.horizontalAlignment='left';
 const notes=[
  'MPS percentages are execution-resource limits, not measured GPU utilization.',
  'First block to GPU MOS plus output to next model-call entry; every preblock frontend is excluded.',
  'Energy integrates the same intervals. The retained 338.2 W host proxy is added once.',
  'BF16 and FP16 refer to compute profiles, not integer quantization. Quality readout remains FP32.',
  'SRCC is a separate matching-batch full558 TEST development reference, reused for both MPS caps.',
  'The optical point retains the earlier component-cost reference, not a newly matched wall-clock run.',
  'Parameter counts retain their recorded basis. Qwen/GLM are nominal; CLIP includes its unused text branch.',
  'Same batch isolates resource-allocation changes; it does not remove numerical precision differences.'
 ];
 notes.forEach((v,i)=>{summary.getRange(`A${22+i}`).values=[[v]];summary.getRange(`A${22+i}:K${22+i}`).format={font:{name:'Arial',size:10,color:'#53687A'},rowHeight:22};});
 trials.getRange('A:D').format.columnWidth=22;trials.getRange('E:M').format.columnWidth=21;
 trials.getRange('N:P').format.columnWidth=52;trials.getRange('Q:U').format.columnWidth=78;
 trials.getRange('B2:B37').setNumberFormat('0%');trials.getRange('M2:M37').setNumberFormat('0.00%');
 trials.getRange('E2:L37').setNumberFormat('0.000000');trials.getRange('I2:I37').setNumberFormat('#,##0');
 trials.getRange('C2:D37').setNumberFormat('0');trials.getRange('Q2:U37').format.wrapText=true;trials.getRange('Q2:U37').format.rowHeight=58;
 function verify(w){const s=w.worksheets.getItem('Plot data');for(const {point:p,row:r}of bound){
  for(const [c,k]of [['D','time16_ms'],['E','energy16_j'],['C','srcc_reference']])if(Math.abs(s.getRange(c+r).values[0][0]-p[k])>1e-8)throw Error('Selected value mismatch '+c+r);
 }}
 book.recalculate();verify(book);
 const first=bound[0],cell=trials.getRange('E'+first.trials[0]),before=cell.values[0][0];cell.values=[[before*1.01]];book.recalculate();
 if(summary.getRange('D'+first.row).values[0][0]===first.point.time16_ms)throw Error('Summary failed to recalculate');
 cell.values=[[before]];book.recalculate();verify(book);
 const errors=(await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1500})).ndjson;
 if(!errors.includes('Cell search matched 0 entries.'))throw Error(errors);
 console.log((await book.inspect({kind:'table',range:'Plot data!A6:K10',include:'values,formulas',tableMaxRows:5,tableMaxCols:11,maxChars:2200})).ndjson);
 for(const [sheetName,range,name]of [['Plot data','A1:K30','mps_deployed_summary'],['Trial data','A1:M10','mps_deployed_trials'],['Trial data','N1:U4','mps_deployed_sources']]){
  const pic=await book.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(repo,'.codex_tmp',name+'.png'),new Uint8Array(await pic.arrayBuffer()));
 }
 await(await SpreadsheetFile.exportXlsx(book)).save(output);verify(await SpreadsheetFile.importXlsx(await FileBlob.load(output)));
 await fs.writeFile(path.join(root,'mps_deployed_workbook_receipt.json'),JSON.stringify({file:output,sha256:crypto.createHash('sha256').update(await fs.readFile(output)).digest('hex'),source_sha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex'),conditions:12,trials:36,batch:1,recalculation:true,reimport:true,formula_errors:0},null,2)+'\n');
 try{await fs.rename(output+'.inspect.ndjson',path.join(root,'mps_deployed_inspect.ndjson'));}catch(e){if(e.code!=='ENOENT')throw e;}
 console.log(JSON.stringify({file:output,conditions:12,trials:36}));
}

async function buildTemporalFp32Reference(){
 const repo=process.cwd(),root=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_strict_tf_rental_dual_20261020/presentation');
 const source=path.join(root,'temporal_fp32_reference.json'),data=JSON.parse(await fs.readFile(source,'utf8'));
 const digest=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
 const proof=JSON.parse(await fs.readFile(path.join(root,'temporal_fp32_reference_audit.json'),'utf8'));
 if(proof.status!=='passed'||proof.reference_sha256!==await digest(source)||data.values.length!==12)throw Error('Verified FP32 selection required');
 const output=path.join(repo,'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013/LGVQ_FP32_temporal_control_reference.xlsx');
 const receiptPath=path.join(root,'temporal_fp32_reference_workbook_receipt.json');
 try{const old=await fs.readFile(output),sha=crypto.createHash('sha256').update(old).digest('hex');
  if(JSON.parse(await fs.readFile(receiptPath,'utf8')).sha256!==sha)throw Error('Workbook edited externally');
  try{await fs.copyFile(output,path.join(root,'prior_fp32_temporal_'+sha+'.xlsx'),1);}catch(e){if(e.code!=='EEXIST')throw e;}
 }catch(e){if(e.code!=='ENOENT')throw e;try{await fs.access(output);throw Error('Unidentified existing workbook');}catch(x){if(x.code!=='ENOENT')throw x;}}
 const book=Workbook.create(),s=book.worksheets.add('Plot data'),t=book.worksheets.add('Trial data');
 const colors={qwen:'#0072B2',glm:'#92527F',clip:'#007F63',yolo:'#9B7000',deepseek:'#79502C',kimi:'#D14F64'};
 const heading={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:52,wrapText:true,horizontalAlignment:'center',verticalAlignment:'center'};
 for(const sh of [s,t]){sh.showGridLines=false;sh.getRange(sh===s?'A1:M34':'A1:Z41').format={font:{name:'Arial',size:10,color:'#263747'},verticalAlignment:'center',rowHeight:25};}
 s.tabColor='#214D68';s.getRange('A2').values=[['LGVQ temporal quality: FP32 time-control reference']];s.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#214D68'};
 s.getRange('A4').values=[['Prepared inputs reused after warmup/calibration. Separate reference, not cold distinct-video MPS measurement.']];
 s.getRange('A5').values=[['Each model uses its matching-batch unrestricted FP32 full558 quality reference for both cost settings.']];
 s.getRange('A7:M7').values=[['Model','Execution','FP32 reference SRCC','Time /16 (ms)','Energy /16 (J)','Time ratio vs optical','Energy ratio vs optical','Actual hot-GPU busy','Video batch','Parameters (B)','Used GPUs','Platform','Qualification']];s.getRange('A7:M7').format=heading;
 const op=data.optical;s.getRange('A8:M8').values=[['LightGen-II','Prior reference',op.srcc,op.time_ms,op.energy_j,1,1,null,null,null,null,'Optical component-cost reference','Not remeasured in this FP32 control study']];s.getRange('A8:M8').format.fill='#EEE6F4';s.getRange('A8').format.font={name:'Arial',size:10,bold:true,color:'#7828A8'};
 t.getRange('A1:Z1').values=[['Model','Execution','Video batch','Repeat','Compute (s)','Intercall gap (s)','Pacing wait inside gap (s)','Integrated used-GPU energy (J)','Host proxy (W)','Video evaluations','Time /16 (ms)','Energy /16 (J)','Hot-GPU busy','Busy sample count','GPU0 busy','GPU1 busy','FP32 full558 reference SRCC','Cost source','Cost SHA256','Accuracy source','Accuracy SHA256','Checkpoint SHA256','Precision','Used GPUs','Interval board power (W)','Platform']];t.getRange('A1:Z1').format=heading;
 let tr=2;const bindings=[];
 for(const [i,p]of data.values.entries()){
  const indices=[],label=p.condition==='unrestricted'?'Unrestricted':'Time-paced';
  for(const r of p.repeats){
   const info=r.gpu_infos[p.hot_gpu_index];
   t.getRange(`A${tr}:Z${tr}`).values=[[p.model_name,label,p.batch,r.repeat,r.compute_s,r.output_to_next_model_input_s,r.internal_wait_s,r.gpu_energy_j,338.2,r.video_evaluations,null,null,info.mean_gpu_utilization_percent/100,info.samples,r.gpu_infos[0].mean_gpu_utilization_percent/100,r.gpu_infos[1]?r.gpu_infos[1].mean_gpu_utilization_percent/100:null,p.srcc_reference,p.source,p.source_sha256,p.accuracy_source,p.accuracy_source_sha256,p.checkpoint_sha256,'FP32',p.gpu_count,null,p.platform]];
   t.getRange(`K${tr}:L${tr}`).formulas=[[`=(E${tr}+F${tr})*16000/J${tr}`,`=(I${tr}*(E${tr}+F${tr})+H${tr})*16/J${tr}`]];t.getRange(`Y${tr}`).formulas=[[`=H${tr}/(E${tr}+F${tr})`]];
   t.getRange(`A${tr}`).format.font={name:'Arial',size:10,bold:true,color:colors[p.model]};indices.push(tr++);
  }
  const row=9+i,sum=c=>indices.map(r=>`'Trial data'!${c}${r}`).join(','),a=indices[0],b=indices[2];
  s.getRange(`A${row}:M${row}`).values=[[p.model_name,label,null,null,null,null,null,null,p.batch,p.model_parameters/1e9,p.gpu_count,p.platform,p.condition==='unrestricted'&&p.busy_percent<98?'Unrestricted; measured below98% busy':p.condition==='busy50'?'Within50±5 percentage points':'Measured near98% or above']];
  s.getRange(`C${row}:H${row}`).formulas=[[`='Trial data'!Q${a}`,`=(SUM(${sum('E')})+SUM(${sum('F')}))*16000/SUM(${sum('J')})`,`=SUMPRODUCT('Trial data'!L${a}:L${b},'Trial data'!J${a}:J${b})/SUM(${sum('J')})`,`=D${row}/$D$8`,`=E${row}/$E$8`,`=SUMPRODUCT('Trial data'!M${a}:M${b},'Trial data'!N${a}:N${b})/SUM(${sum('N')})`]];
  s.getRange(`A${row}:M${row}`).format={fill:Math.floor(i/2)%2===0?'#F1F5F8':'#FFFFFF',rowHeight:36,verticalAlignment:'center'};s.getRange(`A${row}`).format.font={name:'Arial',size:10,bold:true,color:colors[p.model]};bindings.push({row,p});
 }
 s.getRange('A:A').format.columnWidth=26;s.getRange('B:B').format.columnWidth=22;s.getRange('C:H').format.columnWidth=19;s.getRange('I:K').format.columnWidth=15;s.getRange('L:L').format.columnWidth=48;s.getRange('M:M').format.columnWidth=45;
 s.getRange('C8:C20').setNumberFormat('0.000000');s.getRange('D8:E20').setNumberFormat('#,##0.000');s.getRange('F8:G20').setNumberFormat('0.00"×"');s.getRange('H9:H20').setNumberFormat('0.00%');s.getRange('J9:J20').setNumberFormat('0.000');s.getRange('I9:I20').setNumberFormat('0');s.getRange('K9:K20').setNumberFormat('0');
 const notes=['Historical resident-input reference only. Do not merge its cost points with the cold distinct-video MPS version.','Unrestricted means no imposed pacing, not guaranteed100% utilization. DS FP32 reaches57.95% on its hotter card.','Time includes each first-block-to-GPU-output interval and output-to-next-call gaps. Every model-call frontend is excluded.','Explicit pacing waits are already inside intercall gaps. Do not add the pacing-wait column again.','Energy integrates every used GPU over the same charged intervals plus the338.2W host proxy once.','Full558 FP32 quality is a separate unrestricted reference, shared with the time-paced point; not fresh paced accuracy.','DeepSeek FP32 uses two80GB GPUs; Kimi uses one80GB. Qwen/GLM/CLIP/YOLO use one40GB GPU.','All model floating parameters are FP32. Existing deployed weights were cast; lost BF16 information is not recovered.','Video counts denote completed evaluations of resident batches, not that many distinct videos. Three trials are all retained.','Historical BF16 Qwen B1 cost is not substituted into these FP32 measurements. Current MPS figures and workbook are unchanged.'];
 notes.forEach((v,i)=>s.getRange('A'+(24+i)).values=[[v]]);
 t.getRange('A:A').format.columnWidth=26;t.getRange('B:B').format.columnWidth=22;t.getRange('C:D').format.columnWidth=14;t.getRange('E:Q').format.columnWidth=22;t.getRange('R:V').format.columnWidth=78;t.getRange('W:X').format.columnWidth=15;t.getRange('Y:Y').format.columnWidth=24;t.getRange('Z:Z').format.columnWidth=48;
 t.getRange('A2:Z37').format.rowHeight=54;t.getRange('R2:Z37').format.wrapText=true;t.getRange('E2:L37').setNumberFormat('0.000000');for(const col of ['C','D','J','N','X'])t.getRange(`${col}2:${col}37`).setNumberFormat('0');for(const col of ['M','O','P'])t.getRange(`${col}2:${col}37`).setNumberFormat('0.00%');t.getRange('Q2:Q37').setNumberFormat('0.000000');t.getRange('Y2:Y37').setNumberFormat('0.000');t.freezePanes.freezeRows(1);t.freezePanes.freezeColumns(2);
 t.getRange('R40:S41').values=[['Audited source selection','SHA256'],[path.relative(repo,source).replaceAll('\\','/'),await digest(source)]];t.getRange('R40:S40').format=heading;t.getRange('R41:S41').format={wrapText:true,rowHeight:54};
 function verify(w){const sh=w.worksheets.getItem('Plot data');for(const {row,p}of bindings)for(const [col,key]of [['C','srcc_reference'],['D','time16_ms'],['E','energy16_j']])if(Math.abs(sh.getRange(col+row).values[0][0]-p[key])>1e-7)throw Error('FP32 formula mismatch '+col+row);}
 book.recalculate();verify(book);const cell=t.getRange('E2'),before=cell.values[0][0],oldTime=s.getRange('D9').values[0][0];cell.values=[[before*1.01]];book.recalculate();if(s.getRange('D9').values[0][0]===oldTime)throw Error('Recalculation did not propagate');cell.values=[[before]];book.recalculate();verify(book);
 const errors=(await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1200})).ndjson;if(!errors.includes('Cell search matched 0 entries.'))throw Error(errors);
 for(const [sheetName,range,name]of [['Plot data','A1:K33','fp32_temporal_summary'],['Plot data','L7:M20','fp32_temporal_hardware'],['Trial data','A1:Q7','fp32_temporal_trials'],['Trial data','R1:Z3','fp32_temporal_sources']]){const blob=await book.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(repo,'.codex_tmp',name+'.png'),new Uint8Array(await blob.arrayBuffer()));}
 await(await SpreadsheetFile.exportXlsx(book)).save(output);verify(await SpreadsheetFile.importXlsx(await FileBlob.load(output)));
 await fs.writeFile(receiptPath,JSON.stringify({file:output,sha256:await digest(output),source_sha256:await digest(source),conditions:12,trials:36,recalculation:true,reimport:true,formula_errors:0},null,2)+'\n');
 try{await fs.rename(output+'.inspect.ndjson',path.join(root,'fp32_temporal_reference_inspect.ndjson'));}catch(e){if(e.code!=='ENOENT')throw e;}
 console.log(JSON.stringify({file:output,conditions:12,trials:36}));
}

async function editTemporalFp32SharedAccuracy(){
 const repo=process.cwd(),root=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_strict_tf_rental_dual_20261020/presentation');
 const source=path.join(root,'temporal_fp32_paper.json'),data=JSON.parse(await fs.readFile(source,'utf8'));
 const digest=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
 if(data.fp32_selection_sha256!==await digest(path.join(repo,data.fp32_selection_source))||data.display_accuracy_source_sha256!==await digest(path.join(repo,data.display_accuracy_source)))throw Error('Presentation source changed');
 const output=path.join(repo,'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013/LGVQ_FP32_temporal_control_reference.xlsx');
 const receiptPath=path.join(root,'temporal_fp32_reference_workbook_receipt.json');
 const beforeSha=await digest(output),prior=JSON.parse(await fs.readFile(receiptPath,'utf8'));
 if(prior.sha256!==beforeSha)throw Error('Preserve externally edited workbook');
 const archive=path.join(root,'prior_fp32_temporal_'+beforeSha+'.xlsx');
 try{await fs.copyFile(output,archive,1);}catch(e){if(e.code!=='EEXIST')throw e;}
 const book=await SpreadsheetFile.importXlsx(await FileBlob.load(output));book.recalculate();
 const s=book.worksheets.getItem('Plot data'),t=book.worksheets.getItem('Trial data');
 console.log((await book.inspect({kind:'table',range:'Plot data!A7:E10',include:'values,formulas',tableMaxRows:4,tableMaxCols:5,maxChars:1800})).ndjson);
 const snapshot=()=>JSON.stringify({raw:t.getRange('A1:Z41').values,rawFormulas:t.getRange('A1:Z41').formulas,cost:s.getRange('D8:M20').values,costFormulas:s.getRange('D8:M20').formulas});
 const unchanged=snapshot(),models=['qwen','glm','clip','yolo','deepseek','kimi'];
 s.getRange('C7').values=[['Reference SRCC (MPS100)']];
 s.getRange('A5').values=[['Displayed SRCC uses native MPS100. FP32 costs are measured; actual FP32 quality remains in Trial data.']];
 s.getRange('A29').values=[['Shared native MPS100 quality is a cross-precision presentation reference, not measured FP32 performance.']];
 const heading={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:52,wrapText:true,horizontalAlignment:'center',verticalAlignment:'center'};
 t.getRange('A44:H44').values=[['Model','Shared SRCC (native MPS100)','Reference video batch','Actual FP32 full558 SRCC','Checkpoint SHA256','Native MPS100 reports','Native MPS100 report SHA256','Reference policy']];t.getRange('A44:H44').format=heading;
 for(const [i,model]of models.entries()){
  const p=data.values.find(p=>p.model===model),row=45+i;
  t.getRange(`A${row}:H${row}`).values=[[p.model_name,p.display_reference_srcc,p.display_reference_batch,p.srcc_reference,p.checkpoint_sha256,p.display_reference_reports.map(r=>r.source).join('\n'),p.display_reference_reports.map(r=>r.sha256).join('\n'),'Native-precision MPS100 display reference; not FP32 measured SRCC']];
  for(const [j,v]of data.values.entries())if(v.model===model)s.getRange('C'+(9+j)).formulas=[[`='Trial data'!B${row}`]];
 }
 t.getRange('A45:H50').format={font:{name:'Arial',size:10,color:'#263747'},rowHeight:72,wrapText:true,verticalAlignment:'center'};
 t.getRange('B45:B50').setNumberFormat('0.000000');t.getRange('D45:D50').setNumberFormat('0.000000');t.getRange('C45:C50').setNumberFormat('0');
 t.getRange('R44:S46').values=[['Shared display reference source','SHA256'],[data.display_accuracy_source,data.display_accuracy_source_sha256],[path.relative(repo,source).replaceAll('\\','/'),await digest(source)]];t.getRange('R44:S44').format=heading;t.getRange('R45:S46').format={wrapText:true,rowHeight:58};
 function verify(w){const sh=w.worksheets.getItem('Plot data');for(const [i,p]of data.values.entries())for(const [col,key]of [['C','display_reference_srcc'],['D','time16_ms'],['E','energy16_j']])if(Math.abs(sh.getRange(col+(9+i)).values[0][0]-p[key])>1e-7)throw Error('Shared reference mismatch '+p.model+col);}
 book.recalculate();verify(book);
 const cell=t.getRange('B45'),value=cell.values[0][0];cell.values=[[value-.01]];book.recalculate();
 if(Math.abs(s.getRange('C9').values[0][0]-(value-.01))>1e-12)throw Error('Display reference did not recalculate');
 cell.values=[[value]];book.recalculate();verify(book);
 if(snapshot()!==unchanged)throw Error('Original FP32 trial observations or costs changed');
 const errors=(await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1200})).ndjson;if(!errors.includes('Cell search matched 0 entries.'))throw Error(errors);
 for(const [sheetName,range,name]of [['Plot data','A1:K33','fp32_temporal_summary'],['Trial data','A44:D50','fp32_temporal_accuracy_references']]){const blob=await book.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(repo,'.codex_tmp',name+'.png'),new Uint8Array(await blob.arrayBuffer()));}
 await(await SpreadsheetFile.exportXlsx(book)).save(output);const reopened=await SpreadsheetFile.importXlsx(await FileBlob.load(output));verify(reopened);
 if(JSON.stringify(reopened.worksheets.getItem('Trial data').getRange('A1:Z41').values)!==JSON.stringify(t.getRange('A1:Z41').values))throw Error('Raw inputs changed after export');
 await fs.writeFile(receiptPath,JSON.stringify({file:output,sha256:await digest(output),source_sha256:await digest(source),fp32_selection_sha256:data.fp32_selection_sha256,display_accuracy_source_sha256:data.display_accuracy_source_sha256,display_accuracy_policy:data.display_accuracy_policy,conditions:12,trials:36,recalculation:true,reimport:true,formula_errors:0,raw_fp32_and_costs_unchanged:true,prior_workbook_sha256:beforeSha},null,2)+'\n');
 try{await fs.rename(output+'.inspect.ndjson',path.join(root,'fp32_temporal_reference_inspect.ndjson'));}catch(e){if(e.code!=='EEXIST'&&e.code!=='ENOENT')throw e;}
 console.log(JSON.stringify({file:output,conditions:12,raw_fp32_and_costs_unchanged:true}));
}
