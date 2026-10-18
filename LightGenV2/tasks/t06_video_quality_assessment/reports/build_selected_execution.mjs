import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const bundledRequire=createRequire(path.join(process.env.CODEX_ARTIFACT_RUNTIME_BASE??path.join(process.cwd(),'.codex_tmp'),'artifact-runtime.cjs'));
const {FileBlob,SpreadsheetFile,Workbook}=await import(pathToFileURL(bundledRequire.resolve('@oai/artifact-tool')).href);
const repo=process.cwd();
const out=path.join(repo,'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013');
const cleanKimi=process.argv.includes('--kimi-clean-extension');
const rental=process.argv.includes('--rental-extension')||cleanKimi;
const includeCrossDeviceAudit=process.argv.includes('--include-cross-device-audit');
const sequential=process.argv.includes('--yolo-clip-sequential')||rental;
const recheck=process.argv.includes('--yolo-clip-recheck')||sequential;
const baseRun=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_selected_execution_a100_20261016');
const recheckRun=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_yolo_clip_recheck_a100_20261017');
const sequentialRun=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_yolo_clip_sequential_a100_20261017');
const rentalRun=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_full_deepseek_kimi_rental_20261017');
const run=cleanKimi?path.join(rentalRun,'figure5_kimi_refresh_20261018'):rental?rentalRun:sequential?sequentialRun:recheck?recheckRun:baseRun;
const scratch=path.join(repo,'.codex_tmp');
if(process.argv.includes('--compact-groups')){
 const unified=process.argv.includes('--unified-inference');
 const source=path.join(repo,unified?'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_unified_inference_recheck_20261018/presentation/compact_groups_20261018.json':'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_resident_busy50_a100_20261017/attempt03_remaining_20261018/service_presentation/compact_groups_20261018.json');
 const data=JSON.parse(await fs.readFile(source,'utf8'));if(data.status!=='complete')throw Error('Unverified group input');
 const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
 for(const[file,sha]of Object.entries(data.source_sha256))if(digest(await fs.readFile(path.join(repo,file)))!==sha)throw Error('Bound evidence changed '+file);
 const results=[];
 for(const[group,rows]of Object.entries(data.groups)){
  const file=path.join(out,'LGVQ_'+group+'_plot_data.xlsx');
  const book=unified?await SpreadsheetFile.importXlsx(await FileBlob.load(file)):Workbook.create();
  const sheet=unified?book.worksheets.getItem('Plot data'):book.worksheets.add('Plot data'),inputs=unified?book.worksheets.getItem('Measurements'):book.worksheets.add('Measurements');
  if(unified){for(const[name,range]of [['Plot data','A1:G19'],['Measurements','A1:N11']]){const preview=await book.render({sheetName:name,range,scale:1,format:'png'});await fs.writeFile(path.join(scratch,group+'_before_'+name.replaceAll(' ','_')+'.png'),new Uint8Array(await preview.arrayBuffer()));}}
  sheet.showGridLines=false;inputs.showGridLines=false;sheet.tabColor='#214D68';
  const title=group.startsWith('batch1')?'Batch1 and near-saturation inference':unified?'Approximately50% and near-saturation inference':'Approximately50% and near-saturation service';
  sheet.getRange('A2').values=[[title]];sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#1D3345'};
  sheet.getRange('A4').values=[[unified?'Costs: first200 test videos. Pacing waits excluded. Utilization measured independently.':group.startsWith('batch1')?'Costs: first200 test videos normalized to16; utilization measured independently.':'Costs and utilization: the same complete resident windows; all pacing waits included.']];
  sheet.getRange('A4').format.font={name:'Arial',size:10,color:'#53687A'};
  const headers=['Model','Condition','Video B','Temporal SRCC','GPU busy','Time /16 (ms)','Estimated energy /16 (J)','Time / optical ref.','Energy / optical ref.','Parameters (B)','Platform','Data status'];
  sheet.getRange('A6:L6').values=[headers];
  inputs.getRange('A2').values=[['Measurement inputs and provenance']];
  inputs.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#1D3345'};
  const keys=['model','baseline','condition','batch','srcc','utilization_percent','time_ms','energy_j','duration_s','calls','board_w','host_proxy_w','parameters_billion','platform','gpu_uuid','cost_source','utilization_source','accuracy_source','checkpoint_sha256','protocol','notes','repeat1','repeat2','repeat3'];
  if(unified)keys.push('cost_wall_sum_ms','cost_video_count','active_samples','excluded_wait_total_s','first_call_ms','aggregation','excluded_cost_wait_s','excluded_wait_16_ms');
  const last=unified?'AF':'X';inputs.getRange(`A6:${last}6`).values=[keys];
  if(unified)inputs.getRange('V6:X6').values=[['Independent util repeat1 (%)','Independent util repeat2 (%)','Independent util repeat3 (%)']];
  if(unified){
   inputs.getRange('A23:AF23').values=[keys];
   inputs.getRange('V23:X23').values=[['Independent util repeat1 (%)','Independent util repeat2 (%)','Independent util repeat3 (%)']];
   for(const[i,x]of data.audit_points.entries()){
    const r=24+i,v={...x,model:x.baseline,condition:x.kind,protocol:'First200 dataset-once; no warmup; pacing excluded',accuracy_source:x.cost_source};
    inputs.getRange(`A${r}:AF${r}`).values=[keys.map(k=>v[k]??null)];
    inputs.getRange('G'+r).formulas=[[`=Y${r}*16/Z${r}`]];inputs.getRange('H'+r).formulas=[[`=(K${r}+L${r})*G${r}/1000`]];
   }
  }
  for(const[i,x]of rows.entries()){
   const r=i+7,values={...x,repeat1:x.repeats[0]??null,repeat2:x.repeats[1]??null,repeat3:x.repeats[2]??null};
   inputs.getRange(`A${r}:${last}${r}`).values=[keys.map(k=>values[k]??null)];
   if(unified&&x.repeat_inputs?.length&&x.repeat_inputs.every(v=>v.name)){
    const refs=x.repeat_inputs.map(v=>24+data.audit_points.findIndex(p=>p.name===v.name));
    for(const col of ['G','H','K'])inputs.getRange(col+r).formulas=[[`=AVERAGE(${refs.map(j=>col+j).join(',')})`]];
   }
   for(const[c,s]of [['A','A'],['B','C'],['C','D'],['D','E'],['J','M'],['K','N']]){
    if((c==='C'&&x.baseline==='ours')||(c==='J'&&x.parameters_billion==null))sheet.getRange(c+r).values=[['n.a.']];
    else if(c==='D'&&!x.plotted)sheet.getRange(c+r).values=[['Not measured']];
    else sheet.getRange(c+r).formulas=[[`=Measurements!${s}${r}`]];
   }
   if(!x.plotted){sheet.getRange(`E${r}:I${r}`).values=[['Not measured','Not measured','Not measured','n.a.','n.a.']];}
   else{
    if(x.baseline==='ours')sheet.getRange('E'+r).values=[['n.a. (optical)']];
    else sheet.getRange('E'+r).formulas=[[`=Measurements!F${r}/100`]];
    const resident=!unified&&x.duration_s!=null;
    sheet.getRange('F'+r).formulas=[[resident?`=Measurements!I${r}*16000/(Measurements!J${r}*C${r})`:`=Measurements!G${r}`]];
    sheet.getRange('G'+r).formulas=[[resident?`=(Measurements!K${r}+Measurements!L${r})*F${r}/1000`:`=Measurements!H${r}`]];
    sheet.getRange('H'+r).formulas=[[`=F${r}/$F$7`]];sheet.getRange('I'+r).formulas=[[`=G${r}/$G$7`]];
   }
   sheet.getRange('L'+r).values=[[!x.plotted?'Missing compliant50% cost':x.baseline==='ours'?'Retained optical reference':unified&&x.repeat_inputs?.length===3?'All3 full558 repeats retained':group.startsWith('busy50')?'Full558; independent utilization':'Full558; finalized B1 preserved']];
   const color=x.color;
   const pale={ours:'#F2EAF7',qwen:'#EAF3FA',glm:'#F5EBF3',clip:'#E8F5EF',yolo:'#FFF7DD',full:'#F4ECE3',kimi:'#FCECEF'}[x.baseline];
   sheet.getRange(`A${r}:L${r}`).format={fill:pale,font:{name:'Arial',size:10,color:'#243447'},rowHeight:29,verticalAlignment:'center'};
   sheet.getRange(`A${r}`).format.font={name:'Arial',size:10,bold:true,color};
   sheet.getRange(`D${r}:J${r}`).format.horizontalAlignment='right';
  }
  const end=rows.length+6;
  sheet.getRange(`A6:L${end}`).format.columnWidth=18;sheet.getRange(`A6:A${end}`).format.columnWidth=24;
  sheet.getRange(`B6:B${end}`).format.columnWidth=31;sheet.getRange(`C6:C${end}`).format.columnWidth=9;
  sheet.getRange(`J6:J${end}`).format.columnWidth=15;sheet.getRange(`K6:K${end}`).format.columnWidth=26;sheet.getRange(`L6:L${end}`).format.columnWidth=38;
  sheet.getRange(`D7:D${end}`).setNumberFormat('0.000000');sheet.getRange(`E7:E${end}`).setNumberFormat('0.0%');
  sheet.getRange(`F7:G${end}`).setNumberFormat('#,##0.000');sheet.getRange(`H7:I${end}`).setNumberFormat('0.00"×"');sheet.getRange(`J7:J${end}`).setNumberFormat('0.000');
  for(const s of [sheet,inputs])s.getRange(s===sheet?'A6:L6':`A6:${last}6`).format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:42,horizontalAlignment:'center',verticalAlignment:'center'};
  inputs.getRange(`A7:X${end}`).format={font:{name:'Arial',size:10,color:'#243447'},columnWidth:21,rowHeight:26,verticalAlignment:'center'};
  inputs.getRange(`P7:U${end}`).format.columnWidth=66;inputs.getRange(`O7:O${end}`).format.columnWidth=48;
  inputs.getRange(`E7:M${end}`).setNumberFormat('0.000000');inputs.getRange(`J7:J${end}`).setNumberFormat('#,##0');inputs.getRange(`V7:X${end}`).setNumberFormat('0.000000');
  if(unified){
   inputs.getRange('Y6:AF33').format.columnWidth=23;
   inputs.getRange('Y7:AF33').setNumberFormat('0.000000');
   inputs.getRange('A24:AF33').format={font:{name:'Arial',size:10,color:'#243447'},rowHeight:26,verticalAlignment:'center'};
   inputs.getRange('A23:AF23').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:42};
   inputs.getRange('Z7:AA33').setNumberFormat('#,##0');
  }
  const notes=[
   ['图示规则','同一模型保持同一颜色和符号；图中仅标独立GPU时间忙碌率，不显示batch或N。光学约50%为空间使用的作者估计，不是GPU利用率。'],
   ['常规成本',group.startsWith('batch1')?'558个TEST视频独立评估SRCC；前200个视频的同步推理边界时间归一至16视频。利用率来自独立驻留测量。':'驻留服务T16=完整测量秒数×16000/(batch×前向次数)；能量=(338.2W主机代理+同窗口均值板卡功率)×T16/1000。'],
   ['连线解释',group.startsWith('batch1')?'Qwen虚线：历史B1与等价审计优化B3，非纯batch因果对照；完整版DS虚线跨不同物理卡。':'同batch、同权重、同物理卡的50%和近满载服务窗口相连；测量日期、加载历史和控制器差异不消失。'],
   ['精度边界',group.startsWith('batch1')?'每个batch各自的full558 SRCC，不统一覆盖成同一精度。':'50%下仅核对重复输入输出一致性；SRCC引用同batch独立full558评估，不冒称在节流状态重评558条。'],
   ['缺失与排除',group.startsWith('batch1')?'仅展示B1与一个近满载点；不含DS Tiny/Small或租赁Qwen。':'CLIP使用B10约50%点，不使用B1的39.20%；完整版DS无50%数据，图中只显示近满载参考，缺项不填造。'],
   ['平台','Qwen/GLM/CLIP/YOLO：A100 PCIe40GB、250W；完整版DS/Kimi：SXM4 80GB、400W。非严格统一硬件排行。'],
   ['图幅','每张图78×88mm，600dpi PNG和矢量SVG/PDF；同组两图并排宽156mm。图轴为对数成本轴，保留全部实际值。']
  ];
  if(unified){
   notes[1][1]='每项558不同TEST评估；前200同步推理时间×16/200。能量=(338.2W+本轮active功率)×T16/1000。所有限流等待仅审计，不计正式成本。';
   notes[2][1]='两组近满载端点使用完全相同的dataset-once成本。50%为独立驻留实验的忙碌率，不声称成本运行整段恰好50%。';
   notes[3][1]='新补测逐项full558 SRCC；CLIP native保留三轮并展示均值，全部原值在Measurements第24行起。无训练或TEST选模。';
   notes[4][1]='完整版DeepSeek及Kimi不重测；无合格50% dataset-once成本则留缺，不以resident服务时间替代。CLIP50ms功率采样稀疏，保留采样数。';
  }
  for(const[i,[label,text]]of notes.entries()){const r=end+3+i*2;sheet.getRange(`A${r}`).values=[[label]];sheet.getRange(`B${r}:L${r}`).merge();sheet.getRange('B'+r).values=[[text]];sheet.getRange(`A${r}:L${r}`).format={font:{name:'Arial',size:10,color:'#53687A'},rowHeight:32,wrapText:true,verticalAlignment:'center'};}
  book.recalculate();
  const verify=wb=>{const s=wb.worksheets.getItem('Plot data');for(const[i,x]of rows.entries()){if(!x.plotted)continue;for(const[c,k]of [['D','srcc'],['F','time_ms'],['G','energy_j']])if(Math.abs(s.getRange(c+(i+7)).values[0][0]-x[k])>1e-8)throw Error('Group workbook value mismatch '+group+c+(i+7));}};
  verify(book);
  const before=sheet.getRange('F8').values[0][0];
  const driver=unified?'G8':group.startsWith('batch1')?'G8':'I8',driverOriginal=inputs.getRange(driver).values[0][0],driverFormula=inputs.getRange(driver).formulas[0][0];
  inputs.getRange(driver).values=[[driverOriginal*1.01]];book.recalculate();if(Math.abs(sheet.getRange('F8').values[0][0]-before*1.01)>1e-8)throw Error('Recalculation test');
  if(driverFormula)inputs.getRange(driver).formulas=[[driverFormula]];else inputs.getRange(driver).values=[[driverOriginal]];
  book.recalculate();verify(book);
  const errors=(await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson;
  if(!errors.includes('Cell search matched 0 entries.'))throw Error(errors);
  console.log((await book.inspect({kind:'table',range:"'Plot data'!A6:G9",tableMaxRows:4,tableMaxCols:7,maxChars:1600})).ndjson);
  for(const[name,range]of [['Plot data',`A1:G${end}`],['Measurements','A1:N11']]){
   const preview=await book.render({sheetName:name,range,scale:1.3,format:'png'});await fs.writeFile(path.join(scratch,group+'_'+name.replaceAll(' ','_')+'.png'),new Uint8Array(await preview.arrayBuffer()));
  }
  if(unified)for(const[label,range]of [['repeat_inputs','A23:L33'],['timing_wait_audit','Y23:AF33']]){
   const preview=await book.render({sheetName:'Measurements',range,scale:1,format:'png'});await fs.writeFile(path.join(scratch,group+'_'+label+'.png'),new Uint8Array(await preview.arrayBuffer()));
  }
  await(await SpreadsheetFile.exportXlsx(book)).save(file);verify(await SpreadsheetFile.importXlsx(await FileBlob.load(file)));
  results.push({file:path.relative(repo,file).replaceAll('\\','/'),sha256:digest(await fs.readFile(file)),rows:rows.length,plotted:rows.filter(x=>x.plotted).length,formula_errors:0,recalculation_verified:true,reimport_verified:true});
 }
 await fs.writeFile(path.join(path.dirname(source),'compact_workbooks_receipt_20261018.json'),JSON.stringify({status:'complete',source_sha256:digest(await fs.readFile(source)),workbooks:results},null,2)+'\n');
 console.log(JSON.stringify({created_workbooks:2,original_workbooks_unchanged:true}));process.exit(0);
}
function chooseDisplayPoints(points){
 const selected=new Set();
 for(const model of new Set(points.map(x=>x.baseline))){
  const values=points.filter(x=>x.baseline===model),higher=values.filter(x=>x.batch_size>1),near=higher.filter(x=>x.independent_resident_util_mean_percent>=95).slice(-2);
  for(const x of values.filter(x=>x.batch_size===1).concat(near.length?near:higher.slice(-1)))selected.add(model+':'+x.batch_size);
 }
 return selected;
}
if(process.argv.includes('--test-selection')){
 const cases=[['yolo',1,44],['yolo',2,73],['yolo',3,98],['clip',1,39],['clip',9,89],['clip',10,99],['qwen',3,97.1],['qwen',4,97.0],['fallback',1,50],['fallback',2,70],['fallback',3,80]];
 const actual=[...chooseDisplayPoints(cases.map(([baseline,batch_size,independent_resident_util_mean_percent])=>({baseline,batch_size,independent_resident_util_mean_percent})))];
 if(JSON.stringify(actual)!==JSON.stringify(['yolo:1','yolo:3','clip:1','clip:10','qwen:3','qwen:4','fallback:1','fallback:3']))throw Error('Display selection contract failed');
 console.log(JSON.stringify({selection_test_passed:true,midpoints_excluded:true,unmet_target_fallback:true}));process.exit(0);
}
const filenames=['plot_data.xlsx','LGVQ_temporal_multi_baseline_recalculated.xlsx','LGVQ_temporal_gpu_utilization.xlsx'];
const books=await Promise.all(filenames.map(async f=>SpreadsheetFile.importXlsx(await FileBlob.load(path.join(out,f)))));
const [plot,main,util]=books;
async function updateBusy50Service(){
 const base=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_resident_busy50_a100_20261017/attempt03_remaining_20261018');
 const source=path.join(base,'verified_busy50_service_extension.json');
 const evidence=JSON.parse(await fs.readFile(source,'utf8'));
 if(evidence.status!=='complete'||evidence.original40GB_points!==12||evidence.rows.length!==14)throw Error('All authorized service points must pass raw audit');
 const service=plot.worksheets.getItem('Paced service'),points=evidence.rows;
 const verify=()=>{for(const[i,x]of points.entries()){const r=i+7;for(const[c,key]of[['C','mean_busy_percent'],['G','service_T16_ms'],['H','service_E16_j']])if(Math.abs(service.getRange(c+r).values[0][0]-(c==='C'?x[key]/100:x[key]))>1e-8)throw Error('Paced service source mismatch '+c+r);}}
 if(process.argv.includes('--verify-busy50-service')){verify();console.log(JSON.stringify({service_rows_reimported_verified:14,conventional_costs_unchanged:true}));return;}
 if(process.argv.includes('--polish-busy50-service')){
  service.getRange('J7:K20').format.horizontalAlignment='center';service.getRange('M7:M20').format.horizontalAlignment='center';plot.recalculate();verify();
  const preview=await plot.render({sheetName:'Paced service',range:'A1:M26',scale:1,format:'png'});await fs.writeFile(path.join(scratch,'busy50_service_complete.png'),new Uint8Array(await preview.arrayBuffer()));
  const file=path.join(out,filenames[0]);await(await SpreadsheetFile.exportXlsx(plot)).save(file);
  const receiptFile=path.join(base,'service_presentation/receipt.json'),receipt=JSON.parse(await fs.readFile(receiptFile,'utf8'));receipt.output_workbook_sha256=crypto.createHash('sha256').update(await fs.readFile(file)).digest('hex');receipt.visual_QA_formatting=true;await fs.writeFile(receiptFile,JSON.stringify(receipt,null,2)+'\n');
  console.log(JSON.stringify({workbook_polished:1,service_values_unchanged:true}));return;
 }
 const checksum=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
 const archive=path.join(base,'service_presentation/prior_report_archive');await fs.mkdir(archive,{recursive:true});
 const documentation=await Promise.all(['README.md','UTILIZATION_METHOD_20261015.md'].map(async name=>({name,text:await fs.readFile(path.join(out,name),'utf8')})));
 const oldStatement='The original40GB paced experiment is incomplete after a telemetry-reader failure: four points passed independent raw audit, eight remain uncompleted. Neither failed telemetry nor inferred values enter these Kimi tables. Completion of this Kimi follow-up does not imply completion of that separate suite.';
 for(const doc of documentation){if(!doc.text.includes(oldStatement))throw Error('Unexpected methods edit; preserve other writer');try{await fs.copyFile(path.join(out,doc.name),path.join(archive,doc.name),fs.constants.COPYFILE_EXCL);}catch(error){if(error.code!=='EEXIST'||await fs.readFile(path.join(archive,doc.name),'utf8')!==doc.text)throw error;}}
 const filename=path.join(out,filenames[0]),prior=await fs.readFile(filename),priorPath=path.join(archive,filenames[0]);
 try{await fs.copyFile(filename,priorPath,fs.constants.COPYFILE_EXCL);}catch(error){if(error.code!=='EEXIST'||checksum(await fs.readFile(priorPath))!==checksum(prior))throw error;}
 const before={Comparison:{values:plot.worksheets.getItem('Comparison').getRange('A1:M32').values,formulas:plot.worksheets.getItem('Comparison').getRange('A1:M32').formulas},Measurements:{values:plot.worksheets.getItem('Measurements').getRange('A1:Y24').values,formulas:plot.worksheets.getItem('Measurements').getRange('A1:Y24').formulas}};
 service.getRange('A1:Q70').unmerge();service.getRange('A1:Q70').clear({applyTo:'all'});
 service.getRange('A1:M1').merge();service.getRange('A1').values=[['Paced resident inference at approximately 50% GPU busy time']];
 service.getRange('A2:M2').merge();service.getRange('A2').values=[['All imposed waits included. These service costs do not replace conventional dataset-once inference measurements.']];
 service.getRange('A3:M3').merge();service.getRange('A3').values=[['Original 40GB: 12 completed points. Kimi 80GB: 2 separate points. Three complete repeats per point; all samples retained.']];
 service.getRange('A4:M4').merge();service.getRange('A4').values=[['50% driver-window busy time is not equivalent to optical spatial occupancy. Existing full558 SRCC is a separate reference, not remeasured under pacing.']];
 service.getRange('A6:M6').values=[['Model','Video B','GPU busy mean','Repeat1','Repeat2','Repeat3','Service T16 (ms)','Estimated service E16 (J)','Board mean (W)','50±5pp target','Platform / limit','Existing full558 SRCC','Measurement profile']];
 service.getRange('A30:O30').values=[['Model','B','Full duration (s)','Forward calls','Same-window board (W)','Host proxy (W)','Busy mean (%)','Repeat1 (%)','Repeat2 (%)','Repeat3 (%)','Raw report source','GPU UUID','Full558 source','Telemetry profile','Calibration profile']];
 const colors={qwen:['#0072B2','#EAF3FA'],glm:['#92527F','#F5EBF3'],small:['#C25B18','#FFF0E5'],clip:['#007F63','#E8F5EF'],yolo:['#9B7000','#FFF7DD'],deepseek:['#5B6470','#EEF0F3'],kimi:['#B64058','#FCECEF']};
 const comparison=plot.worksheets.getItem('Comparison'),measurements=plot.worksheets.getItem('Measurements');
 for(const[i,x]of points.entries()){
  const r=i+7,s=i+31,color=colors[x.baseline];
  const candidates=measurements.getRange('A5:D23').values;const index=candidates.findIndex(v=>v[0]===x.model&&v[3]===x.batch);if(index<0)throw Error('Missing separately sourced full558 point');
  const ref=index+7;if(Math.abs(comparison.getRange('E'+ref).values[0][0]-x.separate_full558_srcc)>1e-10)throw Error('Accuracy reference drift');
  service.getRange(`A${s}:O${s}`).values=[[x.model,x.batch,x.duration_s,x.completed_forward_calls,x.board_mean_w,x.host_proxy_w,x.mean_busy_percent,...x.repeat_means_percent,x.source,x.gpu_uuid,x.full558_srcc_separate_source??'Conventional Comparison row'+ref,x.telemetry_profile,x.calibration]];
  service.getRange(`A${r}:M${r}`).formulas=[[`=A${s}`,`=B${s}`,`=G${s}/100`,`=H${s}/100`,`=I${s}/100`,`=J${s}/100`,`=C${s}*16000/(D${s}*B${s})`,`=(E${s}+F${s})*G${r}/1000`,`=E${s}`,`=IF(AND(C${r}>=0.45,C${r}<=0.55),"Within","Not reached")`,'=""',`=Comparison!E${ref}`,'=""']];
  service.getRange('K'+r).values=[[x.platform]];service.getRange('M'+r).values=[[x.baseline==='kimi'?'80GB, last-group feedback':i<4?'40GB, original logging':'40GB, raw-line logging']];
  service.getRange(`A${r}:M${r}`).format={fill:color[1],font:{name:'Arial',size:10,color:'#243447'},rowHeight:32,verticalAlignment:'center'};service.getRange(`A${r}`).format.font={name:'Arial',size:10,bold:true,color:color[0]};
 }
 service.getRange('A1:O45').format.columnWidth=17;service.getRange('A1:A45').format.columnWidth=25;service.getRange('B1:B45').format.columnWidth=9;service.getRange('C1:F45').format.columnWidth=13;service.getRange('G1:H45').format.columnWidth=22;service.getRange('K1:K45').format.columnWidth=27;service.getRange('M1:O45').format.columnWidth=28;
 service.getRange('A1:M1').format={fill:'#214D68',font:{name:'Arial',size:15,bold:true,color:'#FFFFFF'},rowHeight:35};service.getRange('A2:M4').format={font:{name:'Arial',size:10,color:'#53687A'},wrapText:true,rowHeight:30};
 for(const range of ['A6:M6','A30:O30'])service.getRange(range).format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:48,verticalAlignment:'center',horizontalAlignment:'center'};
 service.getRange('C7:F20').setNumberFormat('0.00%');service.getRange('G7:I20').setNumberFormat('#,##0.000');service.getRange('L7:L20').setNumberFormat('0.000000');service.getRange('B7:L20').format.horizontalAlignment='right';service.getRange('A31:O44').format={font:{name:'Arial',size:10,color:'#243447'},rowHeight:25};
 service.getRange('J7:K20').format.horizontalAlignment='center';service.getRange('M7:M20').format.horizontalAlignment='center';
 service.getRange('J7:J20').conditionalFormats.add('containsText',{text:'Not reached',format:{fill:'#FFF4D6',font:{bold:true,color:'#9C6500'}}});
 service.getRange('A22:M23').merge();service.getRange('A22').values=[['Service T16 = complete measured duration ×16000 / (forward calls ×B). Estimated E16 = (338.2 W + same-window mean board W) ×service T16/1000. Resident repeats are not558 distinct videos. Missing or below-target utilization is never rounded up.']];service.getRange('A22:M23').format={font:{name:'Arial',size:10,color:'#53687A'},wrapText:true,rowHeight:30};
 service.getRange('A25:M26').merge();service.getRange('A25').values=[['The reference SRCC is linked to the conventional full558 evaluation at the same model/batch. Pacing checks repeated first-B output parity only. The first four 40GB points retain their prior reader; later eight add raw stdout and fail-fast evidence preservation. Kimi uses its separately disclosed last-group controller and400W platform.']];service.getRange('A25:M26').format={font:{name:'Arial',size:10,color:'#53687A'},wrapText:true,rowHeight:30};
 plot.recalculate();const old=service.getRange('C31').values[0][0],answer=service.getRange('G7').values[0][0];service.getRange('C31').values=[[old*1.01]];plot.recalculate();if(Math.abs(service.getRange('G7').values[0][0]-answer*1.01)>1e-8)throw Error('Service recalculation failed');service.getRange('C31').values=[[old]];plot.recalculate();verify();
 for(const[name,state]of Object.entries(before)){const sheet=plot.worksheets.getItem(name),range=name==='Comparison'?'A1:M32':'A1:Y24';if(JSON.stringify({values:sheet.getRange(range).values,formulas:sheet.getRange(range).formulas})!==JSON.stringify(state))throw Error('Unrelated conventional sheet changed');}
 const errors=(await plot.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson;if(!errors.includes('Cell search matched 0 entries.'))throw Error(errors);
 const preview=await plot.render({sheetName:'Paced service',range:'A1:M26',scale:1,format:'png'});await fs.writeFile(path.join(scratch,'busy50_service_complete.png'),new Uint8Array(await preview.arrayBuffer()));
 await(await SpreadsheetFile.exportXlsx(plot)).save(filename);
 const quote=v=>/[",\r\n]/.test(String(v??''))?'"'+String(v??'').replaceAll('"','""')+'"':String(v??'');const keys=[...new Set(points.flatMap(x=>Object.keys(x)))];await fs.writeFile(path.join(out,'LGVQ_paced_service.csv'),[keys,...points.map(x=>keys.map(k=>Array.isArray(x[k])?x[k].join(';'):x[k]))].map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n');
 const f=(v,n=6)=>Number(v).toFixed(n);
 const table=['| Model | B | Pooled GPU busy (%) | Repeat1 /2 /3 (%) | Service T16 (ms) | Estimated service E16 (J) | Same-window board mean (W) | Platform |',
 '| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |',...points.map(x=>`| ${x.model} | ${x.batch} | ${f(x.mean_busy_percent)} | ${x.repeat_means_percent.map(v=>f(v)).join(' / ')} | ${f(x.service_T16_ms)} | ${f(x.service_E16_j)} | ${f(x.board_mean_w)} | ${x.platform} |`)].join('\n');
 const outside=points.filter(x=>!x.target50_reached).map(x=>`${x.model} B=${x.batch} (${f(x.mean_busy_percent,2)}%)`).join(', ');
 const text=`**Supplementary paced resident service: completed original 40GB suite.** We measured the six frozen original-platform models at their predeclared batch pairs on A100-PCIE-40GB at 250 W: Qwen B=1/3, GLM B=1/2, DeepSeek-VL2-Small B=1/3, CLIP B=1/10, YOLO B=1/3 and DeepSeek-VL2-Tiny B=1/8. Kimi B=1/6 retains the separate same-card A100-SXM4-80GB/400W measurements. On the original 40GB platform, a fresh process was used for each model–batch point, retaining one loaded model through calibration and all three repeats. At each point, only the first B TEST videos, each retaining the original four frames, were prepared and transferred once. Subsequent forwards reused these resident inputs and each processed B videos. Queue8 denotes eight sequential forwards before synchronization, not 8B simultaneously processed videos.

The original 40GB controller calibrated on complete compute-plus-imposed-idle groups for at least 30 s, then froze the duty fraction for all three repeats. Each repeat included at least 5 s of warmup and a complete measurement window of at least 30 s. All 50-ms driver samples in the three complete windows were pooled. The target was 50%, with a tolerance of ±5 percentage points of kernel-busy time, not SM occupancy or optical spatial occupancy. ${outside?'The target was not reached at '+outside+'. These observations are retained without rounding up or adding unrelated GPU work.':'All measured means fell within the declared target band.'} CLIP B=1 reached duty1 with zero imposed idle time, so its lower busy fraction is a recorded native-execution outcome, not a50% observation. Kimi used its separately disclosed last-completed-group feedback controller rather than the original 40GB aggregation across groups since the preceding update. No weights, numerical precision, frames, power limits or clocks were changed.

For a model batch B, normalized service T16 equals 16000×the sum of complete measured durations in seconds divided by B×the total completed forward calls. Mean board power is calculated within each complete window and duration-weighted across repeats. Estimated service E16=(338.2 W host-power proxy+same-window board mean)×service T16/1000, with T16 in milliseconds. Every imposed wait is included. These sampled-power service estimates are not calibrated whole-system energy-meter integrals. Repeating the first B inputs is not an evaluation of 558 distinct videos: the displayed full558 SRCC is linked to the separately measured conventional point, and paced runs check repeated-input MOS parity with an absolute MOS tolerance of 0.0002. Different input populations and warmup boundaries mean these resident service costs must not replace, or be interpreted as a causal slowdown ratio against, the first-200 dataset-once inference costs. The finalized Qwen B=1 conventional reference remains unchanged.

The first four 40GB points were preserved with 45 bound evidence hashes. After explicit authorization, only the remaining eight were measured in attempt03_remaining_20261018 using the published telemetry repair. The repair retains raw stdout, rejects unavailable numeric fields and saves available evidence before propagating failure; it does not impute or skip missing observations. All 12 original points passed independent raw-window, waiting-time and cost replay. The failed Small attempt remains excluded and archived. Full-precision service data and per-point source paths are retained in [LGVQ_paced_service.csv](LGVQ_paced_service.csv), alongside the updated Paced service worksheet.

${table}

`;
 for(const doc of documentation){const updated=doc.text.replace(/\*\*Separate paced service measurements\.\*\*[\s\S]*?(?=The 120 delivered raw files)/,text);if(updated===doc.text)throw Error('Missing paced Methods section');await fs.writeFile(path.join(out,doc.name),updated);}
 await fs.writeFile(path.join(base,'service_presentation/receipt.json'),JSON.stringify({status:'complete',rows:14,original40GB_points:12,source_sha256:checksum(await fs.readFile(source)),prior_workbook_sha256:checksum(prior),output_workbook_sha256:checksum(await fs.readFile(filename)),conventional_sheets_values_and_formulas_unchanged:true,formula_errors:0},null,2)+'\n');
 console.log(JSON.stringify({workbooks_updated:1,service_rows:14,conventional_sheets_unchanged:true}));
}
if(process.argv.includes('--busy50-service-inspect')){
 console.log((await plot.inspect({kind:'table',range:"'Paced service'!A1:K18",tableMaxRows:8,tableMaxCols:11,maxChars:2200})).ndjson);
 const preview=await plot.render({sheetName:'Paced service',range:'A1:F11',scale:1,format:'png'});await fs.writeFile(path.join(scratch,'busy50_service_before.png'),new Uint8Array(await preview.arrayBuffer()));process.exit(0);
}
if(process.argv.includes('--busy50-service-extension')||process.argv.includes('--verify-busy50-service')||process.argv.includes('--polish-busy50-service')){
 await updateBusy50Service();process.exit(0);
}
if(process.argv.includes('--inspect')){
 for(let i=0;i<books.length;i++)console.log(filenames[i],(await books[i].inspect({kind:'sheet',include:'id,name',maxChars:1500})).ndjson);
 for(const [book,sheetName,range,file] of [[plot,'Comparison','A1:L17','plot'],[main,'Summary','A1:L13','summary'],[util,'GPU utilization','A1:I13','util']]){
  console.log((await book.inspect({kind:'table',range:sheetName+'!'+range,tableMaxRows:8,tableMaxCols:12,maxChars:2400})).ndjson);
  const image=await book.render({sheetName,range,scale:1,format:'png'});
  await fs.writeFile(path.join(scratch,'selected_before_'+file+'.png'),new Uint8Array(await image.arrayBuffer()));
 }
 for(const [book,range]of [[plot,'Measurements!A1:T7'],[main,"'Batch results'!A1:AR2"],[main,'Sources!A1:B32'],[util,"'Repeat measurements'!A1:O6"]])console.log((await book.inspect({kind:'table',range,tableMaxRows:6,tableMaxCols:44,maxChars:4000})).ndjson);
 process.exit(0);
}
const json=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const data=await json(path.join(baseRun,'verified_selected_execution.json'));
if(data.status!=='complete'||data.paired_rows.length!==16||data.repeats.length!==48)throw Error('All selected evidence must be complete and independently verified');
let historicalRepeats=[],newCount=16;
if(recheck){
 const extra=await json(path.join(recheckRun,'verified_selected_execution.json'));
 if(extra.status!=='complete'||![2,3,4].includes(extra.paired_rows.length)||extra.repeats.length!==extra.paired_rows.length*3)throw Error('Incomplete prescribed recheck');
 const keys=new Set(extra.paired_rows.map(x=>x.baseline+':'+x.batch_size));
 if(extra.paired_rows.some(x=>!['clip','yolo'].includes(x.baseline)))throw Error('Recheck scope');
 historicalRepeats=data.repeats.filter(x=>keys.has(x.baseline+':'+x.batch)).map(x=>({...x,role:'Historical superseded round; never pooled with new repeats'}));
 data.paired_rows=[...data.paired_rows.filter(x=>!keys.has(x.baseline+':'+x.batch_size)),...extra.paired_rows];
 const order=['qwen','glm','small','clip','yolo','deepseek'];
 data.paired_rows.sort((a,b)=>order.indexOf(a.baseline)-order.indexOf(b.baseline)||a.batch_size-b.batch_size);
 data.repeats=[...data.repeats.filter(x=>!keys.has(x.baseline+':'+x.batch)),...extra.repeats];newCount=extra.paired_rows.length;
}
if(sequential){
 const seq=await json(path.join(sequentialRun,'verified_sequential.json'));
 if(seq.status!=='complete'||seq.points.length!==11||seq.repeats.length!==33||!seq.original_measurement_loop_AST_equal)throw Error('Incomplete sequential reproduction');
 for(const s of seq.points.filter(x=>x.target_point)){
  const raw=data.paired_rows.find(x=>x.baseline===s.baseline&&x.batch_size===s.batch);if(!raw)throw Error('No original frozen accuracy/cost for target');
  historicalRepeats.push(...data.repeats.filter(x=>x.baseline===s.baseline&&x.batch===s.batch).map(x=>({...x,role:'Historical fresh-process repeat; distinct from sequential reproduction'})));
  data.repeats=data.repeats.filter(x=>!(x.baseline===s.baseline&&x.batch===s.batch));
  data.repeats.push(...seq.repeats.filter(x=>x.baseline===s.baseline&&x.batch===s.batch));
  Object.assign(raw,{independent_resident_util_mean_percent:s.pooled_mean_percent,independent_resident_util_peak_percent:s.peak_percent,util_repeat_means:s.repeat_means,util_samples:s.samples,util_duration_s:s.duration_s,util_forward_calls:s.forwards,util_sum_percent_points:s.sum_percent_points,measured_local:s.measured_local,utilization_source:s.source,utilization_profile:'single model load with complete ascending predecessor batches; native queue8'});
 }
 historicalRepeats.push(...seq.repeats.filter(x=>!((x.baseline==='yolo'&&x.batch===3)||(x.baseline==='clip'&&x.batch===10))).map(x=>({...x,role:'Sequential predecessor batch; retained audit, not replacement of finalized B1 or cost'})));
 newCount=2;
}
if(rental){
 const extra=await json(path.join(rentalRun,'verified_figure5_extension.json'));
 if(extra.status!=='complete'||extra.paired_rows.length!==5||extra.repeats.length!==15||!extra.no_original_point_replacement)throw Error('Incomplete supplementary hardware evidence');
 data.paired_rows.push(...extra.paired_rows.filter(r=>includeCrossDeviceAudit||r.baseline!=='qwen80'));
 data.repeats.push(...extra.repeats.filter(r=>includeCrossDeviceAudit||r.baseline!=='qwen80'));
 historicalRepeats.push(...extra.repeats.filter(r=>!includeCrossDeviceAudit&&r.baseline==='qwen80').map(r=>({...r,role:'Audit-only cross-device Qwen; explicitly excluded from paper by user on 17 October'})));
 newCount=includeCrossDeviceAudit?5:4;
}
const palette={ours:['#7828A8','#F2E9FA'],qwen:['#0072B2','#EAF3FA'],glm:['#92527F','#F5EBF3'],small:['#C25B18','#FFF0E5'],clip:['#007F63','#E8F5EF'],yolo:['#9B7000','#FFF7DD'],deepseek:['#5B6470','#EEF0F3'],...(rental?{full:['#79502C','#F6EEDC'],kimi:['#B64058','#FCECEF'],qwen80:['#256F99','#E9F2F8']}:{})};
const auditPalette={...palette};
if(!includeCrossDeviceAudit)delete palette.qwen80;
const cleanData=cleanKimi?await json(path.join(rentalRun,'verified_kimi_clean_extension.json')):null;
if(cleanKimi){
 if(cleanData.status!=='complete'||cleanData.paired_rows.length!==2||cleanData.repeats.length!==18||cleanData.paced_service.length!==2||!cleanData.historical_B1_all558_exact)throw Error('Incomplete clean Kimi delivery');
 historicalRepeats.push(...data.repeats.filter(x=>x.baseline==='kimi').map(x=>({...x,role:'Historical superseded Kimi rental session; not pooled with clean same-card run'})));
 data.paired_rows=[...data.paired_rows.filter(x=>x.baseline!=='kimi'),...cleanData.paired_rows];
 data.repeats=[...data.repeats.filter(x=>x.baseline!=='kimi'),...cleanData.repeats];
 newCount=2;
}
const asNumber=v=>v===''||v===undefined||v===null?null:Number(v);
const anchor=data.historical_qwen_B1;
if(+anchor.time_16_videos_ms!==1188.3861464262009||+anchor.energy_16_videos_j!==462.1134010767563||+anchor.display_gpu_util_mean_percent!==63.133111480865225)throw Error('Historical anchor drift');
const checkpoint={qwen:'e102adc6a7b33335c8f49ae1cbeea3302a1a10017d282da8d6091a640276d0d2'};
const oldMeasurements=plot.worksheets.getItem('Measurements');
const oldOursSource=oldMeasurements.getRange('S5').values[0][0];
const historicalFile=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_utilization_execution_20261015/unified_qwen_b1_5s30s01/report.json');
const historical=await json(historicalFile),h=historical.measurement;
const rows=[{
 model:'LightGen-II',baseline:'ours',batch_size_videos:null,videos_per_optical_field:16,workload_videos:16,model_total_parameters_billion:null,moe_active_parameters_billion:null,
 measured_full558_srcc:data.ours.physical_reference_srcc,display_gpu_util_mean_percent:null,time_16_videos_ms:data.ours.time_16_videos_ms,energy_16_videos_j:data.ours.energy_16_videos_j,
 lightgen_time_speedup_x:1,lightgen_energy_efficiency_gain_x:1,same_model_time_gain_vs_b1_x:null,same_model_energy_gain_vs_b1_x:null,independent_resident_util_mean_percent:null,same_round_active_util_mean_percent:null,
 utilization_basis:'Not applicable: N=16 optical field',utilization_source:'',parameter_basis:'Optical degrees of freedom are not stored GPU model weights',parameter_source:'',cost_basis:'Unchanged physical reference and component-cost estimate',execution_profile:'physical optical system',cost_source:oldOursSource,checkpoint_sha256:'',cost_measured_local:'historical physical reference',util_measured_local:''
},{...anchor,batch_size_videos:1,videos_per_optical_field:null,workload_videos:16,model_total_parameters_billion:2,moe_active_parameters_billion:null,
 measured_full558_srcc:+anchor.measured_full558_srcc,display_gpu_util_mean_percent:+anchor.display_gpu_util_mean_percent,time_16_videos_ms:+anchor.time_16_videos_ms,energy_16_videos_j:+anchor.energy_16_videos_j,
 lightgen_time_speedup_x:+anchor.lightgen_time_speedup_x,lightgen_energy_efficiency_gain_x:+anchor.lightgen_energy_efficiency_gain_x,same_model_time_gain_vs_b1_x:1,same_model_energy_gain_vs_b1_x:1,
 independent_resident_util_mean_percent:+anchor.display_gpu_util_mean_percent,historical_B1_three_repeat_audit_percent:+anchor.independent_resident_util_mean_percent,historical_B1_three_repeat_audit_source:oldMeasurements.getRange('T6').values[0][0],same_round_active_util_mean_percent:null,execution_profile:'historical native Qwen B1',cost_basis:'Protected historical anchor, not remeasured in this run',
 utilization_source:path.relative(repo,historicalFile).replaceAll('\\','/'),cost_source:oldMeasurements.getRange('S6').values[0][0],checkpoint_sha256:checkpoint.qwen,cost_measured_local:'historical finalized anchor',util_measured_local:'2026-10-15 14:52:05'}];
const b1=Object.fromEntries(data.paired_rows.filter(x=>x.batch_size===1).map(x=>[x.baseline,x]));
b1.qwen={time_16_videos_ms:+anchor.time_16_videos_ms,energy_16_videos_j:+anchor.energy_16_videos_j,srcc:+anchor.measured_full558_srcc};
const displayCandidates=sequential?data.paired_rows.filter(x=>!(x.baseline==='yolo'&&x.batch_size>3)&&!(x.baseline==='clip'&&x.batch_size>10)):data.paired_rows;
const selectedKeys=chooseDisplayPoints(displayCandidates);
for(const r of data.paired_rows){
 if(!selectedKeys.has(r.baseline+':'+r.batch_size))continue;
 const meta=r.parameter_metadata;
 rows.push({model:r.model,baseline:r.baseline,batch_size_videos:r.batch_size,videos_per_optical_field:null,workload_videos:16,
  model_total_parameters_billion:asNumber(meta.model_total_parameters_billion),moe_active_parameters_billion:asNumber(meta.moe_active_parameters_billion),
  measured_full558_srcc:r.srcc,display_gpu_util_mean_percent:r.independent_resident_util_mean_percent,time_16_videos_ms:r.time_16_videos_ms,energy_16_videos_j:r.energy_16_videos_j,
  lightgen_time_speedup_x:r.time_16_videos_ms/rows[0].time_16_videos_ms,lightgen_energy_efficiency_gain_x:r.energy_16_videos_j/rows[0].energy_16_videos_j,
  same_model_time_gain_vs_b1_x:b1[r.baseline].time_16_videos_ms/r.time_16_videos_ms,same_model_energy_gain_vs_b1_x:b1[r.baseline].energy_16_videos_j/r.energy_16_videos_j,
  independent_resident_util_mean_percent:r.independent_resident_util_mean_percent,same_round_active_util_mean_percent:r.gpu_util_mean_percent,
  utilization_basis:r.utilization_profile??'Independent resident queue8, three complete 5s/30s repeats, all samples pooled',utilization_source:r.utilization_source,parameter_basis:meta.parameter_basis,parameter_source:meta.parameter_source,
  cost_basis:'Same-run first200 synchronized wall and active board-power estimate; '+r.execution,execution_profile:r.execution,cost_source:r.source,checkpoint_sha256:r.checkpoint_sha256,cost_measured_local:r.cost_measured_local,util_measured_local:r.measured_local,
  ...(rental?{hardware_platform:r.hardware_platform??'A100-PCIE-40GB',power_limit_w:r.power_limit_w??250,gpu_uuid:r.gpu_uuid??'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55',hardware_session:r.hardware_session??'original PCIe measurements',same_card_batch_ablation:r.same_card_batch_ablation??null}:{})});
}
for(const model of Object.keys(palette).filter(k=>k!=='ours')){const count=rows.filter(x=>x.baseline===model).length;if(count<(model==='qwen80'?1:2)||count>3)throw Error('Invalid displayed count per GPU model');}
if(rows.length<13||rows.length>(rental?20:18))throw Error('Invalid sparse display');
if(rental){rows[0].hardware_platform='Optical system';rows[1].hardware_platform='A100-PCIE-40GB';rows[1].power_limit_w=250;rows[1].gpu_uuid='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55';}
for(const x of rows){const r=data.paired_rows.find(v=>v.baseline===x.baseline&&v.batch_size===x.batch_size_videos);Object.assign(x,{cost_wall_sum_ms:r?.cost_wall_sum_ms??null,cost_video_count:r?200:null,cost_call_count:r?.cost_calls??null,cost_active_A100_mean_W:r?.board_mean_w??null,cost_active_samples:r?.active_samples??null,util_repeat1_mean_percent:r?.util_repeat_means[0]??(x.baseline==='qwen'&&x.batch_size_videos===1?h.mean_gpu_utilization_percent:null),util_repeat2_mean_percent:r?.util_repeat_means[1]??null,util_repeat3_mean_percent:r?.util_repeat_means[2]??null,util_measurement_seconds:r?.util_duration_s??(x.baseline==='qwen'&&x.batch_size_videos===1?h.duration_s:null),util_samples:r?.util_samples??(x.baseline==='qwen'&&x.batch_size_videos===1?h.samples:null),util_repeat_spread_pp:r?Math.max(...r.util_repeat_means)-Math.min(...r.util_repeat_means):null,util_unstable_repeat_warning:r?Math.max(...r.util_repeat_means)-Math.min(...r.util_repeat_means)>5:null});}
const namesToArchive=[...filenames,'plot_data.csv','LGVQ_temporal_gpu_utilization.csv','LGVQ_temporal_gpu_utilization_repeats.csv','timing_energy_recalculated.csv','batch_metric_audit.csv','README.md','UTILIZATION_METHOD_20261015.md','latency_comparison_reference.png','latency_comparison_reference.svg','energy_comparison_reference.png','energy_comparison_reference.svg','srcc_cost_comparison_reference.png','srcc_cost_comparison_reference.svg','gpu_utilization_comparison.png','gpu_utilization_comparison.svg','batch_cost_comparison.png','batch_cost_comparison.svg'];
function polishReadability(){
 const input=plot.worksheets.getItem('Measurements'),s=main.worksheets.getItem('Summary'),src=main.worksheets.getItem('Sources'),u=util.worksheets.getItem('GPU utilization'),repeat=util.worksheets.getItem('Repeat measurements');
 input.getRange(`B5:L${rows.length+4}`).format.horizontalAlignment='right';
 input.getRange(`B5:C${rows.length+4}`).setNumberFormat('0.#########');input.getRange(`E5:E${rows.length+4}`).setNumberFormat('0.00000000');
 input.getRange(`F5:L${rows.length+4}`).setNumberFormat('#,##0.000000');
 s.getRange(`B6:T${rows.length+5}`).format.horizontalAlignment='right';
 s.getRange('E5').values=[['Estimated energy\nper 16 videos (J)']];s.getRange('E5').format.wrapText=true;
 s.getRange('J5:L5').values=[['First 200 TEST SRCC\n(audit)','Independent resident\nGPU utilization (%)','Independent resident\npeak utilization (%)']];
 s.getRange('A5:T5').format.wrapText=true;s.getRange('A5:T5').format.rowHeight=62;
 src.getRange('A25:B25').format.rowHeight=10;
 src.getRange('A26:B31').format={fill:'#F2F6FA',font:{name:'Arial',size:10,color:'#243447',bold:false},wrapText:true,verticalAlignment:'center',rowHeight:50};
 src.getRange('A26:A31').format.font={name:'Arial',size:10,bold:true,color:'#243447'};
 u.getRange(`E7:E${rows.length+5}`).setNumberFormat('0.000');u.getRange(`F7:G${rows.length+5}`).setNumberFormat('#,##0');
 repeat.getRange(`F5:N${data.repeats.length+historicalRepeats.length+5}`).format.horizontalAlignment='right';
 if(rental){
  s.getRange('U5:W5').values=[['GPU platform','Power limit (W)','Physical GPU UUID']];
  s.getRange(`U6:W${rows.length+5}`).values=rows.map(x=>[x.hardware_platform??'Optical system',x.power_limit_w??'n.a.',x.gpu_uuid??'n.a.']);
  s.getRange(`U5:W${rows.length+5}`).format={font:{name:'Arial',size:10,color:'#243447'},wrapText:true};s.getRange('U5:W5').format={fill:'#105A85',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'}};s.getRange('U5:W40').format.columnWidth=29;
  for(const[i,x]of rows.filter(x=>x.baseline!=='ours').entries())if(['full','kimi','qwen80'].includes(x.baseline))u.getRange('H'+(i+7)).values=[['Native; SXM4-80GB /400W']];
 }
}
function writeKimiTrend(){
 util.worksheets.getItem('GPU utilization').getRange('A28:G38').unmerge();util.worksheets.getItem('GPU utilization').getRange('A28:G38').clear({applyTo:'all'});
 let trend;try{trend=util.worksheets.getItem('Kimi native trend');}catch{trend=util.worksheets.add('Kimi native trend');}
 trend.getRange('A1:G12').format={font:{name:'Arial',size:10,color:'#243447'},rowHeight:27,verticalAlignment:'center'};
 trend.getRange('A1:F1').merge();trend.getRange('A1').values=[['Kimi: same-card native resident GPU utilization']];trend.getRange('A1:F1').format={fill:'#B64058',font:{name:'Arial',size:14,bold:true,color:'#FFFFFF'},rowHeight:35};
 trend.getRange('A2:F2').merge();trend.getRange('A2').values=[['One model process, B1 through B6; three complete queue8 repeats per batch. Every forward processes B videos.']];
 trend.getRange('A3:F3').merge();trend.getRange('A3').values=[['SXM4-80GB /400W. B2–5 are utilization-only evidence, not new accuracy or cost measurements.']];trend.getRange('A2:F3').format={wrapText:true,rowHeight:37};
 trend.getRange('A5:G5').values=[['Video batch B','Pooled mean (%)','Repeat1 (%)','Repeat2 (%)','Repeat3 (%)','Full558 accuracy/cost','Raw source']];
 trend.getRange('A6:G11').values=cleanData.utilization_only_trend.map(x=>[x.batch,x.mean_percent,...x.repeat_means,x.full558_accuracy_available?'Measured':'Not measured',x.source]);
 trend.getRange('A5:G5').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:44};trend.getRange('A6:G11').format.fill='#FCECEF';
 trend.getRange('A1:A12').format.columnWidth=14;trend.getRange('B1:E12').format.columnWidth=19;trend.getRange('F1:F12').format.columnWidth=28;trend.getRange('G1:G12').format.columnWidth=85;trend.getRange('B6:E11').setNumberFormat('0.00000');trend.getRange('A6:F11').format.horizontalAlignment='right';
}
async function renderQA(){
 if(cleanKimi){for(const [book,sheetName,range,file]of[[plot,'Paced service','A1:F11','selected_service'],[util,'Kimi native trend','A1:F12','selected_kimi_trend']]){const image=await book.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(scratch,file+'.png'),new Uint8Array(await image.arrayBuffer()));}}
 for(const [book,label,sheetName,range]of[[plot,'selected_plot','Comparison',`A1:${rental?'M':'L'}${rows.length+12}`],[plot,'selected_inputs','Measurements',`A1:L${rows.length+4}`],[main,'selected_summary','Summary',`A1:L${rows.length+5}`],[main,'selected_build','Batch results',`A1:S${rows.length+1}`],[main,'selected_sources','Sources',rental?'A26:B36':'A19:B31'],[util,'selected_util','GPU utilization',`A1:I${rows.length+5}`],[util,'selected_repeats','Repeat measurements',rental?'A60:O74':'A1:O18']]){const image=await book.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(scratch,label+'.png'),new Uint8Array(await image.arrayBuffer()));}
}
if(process.argv.includes('--polish')){
 if(cleanKimi)writeKimiTrend();
 polishReadability();for(const book of books)book.recalculate();await renderQA();
 for(let i=0;i<books.length;i++)await(await SpreadsheetFile.exportXlsx(books[i])).save(path.join(out,filenames[i]));
 console.log(JSON.stringify({workbooks_polished:3,measurement_values_unchanged:true}));process.exit(0);
}
if(process.argv.includes('--verify')){
 if(cleanKimi){const service=plot.worksheets.getItem('Paced service'),expanded=service.getRange('G6').values[0][0]==='Service T16 (ms)';for(const x of cleanData.paced_service){const index=service.getRange('A7:B20').values.findIndex(v=>v[0]===x.model&&v[1]===x.batch);if(index<0)throw Error('Missing saved Kimi service');for(const[c,k]of [[expanded?'G':'D','service_T16_ms'],[expanded?'H':'E','service_E16_j']])if(Math.abs(service.getRange(c+(index+7)).values[0][0]-x[k])>1e-8)throw Error('Saved service mismatch');}}
 const p=plot.worksheets.getItem('Comparison');
 for(let i=0;i<rows.length;i++){const r=i+7,x=rows[i];for(const[c,k]of[['E','measured_full558_srcc'],['G','time_16_videos_ms'],['H','energy_16_videos_j'],['I','lightgen_time_speedup_x'],['J','lightgen_energy_efficiency_gain_x']])if(Math.abs(p.getRange(c+r).values[0][0]-x[k])>1e-8)throw Error('Saved plot mismatch '+c+r);if(i&&Math.abs(p.getRange('F'+r).values[0][0]*100-x.display_gpu_util_mean_percent)>1e-8)throw Error('Saved util mismatch');}
 const savedSummary=main.worksheets.getItem('Summary'),savedUtil=util.worksheets.getItem('GPU utilization');
 for(let i=0;i<rows.length;i++)for(const[c,k]of[['C','measured_full558_srcc'],['D','time_16_videos_ms'],['E','energy_16_videos_j'],['G','lightgen_time_speedup_x'],['H','lightgen_energy_efficiency_gain_x']])if(Math.abs(savedSummary.getRange(c+(i+6)).values[0][0]-rows[i][k])>1e-8)throw Error('Saved main mismatch');
 for(const[i,x]of rows.filter(x=>x.baseline!=='ours').entries())if(Math.abs(savedUtil.getRange('C'+(i+7)).values[0][0]*100-x.display_gpu_util_mean_percent)>1e-8)throw Error('Saved utilization mismatch');
 for(const book of books){const errors=(await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson;if(!errors.includes('Cell search matched 0 entries.'))throw Error('Saved workbook formula error: '+errors);}
 console.log(JSON.stringify({workbooks_reimported_verified:3,saved_rows_verified:rows.length,sparse_GPU_points:rows.length-1,Qwen_B1_preserved:true,formula_errors:0}));process.exit(0);
}
if(!process.argv.includes('--build'))throw Error('Use --inspect, --build or --verify');
const archive=path.join(run,'prior_report_archive');await fs.mkdir(archive,{recursive:true});const archived=[];
for(const name of namesToArchive){const input=path.join(out,name),bytes=await fs.readFile(input),target=path.join(archive,name);try{await fs.copyFile(input,target,fs.constants.COPYFILE_EXCL);}catch(error){if(error.code!=='EEXIST'||digest(await fs.readFile(target))!==digest(bytes))throw error;}archived.push({name,bytes:bytes.length,sha256:digest(bytes)});}
const archiveManifest=JSON.stringify({files:archived},null,2)+'\n';try{await fs.writeFile(path.join(archive,'manifest.json'),archiveManifest,{flag:'wx'});}catch(error){if(error.code!=='EEXIST'||await fs.readFile(path.join(archive,'manifest.json'),'utf8')!==archiveManifest)throw error;}
const comparison=plot.worksheets.getItem('Comparison'),inputs=plot.worksheets.getItem('Measurements'),summary=main.worksheets.getItem('Summary'),detail=main.worksheets.getItem('Batch results'),sources=main.worksheets.getItem('Sources'),uv=util.worksheets.getItem('GPU utilization'),ur=util.worksheets.getItem('Repeat measurements');
const na=v=>v===undefined||v===null||v===''?'n.a.':v;
const rawMap=new Map(data.paired_rows.map(x=>[x.baseline+':'+x.batch_size,x]));
const getRaw=x=>rawMap.get(x.baseline+':'+x.batch_size_videos);
comparison.getRange('A7:L44').unmerge();comparison.getRange('A7:L44').clear({applyTo:'all'});
comparison.getRange('A1').values=[['LGVQ 时间质量评价：精度与推理成本']];inputs.getRange('A1').values=[['Measurements and provenance']];
inputs.getRange('A5:Y40').clear({applyTo:'all'});summary.getRange('A6:W43').clear({applyTo:'all'});detail.getRange('A2:AR38').clear({applyTo:'all'});
comparison.getRange('M7:M44').clear({applyTo:'all'});
uv.getRange('A7:Q40').clear({applyTo:'all'});ur.getRange('A5:R160').clear({applyTo:'all'});
comparison.getRange('A2').values=[[rental?'原A100 PCIe40GB/250W；新增SXM4 80GB/400W。558 TEST精度，首200视频成本归一至16视频。':'A100 250 W。558 TEST 计算精度，首200视频成本归一到16视频。每模型仅展示 B1 及1–2个近满载点。']];
comparison.getRange('A3').values=[['Qwen B1 定稿不变。Qwen B3/4 为严格等价验证的优化执行，其余模型原生执行；不能把二者差异解释为纯 batch 效应。']];
comparison.getRange('A4').values=[['GPU 利用率为独立驻留测量，不代入成本功率。时延及能耗估计使用同轮交错推理数据。历史完整表和中间点已归档。']];
comparison.getRange('H6').values=[['16 视频\n能耗估计 (J)']];summary.getRange('E5').values=[['16-video estimated energy (J)']];detail.getRange('P1').values=[['16-video estimated energy (J)']];
inputs.getRange('A2').values=[['Columns F/K distinguish selected utilization from the separate historical B1 audit. Timing/power are paired; resident utilization is independent.']];
inputs.getRange('K4').values=[['Separate three-repeat resident audit (%)']];
summary.getRange('A3').values=[['Selected B1 and near-saturation points. Historical Qwen B1 retained; optimized Qwen B3/4 separately identified. Resident utilization is not cost telemetry.']];
summary.getRange('T5').values=[['Separate resident audit (%)']];detail.getRange('AQ1').values=[['Separate resident audit mean (%)']];uv.getRange('Q6').values=[['Separate resident audit mean (%)']];
ur.getRange('A3').values=[['Three complete 5s/30s queue8 repeats for each new selected point. Historical Qwen B1 retained separately. Resident power is not used for cost energy.']];
uv.getRange('A3').values=[['Selected B1 and near-saturation settings; historical Qwen B1 63.13% is retained, other rows are new three-repeat pooled means.']];
uv.getRange('A4').values=[['Qwen B3/4: equivalence-validated execution optimization. Other models: native. Same resident queue8 measurement method; no Graph or clock changes.']];
const inputValues=[],detailValues=[];
for(let i=0;i<rows.length;i++){
 const x=rows[i],raw=getRaw(x),retained=!raw,base=b1[x.baseline];
 inputValues.push([x.model,na(x.model_total_parameters_billion),na(x.moe_active_parameters_billion),x.baseline==='ours'?'N=16':x.batch_size_videos,x.measured_full558_srcc,na(x.display_gpu_util_mean_percent),x.time_16_videos_ms,x.energy_16_videos_j,na(base?.time_16_videos_ms),na(base?.energy_16_videos_j),na(x.historical_B1_three_repeat_audit_percent??x.independent_resident_util_mean_percent),na(x.same_round_active_util_mean_percent),x.utilization_basis,x.utilization_source,x.parameter_basis,x.parameter_source,x.cost_basis,x.checkpoint_sha256,x.cost_source,x.historical_B1_three_repeat_audit_source??x.utilization_source]);
 detailValues.push([x.model,x.baseline==='ours'?16:x.batch_size_videos,x.baseline==='ours'?x.measured_full558_srcc:base.srcc,retained?'n.a.':200,na(raw?.cost_calls),na(raw?.wall_mean_ms),na(raw?.wall_p95_ms),na(raw?.board_mean_w),retained?'n.a.':338.2,x.baseline==='ours'?1:16/x.batch_size_videos,retained?1:0,retained?x.time_16_videos_ms:null,retained?x.energy_16_videos_j:null,null,null,null,null,null,null,x.cost_source,x.measured_full558_srcc,na(raw?.first200_srcc),x.execution_profile,'Original BF16/FP16 backbone and FP32 readout',x.energy_16_videos_j/16,'n.a.',na(raw?.cost_wall_sum_ms),na(raw?.gpu_util_mean_percent),na(raw?.gpu_util_peak_percent),na(raw?.memory_mean_mib),na(raw?.memory_peak_mib),na(raw?.active_samples),na(x.display_gpu_util_mean_percent),raw?raw.independent_resident_util_peak_percent:(x.baseline==='qwen'?h.peak_gpu_utilization_percent:'n.a.'),x.util_measured_local,x.utilization_source,x.cost_basis,retained?'n.a.':x.time_16_videos_ms,retained?'n.a.':x.energy_16_videos_j,na(raw?.plcc),na(x.model_total_parameters_billion),na(x.moe_active_parameters_billion),na(x.historical_B1_three_repeat_audit_percent??x.independent_resident_util_mean_percent),x.utilization_basis]);
}
inputs.getRange(`A5:T${rows.length+4}`).values=inputValues;detail.getRange(`A2:AR${rows.length+1}`).values=detailValues;
for(let i=0;i<rows.length;i++){
 const x=rows[i],r=i+7,ir=i+5,dr=i+2,sr=i+6,raw=getRaw(x),colors=palette[x.baseline];
 comparison.getRange(`A${r}:H${r}`).formulas=[[`=Measurements!A${ir}`,x.model_total_parameters_billion===null?'="n.a."':`=Measurements!B${ir}${x.model_total_parameters_billion<1?'*1000':''}`,x.moe_active_parameters_billion===null?'="n.a."':`=Measurements!C${ir}`,`=Measurements!D${ir}`,`=Measurements!E${ir}`,x.baseline==='ours'?'="n.a."':`=Measurements!F${ir}/100`,`=Measurements!G${ir}`,`=Measurements!H${ir}`]];
 comparison.getRange(`I${r}:L${r}`).formulas=[[`=G${r}/$G$7`,`=H${r}/$H$7`,x.baseline==='ours'?'="n.a."':`=Measurements!I${ir}/G${r}`,x.baseline==='ours'?'="n.a."':`=Measurements!J${ir}/H${r}`]];
 detail.getRange(`N${dr}:S${dr}`).formulas=[[raw?`=H${dr}+I${dr}`:'="n.a."',raw?`=AA${dr}*16/D${dr}`:`=L${dr}`,raw?`=N${dr}*O${dr}/1000`:`=M${dr}`,`=16/P${dr}`,`=O${dr}/$O$2`,`=P${dr}/$P$2`]];
 if(raw)detail.getRange(`F${dr}`).formulas=[[`=AA${dr}/E${dr}`]];
 detail.getRange(`Y${dr}`).formulas=[[`=P${dr}/16`]];
 summary.getRange(`A${sr}:T${sr}`).formulas=[[`='Batch results'!A${dr}`,x.baseline==='ours'?'="N=16"':`='Batch results'!B${dr}`,`='Batch results'!U${dr}`,`='Batch results'!O${dr}`,`='Batch results'!P${dr}`,`=16/E${sr}`,`=D${sr}/$D$6`,`=E${sr}/$E$6`,`='Batch results'!C${dr}`,`='Batch results'!V${dr}`,`='Batch results'!AG${dr}`,`='Batch results'!AH${dr}`,`='Batch results'!AB${dr}`,`='Batch results'!AE${dr}`,`='Batch results'!AK${dr}`,`='Batch results'!T${dr}`,`='Batch results'!AO${dr}`,`='Batch results'!AP${dr}`,`='Batch results'!AR${dr}`,`='Batch results'!AQ${dr}`]];
 for(const[sh,range]of[[comparison,`A${r}:L${r}`],[inputs,`A${ir}:T${ir}`],[summary,`A${sr}:T${sr}`],[detail,`A${dr}:AR${dr}`]]){sh.getRange(range).format={fill:colors[1],font:{name:'Arial',size:10,color:'#243447'},rowHeight:27,verticalAlignment:'center'};}
 comparison.getRange(`A${r}`).format.font={name:'Arial',size:10,bold:true,color:colors[0]};comparison.getRange(`I${r}:J${r}`).format.font={name:'Arial',size:10,bold:true,color:colors[0]};
 comparison.getRange(`B${r}`).setNumberFormat(x.model_total_parameters_billion<1?'0.0" M"':'0.00" B"');comparison.getRange(`C${r}`).setNumberFormat('0.0" B"');
 if(rental&&['full','kimi','qwen80'].includes(x.baseline))comparison.getRange(`A${r}`).format={wrapText:true,rowHeight:38};
}
if(rental){
 comparison.getRange('M6').values=[['GPU platform /\npower limit']];comparison.getRange('M6').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center'};comparison.getRange('M6:M40').format.columnWidth=27;
 for(let i=0;i<rows.length;i++){const x=rows[i];comparison.getRange('M'+(i+7)).values=[[x.baseline==='ours'?'Optical system':x.hardware_platform.replace('A100-','')+' / '+x.power_limit_w+' W']];comparison.getRange('M'+(i+7)).format={fill:palette[x.baseline][1],font:{name:'Arial',size:10,color:'#243447'},verticalAlignment:'center'};}
 inputs.getRange('U4:Y4').values=[['GPU platform','Power limit (W)','Physical GPU UUID','Hardware session','Same-card batch-only ablation']];
 inputs.getRange(`U5:Y${rows.length+4}`).values=rows.map(x=>[x.hardware_platform??'Optical system',x.power_limit_w??'n.a.',x.gpu_uuid??'n.a.',x.hardware_session??'retained historical reference',x.same_card_batch_ablation??'Not established']);
 inputs.getRange(`U4:Y${rows.length+4}`).format={font:{name:'Arial',size:10,color:'#243447'},wrapText:true};inputs.getRange('U4:Y4').format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};inputs.getRange('U4:Y4').format.fill='#214D68';inputs.getRange('U4:Y40').format.columnWidth=26;
}
comparison.getRange(`E7:E${rows.length+6}`).setNumberFormat('0.0000');comparison.getRange(`F7:F${rows.length+6}`).setNumberFormat('0.00%');comparison.getRange(`G7:H${rows.length+6}`).setNumberFormat('#,##0.000');comparison.getRange(`I7:L${rows.length+6}`).setNumberFormat('0.00"×"');comparison.getRange(`B7:L${rows.length+6}`).format.horizontalAlignment='right';
for(const[sh,range]of[[summary,`C6:J${rows.length+5}`],[detail,`F2:S${rows.length+1}`]])sh.getRange(range).setNumberFormat('#,##0.000');
summary.getRange(`C6:C${rows.length+5}`).setNumberFormat('0.000000');summary.getRange(`K6:M${rows.length+5}`).setNumberFormat('0.00');
summary.getRange(`I6:J${rows.length+5}`).setNumberFormat('0.000000');detail.getRange(`U2:V${rows.length+1}`).setNumberFormat('0.000000');detail.getRange(`AN2:AN${rows.length+1}`).setNumberFormat('0.000000');detail.getRange(`C2:C${rows.length+1}`).setNumberFormat('0.000000');
const notes=['Qwen 40GB B1：历史原生定稿锚点不变。租赁80GB Qwen仅留审计，不进入论文图表。','LightGen-II 时间和能耗优势分别为 baseline T16 / LightGen-II T16、baseline E16 / LightGen-II E16。',cleanKimi?'Kimi B1/B6为同一80GB卡/400W；DeepSeek完整版本B1/B2来自不同物理卡。50%服务成本见独立页。':rental?'80GB租赁B1与B2来自同SKU的不同物理卡/会话，不是严格同卡batch-only；Kimi B2利用率86.11%，未达98%。':'同模型相对 B1：Qwen 同时包含 batch 与执行优化，其余模型为原生不同 batch 对照。','利用率为独立驻留负载的驱动窗口活跃比例，不是 SM occupancy，也不用于推算交错推理能耗。','本轮只冻结推理。既有 PT 若曾按 TEST 选模，相关精度仍属开发口径（无独立 VAL），并非新的独立泛化结论。'];
for(let i=0;i<notes.length;i++){const r=rows.length+8+i;comparison.getRange(`A${r}:L${r}`).merge();comparison.getRange(`A${r}`).values=[[notes[i]]];comparison.getRange(`A${r}:L${r}`).format={font:{name:'Microsoft YaHei',size:10,color:'#53687A'},rowHeight:25};}
const urows=rows.filter(x=>x.baseline!=='ours');
for(let i=0;i<urows.length;i++){
 const x=urows[i],raw=getRaw(x),r=i+7,means=raw?.util_repeat_means??[],samples=raw?.util_samples??h.samples,sum=raw?.util_sum_percent_points??(h.mean_gpu_utilization_percent*h.samples);
 uv.getRange(`A${r}:Q${r}`).values=[[x.model,x.batch_size_videos,null,raw?raw.independent_resident_util_peak_percent/100:h.peak_gpu_utilization_percent/100,raw?.util_duration_s??h.duration_s,samples,raw?.util_forward_calls??historical.measured_forward_invocations,raw?(raw.utilization_profile?'Native sequential model process':x.execution_profile):'Historical native, one retained run',new Date(x.util_measured_local.replace(' ','T')+'Z'),null,x.utilization_source,sum,na(means[0]),na(means[1]),na(means[2]),means.length?Math.sqrt(means.reduce((s,v)=>s+(v-means.reduce((a,b)=>a+b)/3)**2,0)/2):'n.a.',x.historical_B1_three_repeat_audit_percent??x.independent_resident_util_mean_percent]];
 uv.getRange(`C${r}`).formulas=[[`=L${r}/F${r}/100`]];uv.getRange(`A${r}:Q${r}`).format={fill:palette[x.baseline][1],font:{name:'Arial',size:10,color:'#243447'},rowHeight:26};
}
uv.getRange(`C7:D${urows.length+6}`).setNumberFormat('0.00%');uv.getRange(`I7:I${urows.length+6}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
const historicRepeat={model:'Qwen3-VL-2B',baseline:'qwen',batch:1,repeat:1,mean_percent:h.mean_gpu_utilization_percent,peak_percent:h.peak_gpu_utilization_percent,duration_s:h.duration_s,samples:h.samples,forwards:historical.measured_forward_invocations,power_w:h.mean_board_power_w,memory_mib:h.mean_memory_used_mib,warmup_s:historical.warmup.duration_s,measured_local:'2026-10-15 14:52:05',source:path.relative(repo,historicalFile).replaceAll('\\','/'),role:'Historical native B1'};
const repeatRows=[historicRepeat,...data.repeats.map(x=>({...x,role:selectedKeys.has(x.baseline+':'+x.batch)?'Displayed point; see source/date':'Audit only; not in main figure'})),...historicalRepeats];
ur.getRange(`A5:R${repeatRows.length+4}`).values=repeatRows.map(x=>[x.model,x.batch,x.repeat,x.mean_percent/100,x.peak_percent/100,x.duration_s,x.samples,x.forwards,x.power_w,x.memory_mib,na(x.sm_clock_mhz),na(x.temperature_c),na(x.cpu_queue_wait_ms),x.warmup_s,new Date(x.measured_local.replace(' ','T')+'Z'),x.baseline,x.role??'New three-repeat measurement',x.source]);
ur.getRange('P4:R4').values=[['Baseline key','Measurement identity','Raw source']];ur.getRange(`D5:E${repeatRows.length+4}`).setNumberFormat('0.00%');ur.getRange(`O5:O${repeatRows.length+4}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
for(let i=0;i<repeatRows.length;i++)ur.getRange(`A${i+5}:R${i+5}`).format={fill:auditPalette[repeatRows[i].baseline][1],font:{name:'Arial',size:10,color:'#243447'},rowHeight:25};
ur.getRange(`D5:E${repeatRows.length+4}`).setNumberFormat('0.00%');ur.getRange(`F5:N${repeatRows.length+4}`).setNumberFormat('0.00');ur.getRange(`G5:H${repeatRows.length+4}`).setNumberFormat('#,##0');ur.getRange(`O5:O${repeatRows.length+4}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
sources.getRange('A7:B7').values=[['Measured candidate batches',data.paired_rows.map(x=>x.baseline+':'+x.batch_size).join(', ')+'. Historical Qwen1 retained. Main view applies the95% rule.']];
sources.getRange('A19:B20').values=[['Resident utilization','Independent unchanged queue8 three-repeat 5s/30s method; no Graph. Qwen B1 is one historical retained run.'],['Cost run','lgvq_selected_execution_a100_20261016;16 new sparse points, preserved historical Qwen B1']];
if(recheck)sources.getRange('B20').values=[['Original selected round plus lgvq_yolo_clip_recheck_a100_20261017. Latest prescribed YOLO3/CLIP10 replaces previous round regardless of direction; conditional adjacent4/11.']];
if(sequential)sources.getRange('B20').values=[['Costs/accuracy retain original selected/recheck sources. YOLO3/CLIP10 independent utilization: lgvq_yolo_clip_sequential_a100_20261017; same-model process and ascending predecessors. No new cost measurement.']];
sources.getRange('A14:B14').values=[['Performance','Full558 evaluated at each batch. Previously TEST-selected checkpoints retain their development/no-VAL interpretation; no new training or selection.']];
sources.getRange('A23:B24').values=[['Dates','See each point source/driver telemetry for actual +0800 dates'],['Historical evidence','lgvq_selected_execution_a100_20261016/prior_report_archive/manifest.json']];
if(recheck)sources.getRange('B24').values=[['lgvq_yolo_clip_recheck_a100_20261017/prior_report_archive/manifest.json. Previous selected round and all original repeats retained.']];
if(sequential)sources.getRange('B24').values=[['lgvq_yolo_clip_sequential_a100_20261017/prior_report_archive/manifest.json. Old fresh-process and sequential predecessor repeats remain separately identified.']];
if(rental){
 sources.getRange('A32:B36').values=[['Supplementary platform','DeepSeek-VL2 full / Kimi B1,B2 and native Qwen B1 use A100-SXM4-80GB /400W. Original PCIe40GB /250W data retained. Not a common-hardware speed comparison.'],['Physical GPU sessions','Rental DS/Kimi B1: GPU-780326a6-79bc-7a45-36f6-683c5b278bee; B2/Qwen: GPU-d06c95d0-7066-9768-0f6e-fc1ee877f9f9. Same SKU but distinct physical cards.'],['Qwen cross-device control','Same revision/head. First TEST tensor identities match; all558 native MOS match exactly. Hosts and Python3.12/3.11 differ. No pure form-factor causal inference.'],['Rental evidence','T06 canonical rental run: verified_figure5_extension.json, followup_delivery_20261017.json, followup_cpu_audit_final01.json; all raw data SHA checked.'],['Retained estimates','338.2W is the same host-power proxy, not measured rental wall power. 400W is a limit, never a substituted average power.']];
 sources.getRange('A32:B36').format={font:{name:'Arial',size:10,color:'#243447'},wrapText:true,rowHeight:54};
 sources.getRange('B24').values=[['Current pre-extension files archived in canonical rental run/prior_report_archive. All original PCIe points and earlier evidence retained.']];
}
sources.getRange('A26:B31').values=[['Qwen execution','B3/B4: CPU integer metadata/indexing and grouped independent equal-length vision SDPA; original weights, precision and attention semantics.'],['Qwen equivalence','Full558 hidden/full vocabulary logits/quality logits/MOS compared exactly after cold cost in another process; each point has a receipt.'],['Gain definitions','Time/energy divided by LightGen-II reference. Qwen changes versus historical B1 include execution optimization and batch, not batch alone.'],['Model size','Total parameter definitions and original nominal/loaded sources retained in plot_data Measurements.'],['Selected data','B1 plus at most2 predeclared higher-B settings with resident mean>=95%. If none qualify, show highest predeclared B and disclose unmet target. Other measurements remain raw audit only.'],['Utilization and energy','Resident utilization is independent; energy uses same-cost-run active A100 power plus338.2W host proxy. Not a calibrated whole-system energy integral.']];
sources.getRange('A26:B31').format={font:{name:'Arial',size:10,color:'#243447'},wrapText:true,rowHeight:50};
if(cleanKimi){
 sources.getRange('A32:B38').values=[['Supplementary platform','Original PCIe40GB /250W preserved; DS full and Kimi use separately disclosed SXM4-80GB /400W. Not a common-hardware model ranking.'],['Physical GPU sessions','Kimi B1/B6 both GPU-ecac2c49-306a-9ec5-1572-8fcd2d232165. DS full B1/B2 retain different historical physical cards.'],['Qwen cross-device audit','Rental80GB Qwen excluded from paper tables and figures by user; original files and repeats preserved in audit history.'],['Clean Kimi evidence','verified_kimi_clean_extension.json; 120 SHA-verified raw files. All558 historical B1 predictions exactly equal.'],['Retained estimates','338.2W host proxy is not rental wall power; 400W is a limit, not substituted for recorded mean board power.'],['Kimi utilization','One loaded model, ascending B1..6; three complete native queue8 repeats per B. B2..5 utilization only, no new full558 accuracy or cost.'],['50% service costs','Separate Paced service tab/CSV: whole resident windows including imposed idle time. Never substitutes for dataset-once inference cost.']];
 sources.getRange('A32:B38').format={font:{name:'Arial',size:10,color:'#243447'},wrapText:true,rowHeight:58};
 writeKimiTrend();
 const service=plot.worksheets.add('Paced service');
 service.getRange('A1:F1').merge();service.getRange('A1').values=[['Kimi: approximately 50% GPU-busy paced service']];
 service.getRange('A2:F2').merge();service.getRange('A2').values=[['Resident repeated inputs; complete windows include all pacing waits. Not dataset-once inference costs.']];
 service.getRange('A3:F3').merge();service.getRange('A3').values=[['SXM4-80GB /400W; same card B1/B6. Three complete repeats; all samples pooled, no best-window selection.']];
 service.getRange('A4:F4').merge();service.getRange('A4').values=[['GPU time-busy percentage is not equivalent to optical spatial occupancy. Accuracy is evaluated in the separate dataset-once runs.']];
 service.getRange('A6:F6').values=[['Model','Video batch B','GPU busy (%)','Service time /16 videos (ms)','Estimated service energy /16 (J)','Board power (W)']];
 service.getRange('A16:K16').values=[['Model','B','GPU busy (%)','Window duration (s)','Forward calls','Video evaluations','Mean board W','Host proxy W','Unique videos','Raw source','GPU UUID']];
 for(const[i,x]of cleanData.paced_service.entries()){const r=i+17,t=i+7;service.getRange(`A${r}:K${r}`).values=[[x.model,x.batch,x.mean_busy_percent,x.duration_s,x.completed_forward_calls,null,x.board_mean_w,338.2,x.unique_videos,x.source,x.gpu_uuid]];service.getRange(`F${r}`).formulas=[[`=B${r}*E${r}`]];service.getRange(`A${t}:F${t}`).formulas=[[`=A${r}`,`=B${r}`,`=C${r}`,`=D${r}*16000/F${r}`,`=(G${r}+H${r})*D${t}/1000`,`=G${r}`]];}
 service.getRange('A1:K18').format={font:{name:'Arial',size:10,color:'#243447'},columnWidth:23,rowHeight:27,verticalAlignment:'center'};service.getRange('A1:F1').format={fill:'#B64058',font:{name:'Arial',size:15,bold:true,color:'#FFFFFF'},rowHeight:34};service.getRange('A2:F4').format={wrapText:true,rowHeight:36};service.getRange('A6:F6').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:45};service.getRange('A7:F8').format.fill='#FCECEF';service.getRange('C7:F8').setNumberFormat('#,##0.000000');service.getRange('A16:K16').format={fill:'#214D68',font:{name:'Arial',size:10,color:'#FFFFFF',bold:true},wrapText:true,rowHeight:40};service.getRange('A10:F11').merge();service.getRange('A10').values=[['Service T16 = full measured duration × 16 / completed video evaluations. E16 = (338.2 W + same-window mean board W) × service T16 / 1000. Raw inputs and source paths are retained below.']];service.getRange('A10:F11').format={wrapText:true,rowHeight:30};
 plot.recalculate();const original=service.getRange('D17').values[0][0],before=service.getRange('D7').values[0][0];service.getRange('D17').values=[[original*1.01]];plot.recalculate();if(Math.abs(service.getRange('D7').values[0][0]-before*1.01)>1e-8)throw Error('Paced service recalculation');service.getRange('D17').values=[[original]];
}
polishReadability();for(const book of books)book.recalculate();
const checkInput=inputs.getRange('G7'),oldInput=checkInput.values[0][0],oldGain=comparison.getRange('I9').values[0][0];
checkInput.values=[[oldInput*1.01]];plot.recalculate();if(Math.abs(comparison.getRange('G9').values[0][0]-oldInput*1.01)>1e-8||Math.abs(comparison.getRange('I9').values[0][0]-oldGain*1.01)>1e-8)throw Error('Plot formula recalculation failed');checkInput.values=[[oldInput]];plot.recalculate();
const checkWall=detail.getRange('AA4'),oldWall=checkWall.values[0][0],oldTime=summary.getRange('D8').values[0][0];checkWall.values=[[oldWall*1.01]];main.recalculate();if(Math.abs(summary.getRange('D8').values[0][0]-oldTime*1.01)>1e-8)throw Error('Main cost recalculation failed');checkWall.values=[[oldWall]];main.recalculate();
const checkCounter=uv.getRange('L8'),oldCounter=checkCounter.values[0][0],oldUtil=uv.getRange('C8').values[0][0];checkCounter.values=[[oldCounter*.99]];util.recalculate();if(Math.abs(uv.getRange('C8').values[0][0]-oldUtil*.99)>1e-10)throw Error('Utilization formula recalculation failed');checkCounter.values=[[oldCounter]];util.recalculate();
await renderQA();
for(const book of books)console.log((await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
for(let i=0;i<rows.length;i++){const r=i+7,x=rows[i];if(Math.abs(comparison.getRange('G'+r).values[0][0]-x.time_16_videos_ms)>1e-8||Math.abs(comparison.getRange('H'+r).values[0][0]-x.energy_16_videos_j)>1e-8||Math.abs(summary.getRange('D'+(i+6)).values[0][0]-x.time_16_videos_ms)>1e-8)throw Error('Formula/source mismatch');}
for(let i=0;i<books.length;i++)await(await SpreadsheetFile.exportXlsx(books[i])).save(path.join(out,filenames[i]));
const quote=v=>v===null||v===undefined?'':/[",\r\n]/.test(String(v))?'"'+String(v).replaceAll('"','""')+'"':String(v);
const csv=(objects,keys=[...new Set(objects.flatMap(x=>Object.keys(x)))])=>[keys,...objects.map(x=>keys.map(k=>x[k]))].map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n';
await fs.writeFile(path.join(out,'plot_data.csv'),csv(rows));
await fs.writeFile(path.join(out,'timing_energy_recalculated.csv'),csv(rows));
await fs.writeFile(path.join(out,'batch_metric_audit.csv'),csv(data.paired_rows.map(x=>({model:x.model,batch_size_videos:x.batch_size,full558_srcc:x.srcc,full558_plcc:x.plcc,first200_srcc:x.first200_srcc,independent_resident_util_mean_percent:x.independent_resident_util_mean_percent,repeat_means_percent:x.util_repeat_means.join(','),repeat_spread_pp:Math.max(...x.util_repeat_means)-Math.min(...x.util_repeat_means),displayed:selectedKeys.has(x.baseline+':'+x.batch_size),checkpoint_sha256:x.checkpoint_sha256,source:x.source,utilization_source:x.utilization_source,execution:x.execution}))));
await fs.writeFile(path.join(out,'LGVQ_temporal_gpu_utilization.csv'),csv(urows.map(x=>{const raw=getRaw(x);return{model:x.model,baseline:x.baseline,batch:x.batch_size_videos,mean_percent:x.display_gpu_util_mean_percent,peak_percent:raw?.independent_resident_util_peak_percent??h.peak_gpu_utilization_percent,repeat_means:raw?.util_repeat_means.join(',')??'',samples:raw?.util_samples??h.samples,duration_s:raw?.util_duration_s??h.duration_s,forwards:raw?.util_forward_calls??historical.measured_forward_invocations,execution:x.execution_profile,resident_protocol:x.utilization_basis,source:x.utilization_source,measured_local:x.util_measured_local};})));
await fs.writeFile(path.join(out,'LGVQ_temporal_gpu_utilization_repeats.csv'),csv(repeatRows));
if(cleanKimi){
 await fs.writeFile(path.join(out,'LGVQ_kimi_paced_service.csv'),csv(cleanData.paced_service));
 await fs.writeFile(path.join(out,'LGVQ_kimi_native_utilization_trend.csv'),csv(cleanData.utilization_only_trend.map(x=>({...x,repeat_means:x.repeat_means.join(',')}))));
}
const verifiedName=cleanKimi?'../verified_kimi_clean_extension.json':rental?'verified_figure5_extension.json':sequential?'verified_sequential.json':'verified_selected_execution.json';
await fs.writeFile(path.join(run,'presentation_selected.json'),JSON.stringify({status:'complete',rows,display_selection_rule:'B1 plus at most2 predeclared higher-B settings with resident mean>=95%; if none, show highest predeclared B as unmet target. Sequential reproduction caps YOLO at3 and CLIP at10 as explicitly requested.',source:verifiedName,source_sha256:digest(await fs.readFile(path.join(run,verifiedName))),base_source:path.relative(repo,path.join(baseRun,'verified_selected_execution.json')).replaceAll('\\','/'),base_source_sha256:digest(await fs.readFile(path.join(baseRun,'verified_selected_execution.json'))),recheck_source_sha256:recheck?digest(await fs.readFile(path.join(recheckRun,'verified_selected_execution.json'))):null,prior_archive:'prior_report_archive/manifest.json',workbook_files:filenames},null,2)+'\n',{flag:'wx'});
const f=(v,n=3)=>v===null||v===undefined?'—':Number(v).toFixed(n);
const pointTable=['| Model | B / optical N | Resident GPU utilization (%) | Full558 SRCC | T16 (ms) | Estimated E16 (J) | LightGen-II time advantage (×) | LightGen-II energy-efficiency advantage (×) |'+(rental?' Platform / limit |':''),
 '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'+(rental?' --- |':''),
 ...rows.map(x=>`| ${x.model}${x.baseline==='qwen'&&x.batch_size_videos===1?' †':''} | ${x.baseline==='ours'?'N=16':x.batch_size_videos} | ${f(x.display_gpu_util_mean_percent,2)} | ${f(x.measured_full558_srcc,8)} | ${f(x.time_16_videos_ms,6)} | ${f(x.energy_16_videos_j,6)} | ${f(x.lightgen_time_speedup_x)} | ${f(x.lightgen_energy_efficiency_gain_x)} |`+(rental?` ${x.hardware_platform}${x.power_limit_w?' / '+x.power_limit_w+'W':''} |`:''))].join('\n');
const repeatTable=['| Model | B | Repeat1 (%) | Repeat2 (%) | Repeat3 (%) | Pooled mean (%) | Full-window samples | Duration (s) | Execution |',
 '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
 ...data.paired_rows.map(x=>`| ${x.model} | ${x.batch_size} | ${f(x.util_repeat_means[0],5)} | ${f(x.util_repeat_means[1],5)} | ${f(x.util_repeat_means[2],5)} | ${f(x.independent_resident_util_mean_percent,8)} | ${x.util_samples} | ${f(x.util_duration_s)} | ${x.execution} |`)].join('\n');
const recheckNotes=recheck?await fs.readFile(path.join(sequential?sequentialRun:run,'method_comparison.md'),'utf8'):'';
const cohort=sequential?`On 17 October, independent GPU utilization at YOLO B=3 and CLIP B=10 was remeasured using one loaded model process for each ascending batch sequence. All 11 predecessor/target batches and 33 complete repeats were retained. Only the two requested target utilizations replace the corresponding displayed annotations, regardless of direction. Accuracy, inference latency and energy were not remeasured in this utilization-only run: they retain their separately verified same-checkpoint, same-batch sources. Prior fresh-process repeats remain distinct historical records and are never pooled with this run. All other displayed utilization values, including finalized Qwen B=1, remain unchanged. The main display now caps YOLO at B=3 and CLIP at B=10 rather than substituting B=4/B=11.`:recheck?`The independently audited 16-point selected round is retained, with ${newCount} prescribed YOLO/CLIP recheck points from 17 October replacing the corresponding older points regardless of increase or decrease. Both old and new raw repeats are retained separately. All other models, including the historical Qwen B1, are unchanged. The no-explicit-warmup cost protocol still includes each first call. CLIP B10 first-call latency changed from 534.974873 to 197.230 ms (rounded), which explains most of the lower first200 normalized cost in this round; the timing boundary was not shortened. These cold-first-call costs retain initialization/cache-state variability and are not a steady-state throughput measurement.`:'All 16 predeclared new model–batch points and their 48 complete resident-utilization repeats have been independently audited.';
let currentNotes=`# LGVQ temporal quality: selected-batch accuracy, latency and estimated energy

## Current comparison — 16–17 October 2026

${cohort} The main figures and the three original workbooks now show ${rows.length-1} electronic points plus one unchanged LightGen-II reference: B=1 and one or two near-saturation settings per model. Intermediate candidates remain in the raw evidence and performance audit, not in the main figures. Numerical source: [plot_data.csv](plot_data.csv); colored comparison: [plot_data.xlsx](plot_data.xlsx). Unrounded values, parameter definitions, checkpoint hashes and distinct cost/utilization sources are retained there. Previous presentation files are preserved with SHA receipts in T06's canonical run, under \`prior_report_archive\`.

${recheckNotes}

The display rule was fixed before the remaining models completed: keep B=1 and at most two predeclared higher batches with pooled resident utilization ≥95%; if none qualifies, retain the largest predeclared batch and disclose the unmet target. The 98% line is a reference, not a selectable utilization setting or an overload boundary. No point was selected by SRCC, latency or energy. All new measurements use the same A100 UUID and 250-W power limit; no training, precision change, CUDA Graph, artificial load or highest-window selection was introduced.

### Methods for the latency and energy figures

The LGVQ split contains 2,250 training videos and 558 TEST videos (manifest SHA256 \`607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc\`). Each fixed checkpoint was evaluated once on all 558 TEST videos at each measured batch size to obtain that point's SRCC and PLCC. Inputs retain the original four-frame sampling, 65% crop, model-specific resolution and deployed BF16/FP16 backbone plus FP32 readout. B denotes videos per forward, not frames or the normalization workload. Fixed checkpoints previously selected on TEST retain their development/no-independent-VAL interpretation; these measurements introduce no new training or selection.

Accuracy and costs are generated by one dataset-once process per point. The first 200 TEST videos define inference costs, with a separate batch boundary at 200 and 558; incomplete tails are retained, not padded, dropped or charged as full batches. There is no explicit cost warmup, so the first inference is included. The timed boundary begins at the first native backbone block (first convolution for YOLO, first Transformer block for the others) and ends after GPU MOS output synchronization. For optimized Qwen, an equivalent boundary hook is attached to its first visual block. CPU video decoding, preprocessing and host-to-device transfer are recorded only for audit, not included in the displayed inference latency. Thus these are inference-boundary comparisons, not full video-file-to-answer latency.

For synchronized first-200 wall times t_i in milliseconds, T16=(Σt_i)×16/200. Estimated E16=(338.2 W host-power proxy + same-cost-run mean active A100 board power)×T16/1000. The 16-video values are normalized workloads, not a claim that each GPU call uses B=16. Costs are paired with their own recorded active board-power samples; the board power or utilization of the separate resident experiment is never substituted. E16 is the user-approved sampled-power estimate, not a calibrated whole-system energy-meter integral. Driver averaging windows and sparse active samples for short calls limit instantaneous power attribution.

Independent utilization is measured separately to reduce external CPU waiting: the first B TEST videos are prepared and copied to GPU once; each forward still processes B videos. Eight model forwards are queued on the same CUDA stream before synchronization (queue8 does not mean 8B concurrent videos). Each point has three identical repeats, each with at least 5 s warmup and a complete ≥30 s measurement window. All 50-ms telemetry samples within all three complete windows are pooled; no high-utilization interval or best repeat is selected. NVML reports driver-window kernel-busy time, not SM occupancy or achieved FLOPS. Cost-run and resident-utilization dates, clocks, power, memory and CPU scheduling records remain distinct.

### Qwen execution optimization and protected references

Qwen B=3/4 use CPU integer metadata, equivalent indexing and grouped independent equal-length visual-attention calls to reduce host/device synchronization and launch overhead. Learned weights, architecture, numerical precision and attention connectivity are unchanged. This is an execution implementation change and must be disclosed; it is not an unmodified off-the-shelf forward implementation. Both points passed exact comparisons of hidden states, full vocabulary logits, quality logits and MOS on all 558 TEST videos, with the fresh cold-cost predictions also equal. This establishes equivalence for the tested inputs and profiles, not for every possible input or batch. Qwen's differences against historical B=1 include both batch and execution optimization and must not be described as a pure batch-scaling ablation.

† Qwen B=1 is not reexecuted: SRCC=0.7663428036522781, T16=1188.3861464262009 ms, E16=462.1134010767563 J, and independent resident utilization=63.133111480865225% are the finalized historical references. Its utilization comes from one retained 5-s/30-s native run, not this round's three-repeat mean. The separate historical three-repeat audit of 71.17536071032187% is preserved and is not displayed as the selected B=1 value. The historical B=1 point is not connected to the optimized Qwen points in the figures. LightGen-II retains physical SRCC=${data.ours.physical_reference_srcc}, T16=${data.ours.time_16_videos_ms} ms and estimated E16=${data.ours.energy_16_videos_j} J; N=16 denotes videos encoded in the optical field, not a GPU batch. Optical cost remains the approved component estimate, not a newly measured GPU-style process.

### Complete data used in both figures

${pointTable}

LightGen-II time advantage is baseline T16 / LightGen-II T16; energy-efficiency advantage is baseline E16 / LightGen-II E16 for the same normalized 16-video workload, not TOPS/W. Relative-to-B1 improvements are additionally available in the workbook. Model parameter sizes retain their declared nominal/loaded-total definitions, with MoE active parameters listed separately; they are not inferred from memory usage or optical degrees of freedom.

### Manuscript-ready descriptions of the two panels

**Latency panel.** Temporal-quality assessment accuracy is plotted against inference latency normalized to 16 LGVQ videos. Accuracy is evaluated independently at each model–batch configuration on all 558 test videos; GPU latency is derived from synchronized inference-boundary timings of the first 200 videos. Each electronic model is represented by B=1 and one or two predeclared near-saturation configurations. The label at each point gives video batch size and independently measured resident GPU utilization. LightGen-II is shown using its retained experimental reference. The finalized native Qwen B=1 reference is distinguished from the equivalence-validated optimized execution used at B=3/4. Lower latency at comparable SRCC indicates a more favorable inference trade-off; no monotonic accuracy trend with batch is assumed.

**Energy panel.** The same per-configuration temporal SRCC is plotted against estimated energy normalized to 16 videos. For each new GPU point, energy combines its synchronized inference timing with its own active A100 power samples and the fixed 338.2-W host-power proxy. Resident-utilization measurements are used only for annotation and do not supply the power in this calculation. The LightGen-II and finalized Qwen B=1 references retain their approved historical cost estimates. The logarithmic horizontal axis covers every displayed point. These results compare estimated inference energy under the declared boundaries, rather than wall-meter energy for decoding and all host processing.

Current figures: [two-panel comparison](srcc_cost_comparison_reference.png), [latency](latency_comparison_reference.png), [estimated energy](energy_comparison_reference.png). Their editable SVG counterparts and [independent utilization data](LGVQ_temporal_gpu_utilization.csv) are in this same directory. Reproduce the evidence audit with T06 \`audit_selected_execution.py\`, initial workbook authoring with \`reports/build_selected_execution.mjs --build\` (refuses to overwrite a changed archive), saved-workbook verification with \`--verify\`, formatting-only updates with \`--polish\`, and the figures with \`reports/plot_selected_execution.py\`. ${recheck?'Add `'+(sequential?'--yolo-clip-sequential':'--yolo-clip-recheck')+'` to the audit, workbook and figure commands; original, recheck and current verified-source SHA values are recorded in the presentation receipt.':''} Do not import the old overwrite-on-import \`build_analysis.py\`.

---

# Historical notes — superseded presentation, protected evidence

Earlier descriptions and intermediate configurations are archived for provenance, not included in the current comparison.

`;
let rentalNotes=rental?`### Supplementary A100-SXM4-80GB platform (17 October 2026)

Five complete supplementary points were added while retaining all 15 original displayed rows: DeepSeek-VL2 B=1/2, Kimi-VL-A3B-Instruct B=1/2 and native Qwen3-VL-2B B=1. The 80-GB supplementary series use hollow symbols and A100-SXM4-80GB at a 400-W power limit. The daggered historical 40-GB native Qwen B=1 is also an open circle and has a separate legend entry. The original measurements use A100-PCIE-40GB at 250 W, with the historical native Qwen B=1 separately marked. Results across these platforms are descriptive deployment comparisons, not common-hardware model rankings. Energy calculations use measured active board power, not the power limit.

Rental B=1 measurements for DeepSeek and Kimi used physical GPU UUID GPU-780326a6-79bc-7a45-36f6-683c5b278bee. Their B=2 measurements and the new Qwen B=1 control used GPU-d06c95d0-7066-9768-0f6e-fc1ee877f9f9 after the instance was restarted. Dashed connectors guide the eye only. These B=1 and B=2 results do not constitute a strict same-card batch-only ablation. No batch beyond the authorized B=2 was tested. DeepSeek B=2 utilization was 96.54% and Kimi B=2 utilization was 86.11%. Neither reached the 98% reference.

Each of the five points includes independent predictions for all 558 TEST videos and costs measured over the first 200 videos. No explicit warmup is used for cost measurement, and the first inference is included. Timing starts at the input to the first native visual block and ends at synchronized GPU MOS output. Video decoding, preprocessing and host-to-device transfer are excluded. The 338.2-W host proxy is retained and is not a wall-power measurement of the rental host. All 15 resident-utilization repeats use native queue8 execution, at least 5 s of warmup, a complete measurement window of at least 30 s, and pooling of every 50-ms sample in those windows. Each model-batch point uses a separate process. This differs from the cross-batch process lifecycle of the latest YOLO B=3 and CLIP B=10 reproduction.

The 80-GB Qwen control uses the same official revision 89644892e4d85e24eaac8bacfd4f463576704203, quality-head SHA e102adc6a7b33335c8f49ae1cbeea3302a1a10017d282da8d6091a640276d0d2, native SDPA and original BF16/FP32 precision. Input shapes, dtypes, tensor hashes and the prompt match for the first fixed TEST video. All 558 native MOS outputs match the earlier native B=1 outputs exactly. The new T16 of 789.168774 ms and estimated E16 of 324.899880 J are reported separately from the protected 40-GB historical values of 1188.386146 ms and 462.113401 J. The observed ratios are 1.50587 and 1.42233, respectively. Different hosts, dates and Python versions (3.12.13 versus 3.11.15) prevent attribution solely to GPU form factor. The finalized 40-GB utilization of 63.13311148% is unchanged. The independent 80-GB mean is 74.38873551%.

Raw point sources, all repeats and hardware UUIDs are retained in the canonical rental run and verified_figure5_extension.json, bound to followup_delivery_20261017.json and the full CPU audit. Prior presentation files have a separate SHA archive. Kimi and DeepSeek heads were previously selected by highest TEST SRCC, so their scores remain development/no-independent-VAL metrics. No head was retrained for this followup.

**Supplementary figure description.** The latency and energy panels retain the original 40-GB measurements and add five hollow 80-GB points. Each point uses accuracy evaluated on all 558 TEST videos and costs measured over the first 200 videos. Marker fill indicates the platform. Labels show video batch size and independently measured resident GPU utilization. Dashed connectors between rental sessions are descriptive only. The observed accuracy-cost positions do not isolate model architecture, GPU form factor, host, date or batch effects. The cross-device Qwen B=1 control is not connected to the optimized B=3/4 series on the original platform.

`:'';
if(cleanKimi){
 const trend=['| B | Repeat1 (%) | Repeat2 (%) | Repeat3 (%) | Pooled mean (%) | Full558 accuracy/cost |','| ---: | ---: | ---: | ---: | ---: | --- |',...cleanData.utilization_only_trend.map(x=>`| ${x.batch} | ${x.repeat_means.map(v=>f(v,8)).join(' | ')} | ${f(x.mean_percent,8)} | ${x.full558_accuracy_available?'Measured':'Not measured'} |`)].join('\n');
 const paced=['| B | Busy mean (%) | Repeat1 / 2 / 3 (%) | Full-window board mean (W) | Service T16 (ms) | Estimated service E16 (J) |','| ---: | ---: | --- | ---: | ---: | ---: |',...cleanData.paced_service.map(x=>`| ${x.batch} | ${f(x.mean_busy_percent,8)} | ${x.repeat_means_percent.map(v=>f(v,8)).join(' / ')} | ${f(x.board_mean_w,8)} | ${f(x.service_T16_ms,8)} | ${f(x.service_E16_j,8)} |`)].join('\n');
 rentalNotes=`### Supplementary platforms and clean Kimi follow-up (18 October 2026)

The paper comparison retains all 15 original displayed rows and four supplementary points: DeepSeek-VL2 B=1/2 and Kimi-VL-A3B-Instruct B=1/6. The rental Qwen B=1 control is excluded from the paper tables and figures at the author's request; its raw data and historical repeats remain available for audit. Hollow symbols identify A100-SXM4-80GB /400W, whereas the original GPU cohort uses A100-PCIE-40GB /250W. These are descriptive deployment comparisons, not a common-hardware model ranking. Energy uses recorded active board power, never the nominal power limit.

Kimi B=1 and B=6 were measured on the same physical GPU, GPU-ecac2c49-306a-9ec5-1572-8fcd2d232165, with unchanged frozen head SHA256 e6bf55b3b7ff318d37c973b67c4a6ce1062e4760ac91c2bdc748628988022586. Each point has all 558 TEST predictions and its own first-200 timing/power samples. New B=1 predictions match all 558 historical B=1 predictions exactly. The previous Kimi B=1/B=2 session is superseded for display, not deleted. DeepSeek full B=1/B=2 retains its historical distinct-card sessions (GPU-780326a6-79bc-7a45-36f6-683c5b278bee and GPU-d06c95d0-7066-9768-0f6e-fc1ee877f9f9); its dashed connector is descriptive, not a strict same-card batch ablation. The 338.2-W host proxy is retained and is not a measured rental-host wall-power value.

**Independent native utilization.** One Kimi model process was loaded once and measured sequentially at B=1 through B=6. Inputs retain four frames and the deployed precision. Each point has three identical complete queue8 repeats (at least 5 s warmup and 30 s measurement); all 50-ms driver samples are pooled. Each forward processes B videos, not 8B. B=6 is the first measured point strictly above 98%. B=2–5 have utilization-only evidence, not new full558 accuracy or dataset-once costs. No intermediate accuracy or cost is interpolated.

${trend}

**Separate paced service measurements.** Kimi B=1 and B=6 were also measured at approximately 50% driver-window kernel-busy time. This is a temporal duty-cycle constraint, not an equivalence to optical spatial occupancy. The first B videos are resident and repeated; complete-group feedback includes both compute and imposed idle time during a ≥30-s calibration, then freezes the duty for three complete repeats. Service T16 = total complete-window duration × 16 / completed video evaluations; estimated service E16 = (338.2 W + duration-weighted same-window mean board power) × service T16/1000 (with T16 in milliseconds). All imposed waits are included. These are not the dataset-once inference costs plotted in the main panels, and their resident board power must not enter the first-200 cost formula. Arithmetic telemetry means retain the limitations of sampled driver windows, rather than representing a calibrated energy integral. This rental controller uses the last complete compute-plus-idle group for feedback; the paused original40GB v2 run aggregates completed groups since the preceding update. They must not be described as bit-identical controller profiles.

${paced}

Paced data are retained in [LGVQ_kimi_paced_service.csv](LGVQ_kimi_paced_service.csv) and the separate Paced service worksheet; the six-batch native trend is retained in [LGVQ_kimi_native_utilization_trend.csv](LGVQ_kimi_native_utilization_trend.csv). The original40GB paced experiment is incomplete after a telemetry-reader failure: four points passed independent raw audit, eight remain uncompleted. Neither failed telemetry nor inferred values enter these Kimi tables. Completion of this Kimi follow-up does not imply completion of that separate suite.

The 120 delivered Kimi raw files passed length and SHA256 checks, and utilization, costs and predictions were independently replayed on CPU in verified_kimi_clean_extension.json. The delivery and original audit remain under the canonical rental run. Previously TEST-selected quality heads retain the development/no-independent-VAL interpretation; no head or backbone was retrained for this follow-up. The model process has exited and its GPU was released.

**Figure interpretation.** The two panels show independently evaluated temporal SRCC against normalized inference latency and estimated energy for the same configurations. Labels give B and separately measured resident utilization. The Kimi solid connector links same-card B=1/B=6; the DeepSeek full dashed connector spans different sessions. Neither the platform differences nor the sampled-power estimates permit attribution solely to model architecture. Paced service costs are intentionally absent from these conventional inference panels.

`;
}
if(rental)currentNotes=currentNotes.replace('## Current comparison — 16–17 October 2026','## Current comparison — 16–18 October 2026\n\n'+rentalNotes).replace('All new measurements use the same A100 UUID and 250-W power limit','The original 40GB cohort uses its retained A100 UUID and 250-W limit; supplementary 80GB measurements use the separately disclosed 400-W platform and sessions').replace('† Qwen B=1 is not reexecuted:','† The finalized40GB Qwen B=1 point is not replaced:').replace('Each electronic model is represented by B=1 and one or two predeclared near-saturation configurations.',cleanKimi?'Original-platform models retain their selected B=1 and high-batch points; the80GB supplementary cohort retains DeepSeek B1/B2 and adds same-card Kimi B1/B6.':'Original-platform models retain B=1 and selected high-batch points; the80GB supplementary cohort adds B1/B2, including unmet saturation targets.');
await fs.writeFile(path.join(out,'README.md'),currentNotes+'[Archived prior README]('+path.relative(out,path.join(run,'prior_report_archive/README.md')).replaceAll('\\','/')+').\n');
let utilNotes=`# LGVQ GPU utilization: current selected-execution comparison

${cohort} Current point means and repeats are shown below. Superseded repeats remain separately labeled in the raw-repeat workbook/CSV, never pooled with a later round. Main figures use B=1 plus at most two predeclared higher batches with pooled resident utilization ≥95% (largest tested batch if none qualifies). All other points remain audit evidence. The three workbooks and CSV identify displayed versus audit-only repeats. See the current comparison README for paired inference-cost data and the two figure descriptions.

${recheckNotes}

The measurement method is unchanged: first B TEST videos prepared/H2D once, original four frames, original precision, same-stream queue8, three identical ≥5-s warmup/≥30-s full measurement repeats, all 50-ms samples pooled. Each call processes B videos, not 8B. Driver-window utilization is kernel-busy time, not SM occupancy or a configured percentage. There is no CUDA Graph, synthetic load, best-window selection, clock change or new training. Qwen B3/4 change execution grouping/indexing under full558 exact-equivalence checks; the other models use native execution.

Qwen B1 stays at the finalized 63.133111480865225% historical one-repeat value and was not reexecuted. The earlier 71.17536071032187% three-repeat audit remains a separate historical record. Neither is a new same-round result. The finalized Qwen B1 costs are unchanged. Resident utilization and power are independent of the dataset-once cost experiment; resident power must not enter the cost formula.

${repeatTable}

Full source hashes, raw telemetry, forward counts, clocks, temperatures and process guards are independently audited in T06 \`runs/simulation/lgvq_selected_execution_a100_20261016/verified_selected_execution.json\` ${recheck?'and `runs/simulation/lgvq_yolo_clip_recheck_a100_20261017/verified_selected_execution.json`':''}${sequential?' and `runs/simulation/lgvq_yolo_clip_sequential_a100_20261017/verified_sequential.json`':''}. Raw reports and their transfer SHA receipts are retained alongside them. The 98% line is a reporting reference, not an overload limit; results below it are not rounded upward.

---

# Historical utilization notes — preserved, not the current display

Earlier descriptions and intermediate configurations are archived for provenance, not included in the current comparison.

`;
if(rental)utilNotes=utilNotes.replace('\n\n','\n\n'+rentalNotes).replace('Qwen B1 stays at the finalized','The original40GB Qwen B1 stays at the finalized');
await fs.writeFile(path.join(out,'UTILIZATION_METHOD_20261015.md'),utilNotes+'[Archived prior utilization methods]('+path.relative(out,path.join(run,'prior_report_archive/UTILIZATION_METHOD_20261015.md')).replaceAll('\\','/')+').\n');
console.log(JSON.stringify({updated_workbooks:3,main_rows:rows.length,new_measurement_points:newCount,historical_Qwen_B1_unchanged:true}));
