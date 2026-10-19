import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const require=createRequire(path.join(process.cwd(),'.codex_tmp/artifact-runtime.cjs'));
const {Workbook,SpreadsheetFile,FileBlob}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const repo=process.cwd(),root=path.join(repo,'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_busy_resource_20261020/presentation');
const data=JSON.parse(await fs.readFile(path.join(root,'eight_tables.json'),'utf8'));
const out=path.join(repo,'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013/LGVQ_busy_resource_precision_comparison.xlsx');
const book=Workbook.create(),views=[],models=['deepseek','kimi','qwen','glm','clip','yolo'];
const colors={deepseek:'#79502C',kimi:'#D14F64',qwen:'#0072B2',glm:'#9B567C',clip:'#008B70',yolo:'#B17D00'};
for(const method of ['temporal','mps'])for(const precision of ['native','fp32'])for(const metric of ['time','energy']){
 const name=`${method==='temporal'?'Duty':'MPS'}_${precision==='native'?'Native':'FP32'}_${metric==='time'?'Time':'Energy'}`;
 views.push({method,precision,metric,sheet:book.worksheets.add(name)});
}
const raw=book.worksheets.add('Measurements');
raw.getRange('A1:AJ1').merge();raw.getRange('A1').values=[['RAW INPUTS AND SAME-WINDOW FORMULAS — missing measurements are not estimates']];
raw.getRange('A1:AJ1').format={font:{name:'Arial',size:13,bold:true,color:'#214D68'},rowHeight:30};
const headers=['Model','Method','Precision','Condition','Video batch','MPS cap (%)','Visible SMs','Independent558 SRCC reference',
 'Repeat1 seconds','Repeat1 video evaluations','Repeat1 boardW','Repeat1 busy%','Repeat1 samples',
 'Repeat2 seconds','Repeat2 video evaluations','Repeat2 boardW','Repeat2 busy%','Repeat2 samples',
 'Repeat3 seconds','Repeat3 video evaluations','Repeat3 boardW','Repeat3 busy%','Repeat3 samples',
 'Host proxy W','Total seconds','Total video evaluations','Estimated window energy J','Pooled busy%','Equivalent16-video ms','Equivalent16-video J','Repeats','Status','Platform','Report source','Report SHA256','Accuracy/protocol note'];
raw.getRange('A6:AJ6').values=[headers];raw.getRange('A6:AJ6').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:48};
raw.getRange('A7:AJ55').format={font:{name:'Arial',size:10,color:'#243447'},rowHeight:26};
raw.getRange('A:AJ').format.columnWidth=18;raw.getRange('A:A').format.columnWidth=26;raw.getRange('AH:AJ').format.columnWidth=65;
raw.getRange('H7:AD55').setNumberFormat('0.000000');raw.freezePanes.freezeRows(6);raw.freezePanes.freezeColumns(4);
raw.getRange('A7:H7').values=[['LightGen-II','optical reference','not applicable','~50% spatial use',null,null,null,data.optical.srcc]];
raw.getRange('AC7:AD7').values=[[data.optical.time_ms,data.optical.energy_j]];raw.getRange('AF7:AJ7').values=[['unchanged optical reference','Optical platform',data.optical.cost_source,'','Spatial usage is not GPU busy time or a MPS resource cap. No FP32 designation.']];
let row=8;const entries=[];
for(const method of ['temporal','mps'])for(const precision of ['native','fp32'])for(const model of models)for(const condition of method==='temporal'?['busy50','unrestricted']:['mps_cap50','mps_cap100']){
 const point=data.values.find(x=>x.method===method&&x.precision===precision&&x.model===model&&x.condition===condition);
 const missing=model==='deepseek'&&precision==='fp32'?'capacity exceeds80GB':'pending measurement';
 const values=[data.models[model],method,precision,condition,point?.batch??(method==='mps'?1:null),point?.resource_cap??null,point?.visible_sms??null,point?.srcc_reference??null];
 for(let i=0;i<3;i++){const r=point?.repeats?.[i];values.push(r?.elapsed_s??null,r?.video_evaluations??null,r?.measurement.mean_board_power_w??null,r?.measurement.mean_gpu_utilization_percent??null,r?.measurement.samples??null);}
 values.push(338.2,null,null,null,null,null,null,point?.repeats?.length??null,point?.status??missing,point?.platform??'',point?.source??'',point?.source_sha256??'',point?.accuracy_note??(missing==='capacity exceeds80GB'?data.capacity_missing:'No completed new-protocol point. Do not substitute historical pure-inference cost.'));
 raw.getRange(`A${row}:AJ${row}`).values=[values];
 if(point){
  raw.getRange(`Y${row}:AE${row}`).formulas=[[
   `=SUM(I${row},N${row},S${row})`,`=SUM(J${row},O${row},T${row})`,
   `=(X${row}+K${row})*I${row}+(X${row}+P${row})*N${row}+(X${row}+U${row})*S${row}`,
   `=(L${row}*M${row}+Q${row}*R${row}+V${row}*W${row})/SUM(M${row},R${row},W${row})`,
   `=Y${row}*16000/Z${row}`,`=AA${row}*16/Z${row}`,`=3`]];
 }
 entries.push({method,precision,model,condition,row,point});row++;
}
for(const {method,precision,metric,sheet} of views){
 sheet.getRange('A:M').format.columnWidth=16;sheet.getRange('A:A').format.columnWidth=25;sheet.getRange('B:B').format.columnWidth=25;sheet.getRange('J:J').format.columnWidth=25;
 sheet.getRange('A1:M1').merge();sheet.getRange('A1').values=[['LGVQ temporal quality · paired GPU cost and utilization']];
 sheet.getRange('A1:M1').format={font:{name:'Arial',size:16,bold:true,color:'#214D68'},rowHeight:34};
 sheet.getRange('A2:M2').merge();sheet.getRange('A2').values=[[`${method==='mps'?'MPS execution-resource provisioning':'Temporal duty control'} / ${precision==='fp32'?'FP32':'original deployed precision'} / ${metric==='time'?'processing time':'estimated energy'}`]];
 sheet.getRange('A2:M2').format={font:{name:'Arial',size:12,color:'#53687A'},rowHeight:28};
 sheet.getRange('A4:M4').merge();sheet.getRange('A4').values=[['First TF input → final GPU output; internal waits included, terminal wait excluded. Prepared inputs reused; three full windows.16-video equivalents are throughput-normalized, not a16-video batch.']];
 sheet.getRange('A4:M4').format={font:{name:'Arial',size:10,color:'#53687A'},rowHeight:36,wrapText:true};
 sheet.getRange('A5:M5').merge();sheet.getRange('A5').values=[['PARTIAL DATA — measured values only. Independent558-video accuracy references are not fresh restricted-condition accuracy tests.']];
 sheet.getRange('A5:M5').format={font:{name:'Arial',size:10,bold:true,color:'#9C3D10'},rowHeight:28,wrapText:true};
 sheet.getRange('A6:M6').values=[['Model','Setting','Video batch','Resource cap','Actual GPU busy',metric==='time'?'Time /16 (ms)':'Energy /16 (J)','Baseline / optical','SRCC reference','Repeats','Status','Video evaluations','Precision','Platform']];
 sheet.getRange('A6:M6').format={fill:'#214D68',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:38,wrapText:true};
 sheet.getRange('A7:M19').format={font:{name:'Arial',size:10,color:'#243447'},rowHeight:29,wrapText:true};
 sheet.getRange('A7:M7').values=[['LightGen-II','Optical reference','n.a.','n.a.','~50% spatial',null,null,null,'n.a.','unchanged reference','n.a.','n.a.','Optical platform']];
 sheet.getRange('F7:H7').formulas=[[`=Measurements!${metric==='time'?'AC':'AD'}7`,'=1','=Measurements!H7']];
 sheet.getRange('A7:M7').format.fill='#EEE6F4';sheet.getRange('A7').format.font={name:'Arial',size:10,bold:true,color:'#7A3E9D'};
 let r=8;
 for(const e of entries.filter(e=>e.method===method&&e.precision===precision)){
  const setting=method==='mps'?`${e.condition==='mps_cap50'?'50%':'100%'} resource cap`:e.condition==='busy50'?'~50% time busy':'Unrestricted duty';
  sheet.getRange(`A${r}:M${r}`).values=[[data.models[e.model],setting,e.point?.batch??(method==='mps'?1:'n.a.'),method==='mps'?(e.condition==='mps_cap50'?.5:1):'n.a.',null,null,null,null,null,e.point?.status??(e.model==='deepseek'&&precision==='fp32'?'capacity exceeds80GB':'pending measurement'),null,precision==='fp32'?'FP32':'BF16/FP16 + FP32 head',e.point?.platform??'']];
  const q=e.row;
  if(e.point){sheet.getRange(`E${r}:I${r}`).formulas=[[`=Measurements!AB${q}/100`,`=Measurements!${metric==='time'?'AC':'AD'}${q}`,`=F${r}/$F$7`,e.point.srcc_reference!==undefined?`=Measurements!H${q}`:'="n.a."',`=Measurements!AE${q}`]];sheet.getRange('K'+r).formulas=[[`=Measurements!Z${q}`]];}
  else sheet.getRange(`E${r}:I${r}`).values=[['n.a.','n.a.','n.a.','n.a.','n.a.']];
  sheet.getRange(`A${r}:M${r}`).format.fill=r%2===0?'#F2F5F7':'#FFFFFF';
  sheet.getRange('A'+r).format.font={name:'Arial',size:10,bold:true,color:colors[e.model]};r++;
 }
 sheet.getRange('D8:E19').setNumberFormat('0.0%');sheet.getRange('F7:F19').setNumberFormat('#,##0.000');sheet.getRange('G7:G19').setNumberFormat('0.00"×"');sheet.getRange('H7:H19').setNumberFormat('0.000000');sheet.freezePanes.freezeRows(6);sheet.freezePanes.freezeColumns(2);
 const notes=[['Control',method==='mps'?'Resource cap is the nominal MPS thread limit; actual visible SM count is audited. GPU busy can still be high under50% MPS.':'The predeclared near-saturation batch is fixed. Idle intervals occur only between actual inference groups. Actual busy is measured in the cost window.'],
 ['Energy','(338.2W retained host proxy + same-window sampled board power) × complete window duration, normalized by actual processed video evaluations.'],
 ['Accuracy','SRCC is an explicitly separate frozen full558-video reference when available. No reference of another precision is copied intoFP32 cells.'],
 ['Capacity',data.capacity_missing],['Scope','80GB rental models prioritized. Missing original-platform measurements and unfinished precision cases remain blank. Historical QwenB1 not changed.']];
 for(const[i,[label,text]]of notes.entries()){const rr=22+i*2;sheet.getRange('A'+rr).values=[[label]];sheet.getRange(`B${rr}:M${rr}`).merge();sheet.getRange('B'+rr).values=[[text]];sheet.getRange(`A${rr}:M${rr}`).format={font:{name:'Arial',size:10,color:'#53687A'},wrapText:true,rowHeight:34};}
}
book.recalculate();
function verify(b){const s=b.worksheets.getItem('Measurements');for(const e of entries)if(e.point){for(const[c,k]of [['AC','time16_ms'],['AD','energy16_j'],['AB','busy_percent']])if(Math.abs(s.getRange(c+e.row).values[0][0]-e.point[k])>1e-8)throw Error('Raw arithmetic mismatch '+c+e.row);}}
verify(book);
const first=entries.find(e=>e.point);if(first){const cell=raw.getRange('I'+first.row),v=cell.values[0][0];cell.values=[[v*1.01]];book.recalculate();if(book.worksheets.getItem('Measurements').getRange('AC'+first.row).values[0][0]===first.point.time16_ms)throw Error('Formula did not update');cell.values=[[v]];book.recalculate();verify(book);}
const errors=(await book.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:2000})).ndjson;
if(!errors.includes('Cell search matched 0 entries.'))throw Error(errors);
for(const {sheet}of views){const pic=await book.render({sheetName:sheet.name,range:'A1:J19',scale:1.1,format:'png'});await fs.writeFile(path.join(repo,'.codex_tmp','busy_resource_'+sheet.name+'.png'),new Uint8Array(await pic.arrayBuffer()));}
const rawPic=await book.render({sheetName:'Measurements',range:'A6:N12',scale:1,format:'png'});await fs.writeFile(path.join(repo,'.codex_tmp/busy_resource_raw.png'),new Uint8Array(await rawPic.arrayBuffer()));
await(await SpreadsheetFile.exportXlsx(book)).save(out);verify(await SpreadsheetFile.importXlsx(await FileBlob.load(out)));
await fs.writeFile(path.join(root,'workbook_receipt.json'),JSON.stringify({file:out,sha256:crypto.createHash('sha256').update(await fs.readFile(out)).digest('hex'),source_sha256:crypto.createHash('sha256').update(await fs.readFile(path.join(root,'eight_tables.json'))).digest('hex'),views:8,raw_sheets:1,audited_points:data.values.length,recalculation:true,reimport:true,formula_errors:0},null,2));
console.log(JSON.stringify({file:out,views:8,audited_points:data.values.length}));
