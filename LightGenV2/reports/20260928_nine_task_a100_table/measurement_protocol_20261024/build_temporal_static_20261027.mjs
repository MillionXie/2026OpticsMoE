// Fixed-value comparison plus a source-derived LightGen phase table.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';

const [input,out,revision]=process.argv.slice(2);
const data=JSON.parse(await fs.readFile(input,'utf8'));
if(data.status!=='raw_rechecked_fixed_values'||data.rows.length!==13||data.headers.length!==13)throw Error('Audited full fixed-value cohort required');
try{await fs.access(out);if(revision!=='--revise-unsealed')throw Error('Preserve existing output');
  try{await fs.access(path.join(out,'delivery_manifest.json'));throw Error('Never modify sealed delivery');}catch(e){if(e.code!=='ENOENT')throw e;}
}catch(e){if(e.code!=='ENOENT')throw e;}
const sourceBytes=await fs.readFile(data.base_workbook);
if(createHash('sha256').update(sourceBytes).digest('hex')!==data.base_workbook_sha256)throw Error('Reinspect changed source workbook');
const w=await SpreadsheetFile.importXlsx(await FileBlob.load(data.base_workbook));
const s=w.worksheets.getItem('统一结果'); // Preserve all user deletions, values and formatting.
const p=data.appendix_table;
if(!p||p.rows.length!==18||p.headers.length!==4||p.module_rows.length!==6)throw Error('Complete source-derived appendix table required');
const t=w.worksheets.getItem('LightGen附录计时');
const lastRow=3+p.rows.length,notesStart=p.rows.length+6,endRow=notesStart+p.notes.length-1;
if(data.edit_mode==='raw_appendix'){
  for(const row of [5,7,9,13,15,17])t.getRange('D'+row).values=[[p.rows[row-4][3]]];
}else if(data.edit_mode==='optical_control_to_electronics'){
  for(const row of [5,6,7,8,9,10,13,14,15,16,17,18])t.getRange('C'+row).values=[[p.rows[row-4][2]]];
  for(const row of [6,8,10,14,16,18])t.getRange('D'+row).values=[[p.rows[row-4][3]]];
  t.getRange('A26').values=[[p.notes[2]]];
}else{
t.getUsedRange().clear({applyTo:'all'});t.showGridLines=false;
t.getRange(`A1:D${endRow}`).format={font:{name:'Arial',size:11,color:'#192E40'},rowHeight:40,verticalAlignment:'center'};
for(const [i,width] of [22,54,28,105].entries())t.getRange(`${String.fromCharCode(65+i)}1`).format.columnWidth=width;
t.getRange('A1').values=[[p.title]];t.getRange('A1').format.font={size:14,bold:true};
t.getRange('A2').values=[[p.scope]];t.getRange('A2:D2').format={font:{size:10,color:'#526879'},rowHeight:28};
t.getRange(`A3:D${lastRow}`).values=[p.headers,...p.rows];
t.getRange('A3:D3').format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true,rowHeight:55};
t.getRange(`A4:D${lastRow}`).format.wrapText=true;
t.getRange(`C4:C${lastRow}`).format.horizontalAlignment='right';t.getRange(`C4:C${lastRow}`).setNumberFormat('0.000000');
let previous=null,shade=false;
for(const [i,type] of p.row_types.entries()){
  const range=t.getRange(`A${i+4}:D${i+4}`);
  if(p.rows[i][0]!==previous){shade=!shade;previous=p.rows[i][0];}
  if(shade)range.format.fill='#F3F6F9';
  if(i===0)range.format.rowHeight=70;
  if(i===7||i===8||i===16)range.format.rowHeight=55;
  if(type==='subtotal')range.format={fill:'#EAF1F7',font:{bold:true}};
  if(type==='total')range.format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},rowHeight:42};
  if(type==='unmeasured')range.format={font:{color:'#526879'},rowHeight:29};
}
for(const [i,note] of p.notes.entries()){
  t.getRange('A'+(notesStart+i)).values=[[note]];
  t.getRange(`A${notesStart+i}:D${notesStart+i}`).format={font:{size:10,color:'#526879'},rowHeight:27,wrapText:false};
}
t.freezePanes.freezeRows(3);
}
const extraPreviews=[];
if(data.edit_mode==='raw_appendix'){
  const raw=data.raw_appendix;
  const create=(name,headers,values,row,widths)=>{
    const sheet=w.worksheets.add(name);sheet.showGridLines=false;
    const end=row+values.length,last=String.fromCharCode(64+headers.length);
    sheet.getRange(`A1:${last}${end+4}`).format={font:{name:'Arial',size:10,color:'#192E40'},rowHeight:25,verticalAlignment:'center'};
    for(const [i,width] of widths.entries())sheet.getRange(`${String.fromCharCode(65+i)}1`).format.columnWidth=width;
    sheet.getRange(`A${row}:${last}${end}`).values=[headers,...values];
    sheet.getRange(`A${row}:${last}${row}`).format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},rowHeight:52,wrapText:true,horizontalAlignment:'center'};
    sheet.getRange(`A${row+1}:${last}${end}`).setNumberFormat('0.000000000');
    sheet.freezePanes.freezeRows(row);
    return sheet;
  };
  const header=(sheet,row,headers,values)=>{
    const last=String.fromCharCode(64+headers.length),end=row+values.length;
    sheet.getRange(`A${row}:${last}${end}`).format={font:{name:'Arial',size:10,color:'#192E40'},rowHeight:26,verticalAlignment:'center'};
    sheet.getRange(`A${row}:${last}${end}`).values=[headers,...values];
    sheet.getRange(`A${row}:${last}${row}`).format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},rowHeight:52,wrapText:true,horizontalAlignment:'center'};
    sheet.getRange(`A${row+1}:${last}${end}`).setNumberFormat('0.000000000');
  };
  const title=(sheet,text,note)=>{
    sheet.getRange('A1').values=[[text]];sheet.getRange('A1').format.font={bold:true,size:14};
    sheet.getRange('A2').values=[[note]];sheet.getRange('A2').format.font={size:10,color:'#526879'};
  };
  const f=create('原始分块核对',raw.summary_headers,raw.summary_rows,5,[24,48,25,25,25,25,28]);
  title(f,'LightGen原始分块与附录核对','三轮共384场，每场16视频64帧。原事件累计不去重；等效时间 =（并行后计算累计 + 控制累计）/384。');
  f.getRange('A3').values=[['光学数据为固定器件时序，电子数据来自原A100事件。未处理重叠的累计值不可相加作为总时间。']];
  f.getRange('A6:B22').format.wrapText=true;f.getRange('A6:G22').format.rowHeight=38;
  const totals=Array.from({length:4},(_,j)=>raw.summary_rows.reduce((a,r)=>a+r[j+3],0));
  f.getRange('A23:G23').values=[['合计','',null,...totals]];
  f.getRange('A23:G23').format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},rowHeight:32};
  f.getRange('D23:G23').setNumberFormat('0.000000000');
  f.getRange('A25').values=[['逐轮分块 每轮128场']];header(f,27,raw.round_headers,raw.round_rows);
  f.getRange('A28:A78').setNumberFormat('0');f.getRange('B28:C78').format.wrapText=true;f.getRange('A28:G78').format.rowHeight=37;
  f.getRange('A80:G111').format={font:{name:'Arial',size:10,color:'#526879'},rowHeight:27};
  for(const [i,note] of raw.method_notes.entries())f.getRange('A'+(81+i)).values=[[note]];
  f.getRange('A90').values=[['单个光周期的三个固定组成部分']];
  header(f,91,['组成部分','单次时长/ms'],[...raw.optical_components,['单周期合计',raw.optical_components.reduce((a,r)=>a+r[1],0)]]);
  for(const [i,source] of raw.sources.entries()){
    const n=99+i*4;f.getRange('A'+n).values=[[`第${source.round}轮 ${source.run} 原文件SHA256`]];
    for(const [j,key] of ['raw_sha256','trace_sha256','ledger_sha256'].entries())f.getRange(`A${n+1+j}:B${n+1+j}`).values=[[["原始计时","原始依赖节点","批准时间账本"][j],source[key]]];
  }
  const g=create('调度分类核对',raw.global_headers,raw.global_rows,5,[25,17,27,29,25,28]);
  title(g,'全程调度分类核对','原始明细逐条保存在下一表。本表是额外统计，不替代原始记录；三轮累计除384，对应附录全程调度行。');
  g.getRange('A3').values=[['原始同步区间含等待GPU完成的阻塞；依赖分离后不把同一等待重复计时；并行后只计原最长依赖链上的节点。']];
  g.getRange('B6:B12').setNumberFormat('0');
  g.getRange('A13:F13').values=[['合计',...Array.from({length:5},(_,j)=>raw.global_rows.reduce((a,r)=>a+r[j+1],0))]];
  g.getRange('A13:F13').format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},rowHeight:32};
  g.getRange('B13').setNumberFormat('0');g.getRange('C13:F13').setNumberFormat('0.000000000');
  g.getRange('A15').values=[['逐轮分类统计 每轮128场']];header(g,17,raw.global_round_headers,raw.global_round_rows);
  g.getRange('G1').format.columnWidth=28;g.getRange('A18:A38').setNumberFormat('0');g.getRange('C18:C38').setNumberFormat('0');
  g.getRange('A40').values=[['最后一列三轮归一后相加等于0.7377455263134227 ms/16视频，与附录C20一致。部件相关的提交与队列不在此重复累计。']];
  const q=create('调度原始明细',raw.raw_headers,raw.raw_rows,4,[8,11,25,15,23,23,24,26,16,23,23,23,28,66]);
  title(q,'全程调度原始逐条明细','每行保留一个原始动作或相邻间隙，不合并、平均或堆叠。时间相对各轮原始host_start；场次沿用0—127。');
  q.getRange('A3').values=[['原始开始/结束与原始区间时长来自原记录；核算时间单列。同步阻塞通过依赖表达；零时长依赖标记不作为实际动作。']];
  q.getRange(`A5:B${4+raw.raw_rows.length}`).setNumberFormat('0');
  extraPreviews.push(['raw_summary','原始分块核对','A1:G23'],['raw_rounds','原始分块核对','A25:G44'],
    ['raw_components','原始分块核对','A90:C95'],['global_summary','调度分类核对','A1:F13'],
    ['global_rounds','调度分类核对','A15:G38'],['raw_controls_observed','调度原始明细','A1:H15'],
    ['raw_controls_accounted','调度原始明细','I4:N15']);
}
w.recalculate();
if(s.getUsedRange().formulas.flat().some(Boolean))throw Error('No formulas permitted');
if(t.getRange(`A1:D${endRow}`).formulas.flat().some(Boolean))throw Error('No phase formulas permitted');
console.log((await w.inspect({kind:'table',range:'统一结果!A1:M4',include:'values,formulas',tableMaxRows:4,tableMaxCols:13,maxChars:1500})).ndjson);
await fs.mkdir(out,{recursive:true});
const target=path.join(out,'Temporal_LightGen与baseline_模型对比与附录计时_20261027.xlsx');
await (await SpreadsheetFile.exportXlsx(w)).save(target);
const preview=await w.render({sheetName:'统一结果',range:'A1:M16',scale:1.2,format:'png'});
await fs.writeFile(path.join(out,'preview.png'),new Uint8Array(await preview.arrayBuffer()));
for(const [name,range] of [['phase_pre_vision','A1:D11'],['phase_language_control',`A12:D${lastRow}`],['phase_notes',`A${notesStart-1}:D${endRow}`]]){
  const blob=await w.render({sheetName:'LightGen附录计时',range,scale:1.3,format:'png'});
  await fs.writeFile(path.join(out,name+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
for(const [name,sheetName,range] of extraPreviews){
  const blob=await w.render({sheetName,range,scale:1.2,format:'png'});
  await fs.writeFile(path.join(out,name+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
await fs.copyFile(input,path.join(out,'static_values.json'));
console.log(JSON.stringify({target,rows:13,sheets:data.edit_mode==='raw_appendix'?5:2,formulas:0}));
