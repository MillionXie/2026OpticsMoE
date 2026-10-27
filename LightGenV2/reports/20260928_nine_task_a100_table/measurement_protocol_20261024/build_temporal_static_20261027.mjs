// Fixed-value comparison plus a source-derived LightGen phase table.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';

const [input,out]=process.argv.slice(2);
const data=JSON.parse(await fs.readFile(input,'utf8'));
if(data.status!=='raw_rechecked_fixed_values'||data.rows.length!==13||data.headers.length!==13)throw Error('Audited full fixed-value cohort required');
try{await fs.access(out);throw Error('Preserve existing output');}catch(e){if(e.code!=='ENOENT')throw e;}
const sourceBytes=await fs.readFile(data.base_workbook);
if(createHash('sha256').update(sourceBytes).digest('hex')!==data.base_workbook_sha256)throw Error('Reinspect changed source workbook');
const w=await SpreadsheetFile.importXlsx(await FileBlob.load(data.base_workbook));
const s=w.worksheets.getItem('统一结果'); // Preserve all user deletions, values and formatting.
const p=data.appendix_table;
if(!p||p.rows.length!==36||p.module_rows.length!==6)throw Error('Complete source-derived appendix table required');
const t=w.worksheets.getItem('LightGen附录计时');t.getUsedRange().clear({applyTo:'all'});t.showGridLines=false;
const lastRow=3+p.rows.length,notesStart=p.rows.length+6,modulesStart=notesStart+p.notes.length+2,endRow=modulesStart+p.module_rows.length;
t.getRange(`A1:G${endRow}`).format={font:{name:'Arial',size:11,color:'#192E40'},rowHeight:34,verticalAlignment:'center'};
for(const [i,width] of [21,48,23,23,23,23,88].entries())t.getRange(`${String.fromCharCode(65+i)}1`).format.columnWidth=width;
t.getRange('A1').values=[[p.title]];t.getRange('A1').format.font={size:14,bold:true};
t.getRange(`A3:G${lastRow}`).values=[p.headers,...p.rows];
t.getRange('A3:G3').format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true,rowHeight:55};
t.getRange(`A4:G${lastRow}`).format.wrapText=true;
t.getRange(`C4:F${lastRow}`).format.horizontalAlignment='right';t.getRange(`C4:F${lastRow}`).setNumberFormat('0.000000');
for(const [i,type] of p.row_types.entries()){
  const range=t.getRange(`A${i+4}:G${i+4}`);
  if(type==='subtotal')range.format={fill:'#EAF1F7',font:{bold:true}};
  if(type==='total')range.format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},rowHeight:42};
  if(type==='unmeasured')range.format={font:{color:'#526879'},rowHeight:29};
}
for(const [i,note] of p.notes.entries()){
  t.getRange('A'+(notesStart+i)).values=[[note]];
  t.getRange(`A${notesStart+i}:G${notesStart+i}`).format={font:{size:10,color:'#526879'},rowHeight:25,wrapText:false};
}
t.getRange(`A${modulesStart}:D${endRow}`).values=[p.module_headers,...p.module_rows];
t.getRange(`A${modulesStart}:D${modulesStart}`).format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:32,wrapText:true};
t.getRange(`B${modulesStart+1}:D${endRow}`).format.horizontalAlignment='right';t.getRange(`B${modulesStart+1}:D${endRow}`).setNumberFormat('0.000000');
t.freezePanes.freezeRows(3);
w.recalculate();
if(s.getUsedRange().formulas.flat().some(Boolean))throw Error('No formulas permitted');
if(t.getRange(`A1:G${endRow}`).formulas.flat().some(Boolean))throw Error('No phase formulas permitted');
console.log((await w.inspect({kind:'table',range:'统一结果!A1:M4',include:'values,formulas',tableMaxRows:4,tableMaxCols:13,maxChars:1500})).ndjson);
await fs.mkdir(out);
const target=path.join(out,'Temporal_LightGen与baseline_模型对比与附录计时_20261027.xlsx');
await (await SpreadsheetFile.exportXlsx(w)).save(target);
const preview=await w.render({sheetName:'统一结果',range:'A1:M16',scale:1.2,format:'png'});
await fs.writeFile(path.join(out,'preview.png'),new Uint8Array(await preview.arrayBuffer()));
for(const [name,range] of [['phase_pre_vision','A1:G21'],['phase_language_control',`A22:G${lastRow}`],['phase_notes',`A${notesStart-1}:G${endRow}`]]){
  const blob=await w.render({sheetName:'LightGen附录计时',range,scale:1.3,format:'png'});
  await fs.writeFile(path.join(out,name+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
await fs.copyFile(input,path.join(out,'static_values.json'));
console.log(JSON.stringify({target,rows:13,sheets:2,formulas:0}));
