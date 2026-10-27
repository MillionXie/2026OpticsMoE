// Fixed-value comparison plus a source-derived LightGen phase table.
import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const [input,out]=process.argv.slice(2);
const data=JSON.parse(await fs.readFile(input,'utf8'));
if(data.status!=='raw_rechecked_fixed_values'||data.rows.length!==13||data.headers.length!==13)throw Error('Audited full fixed-value cohort required');
try{await fs.access(out);throw Error('Preserve existing output');}catch(e){if(e.code!=='ENOENT')throw e;}
const w=Workbook.create();const s=w.worksheets.add('统一结果');s.showGridLines=false;
s.getRange('A1:M23').format={font:{name:'Arial',size:11,color:'#192E40'},rowHeight:30,verticalAlignment:'center'};
s.getRange('A1:M14').values=[data.headers,...data.rows];
s.getRange('A1:M1').format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true,rowHeight:44};
const widths=[27,35,18,10,8,18,18,18,18,27,22,22,27];
for(let i=0;i<widths.length;i++)s.getRange(`${String.fromCharCode(65+i)}1`).format.columnWidth=widths[i];
s.getRange('C2:M14').format.horizontalAlignment='right';
s.getRange('C2:C14').setNumberFormat('0.000000');s.getRange('D2:E14').setNumberFormat('0');s.getRange('F2:J14').setNumberFormat('#,##0.000');s.getRange('K2:M14').setNumberFormat('0.000"x"');
s.getRange('A2:M2').format.fill='#EAF1F7';s.getRange('B2').format.font.bold=true;s.getRange('F2').format.font.bold=true;
for(const [i,note] of data.notes.entries()){
  s.getRange('A'+(16+i)).values=[[note]];
  s.getRange(`A${16+i}:M${16+i}`).format={font:{size:10,color:'#526879'},rowHeight:25};
}
const p=data.appendix_table;
if(!p||p.rows.length!==7||p.rounds.length!==3)throw Error('Complete source-derived appendix table required');
const t=w.worksheets.add('LightGen附录计时');t.showGridLines=false;
t.getRange('A1:E28').format={font:{name:'Arial',size:11,color:'#192E40'},rowHeight:30,verticalAlignment:'center'};
for(const [i,width] of [24,62,22,15,88].entries())t.getRange(`${String.fromCharCode(65+i)}1`).format.columnWidth=width;
t.getRange('A1').values=[[p.title]];t.getRange('A1').format.font={size:14,bold:true};
t.getRange('A3:E10').values=[p.headers,...p.rows];
t.getRange('A3:E3').format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true,rowHeight:36};
t.getRange('A4:E10').format={wrapText:true,rowHeight:58};
t.getRange('C4:D10').format.horizontalAlignment='right';t.getRange('C4:C10').setNumberFormat('0.000000');t.getRange('D4:D10').setNumberFormat('0.00%');
t.getRange('A10:E10').format={fill:'#EAF1F7',font:{bold:true},rowHeight:40};
for(const [i,note] of p.notes.entries()){
  t.getRange('A'+(13+i)).values=[[note]];
  t.getRange(`A${13+i}:E${13+i}`).format={font:{size:10,color:'#526879'},rowHeight:25,wrapText:false};
}
t.getRange('A25:E28').values=[p.round_headers,...p.rounds];
t.getRange('A25:E25').format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:32};
t.getRange('B26:E28').format.horizontalAlignment='right';t.getRange('A26:B28').setNumberFormat('0');t.getRange('C26:E28').setNumberFormat('0.000000');
w.recalculate();
if(s.getRange('A1:M23').formulas.flat().some(Boolean))throw Error('No formulas permitted');
if(t.getRange('A1:E28').formulas.flat().some(Boolean))throw Error('No phase formulas permitted');
console.log((await w.inspect({kind:'table',range:'统一结果!A1:M4',include:'values,formulas',tableMaxRows:4,tableMaxCols:13,maxChars:1500})).ndjson);
await fs.mkdir(out);
const target=path.join(out,'Temporal_LightGen与baseline_模型对比与附录计时_20261027.xlsx');
await (await SpreadsheetFile.exportXlsx(w)).save(target);
const preview=await w.render({sheetName:'统一结果',range:'A1:M23',scale:1.3,format:'png'});
await fs.writeFile(path.join(out,'preview.png'),new Uint8Array(await preview.arrayBuffer()));
for(const [name,range] of [['phase_preview','A1:E10'],['phase_notes','A12:E28']]){
  const blob=await w.render({sheetName:'LightGen附录计时',range,scale:1.3,format:'png'});
  await fs.writeFile(path.join(out,name+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
await fs.copyFile(input,path.join(out,'static_values.json'));
console.log(JSON.stringify({target,rows:13,sheets:2,formulas:0}));
