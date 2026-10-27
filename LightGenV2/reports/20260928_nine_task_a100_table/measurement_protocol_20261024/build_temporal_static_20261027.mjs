// One-sheet, fixed-value display. No cell formulas or workbook imports.
import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const [input,out]=process.argv.slice(2);
const data=JSON.parse(await fs.readFile(input,'utf8'));
if(data.status!=='raw_rechecked_fixed_values'||data.rows.length!==13||data.headers.length!==12)throw Error('Audited full fixed-value cohort required');
try{await fs.access(out);throw Error('Preserve existing output');}catch(e){if(e.code!=='ENOENT')throw e;}
const w=Workbook.create();const s=w.worksheets.add('统一结果');s.showGridLines=false;
s.getRange('A1:L18').format={font:{name:'Arial',size:11,color:'#192E40'},rowHeight:30,verticalAlignment:'center'};
s.getRange('A1:L14').values=[data.headers,...data.rows];
s.getRange('A1:L1').format={fill:'#243B53',font:{bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true,rowHeight:44};
const widths=[27,20,10,8,18,18,18,18,27,22,22,27];
for(let i=0;i<widths.length;i++)s.getRange(`${String.fromCharCode(65+i)}1`).format.columnWidth=widths[i];
s.getRange('C2:L14').format.horizontalAlignment='right';
s.getRange('C2:D14').setNumberFormat('0');s.getRange('E2:I14').setNumberFormat('#,##0.000');s.getRange('J2:L14').setNumberFormat('0.000"x"');
s.getRange('A2:L2').format.fill='#EAF1F7';s.getRange('B2').format.font.bold=true;s.getRange('E2').format.font.bold=true;
for(const [i,note] of data.notes.entries()){
  s.getRange('A'+(16+i)).values=[[note]];
  s.getRange(`A${16+i}:L${16+i}`).format={font:{size:10,color:'#526879'},rowHeight:25};
}
w.recalculate();
if(s.getRange('A1:L18').formulas.flat().some(Boolean))throw Error('No formulas permitted');
console.log((await w.inspect({kind:'table',range:'统一结果!A1:L4',include:'values,formulas',tableMaxRows:4,tableMaxCols:12,maxChars:1500})).ndjson);
await fs.mkdir(out);
const target=path.join(out,'Temporal_LightGen与baseline_固定数值_20261027.xlsx');
await (await SpreadsheetFile.exportXlsx(w)).save(target);
const preview=await w.render({sheetName:'统一结果',range:'A1:L18',scale:1.3,format:'png'});
await fs.writeFile(path.join(out,'preview.png'),new Uint8Array(await preview.arrayBuffer()));
await fs.copyFile(input,path.join(out,'static_values.json'));
console.log(JSON.stringify({target,rows:13,sheets:1,formulas:0}));
