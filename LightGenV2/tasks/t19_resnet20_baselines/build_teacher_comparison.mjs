/** Build the presentation table from immutable, audited result metadata.
 * Usage: bundled-node build_teacher_comparison.mjs <sources.json> <output-dir> <authoring-dir>
 * authoring-dir/node_modules must point to the bundled workspace dependency runtime.
 * This script never loads a training checkpoint or performs model inference.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';

const [inputFile,outputDir,authorDir]=process.argv.slice(2);
if (!inputFile || !outputDir || !authorDir) throw new Error('Expected input JSON, output directory, authoring directory.');
const require=createRequire(path.join(path.resolve(authorDir),'package.json'));
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const data=JSON.parse(await fs.readFile(inputFile,'utf8'));
if (data.rows.length!==17) throw new Error('Expected nine tasks with all completed ABO budgets: 17 rows.');
const taskRoot=path.resolve(path.dirname(inputFile),'../..');
for(const [source,expected] of Object.entries(data.sources)){
  const actual=createHash('sha256').update(await fs.readFile(path.join(taskRoot,source))).digest('hex');
  if(actual!==expected) throw new Error(`Source changed: ${source}`);
}
const wb=Workbook.create();
const sh=wb.worksheets.add('最终结果与全部ABO预算');
sh.showGridLines=false;
sh.tabColor='#204B70';
const headers=['任务／数据集','主指标','当前版本／用途','epoch','当前结果','完整参数量','可训练参数量','预训练来源','冻结与训练范围','ResNet-20 last30','RN20完整＝可训参数','Ours Sim 原图','Qwen3VL 原图','版本与协议说明'];
const values=data.rows.map(r=>[r.task,r.metric,r.variant,r.epoch,r.value,r.total,r.trainable,r.pretraining,r.scope,r.rn20,r.rn20_total,r.ours_sim,r.qwen,r.note]);
sh.getRange('A2').values=[[data.title]];
sh.getRange('A3').values=[['2026-10-24　固定末轮结果；总参数含冻结骨干；Ours/Qwen为历史参考']];
sh.getRange('A5:N5').values=[headers];
sh.getRange('A6:N22').values=values;
sh.getRange('A2:N22').format.font={name:'Arial',size:11,color:'#172B3A'};
sh.getRange('A2').format.font={size:16,bold:true};
sh.getRange('A3').format.font={size:11,color:'#526575'};
sh.getRange('A2:N3').format.rowHeight=28;
sh.getRange('A5:N5').format={fill:'#204B70',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:38,wrapText:true,horizontalAlignment:'center',verticalAlignment:'center'};
sh.getRange('A6:N22').format={rowHeight:58,verticalAlignment:'center',wrapText:true};
sh.getRange('D6:G22').format.horizontalAlignment='right';
sh.getRange('J6:M22').format.horizontalAlignment='right';
sh.getRange('D6:D22').setNumberFormat('0');
sh.getRange('F6:G22').setNumberFormat('#,##0');
sh.getRange('K6:K22').setNumberFormat('#,##0');
const widths=[25,21,29,10,15,18,18,25,43,20,25,18,18,67];
for (let i=0;i<widths.length;i++) sh.getRangeByIndexes(0,i,22,1).format.columnWidth=widths[i];
for (let i=0;i<data.rows.length;i++){
  const r=data.rows[i], row=i+6;
  if(i%2===0) sh.getRange(`A${row}:N${row}`).format.fill='#F1F5F9';
  const nf=(r.metric==='SRCC'||r.metric==='CC')?'0.0000':r.metric==='PSNR (dB)'?'0.0000" dB"':'0.00%';
  for (const col of ['E','J','L','M']) sh.getRange(`${col}${row}`).setNumberFormat(nf);
  if(r.variant.startsWith('采用')) sh.getRange(`C${row}:E${row}`).format.font={bold:true};
  if(i===0||r.task_id!==data.rows[i-1].task_id) sh.getRange(`A${row}:N${row}`).format.borders={top:{style:'thin',color:'#CCD7E0'}};
}
sh.tables.add('A5:N22',true,'T19FinalComparison');
sh.freezePanes.freezeRows(5);
sh.freezePanes.freezeColumns(2);

// A single worksheet: common model definitions and non-comparable reference details stay beside the table.
sh.getRange('A25').values=[['口径与必要说明']];
sh.getRange('A25').format.font={name:'Arial',size:12,bold:true,color:'#204B70'};
data.notes.forEach((s,i)=>{
  sh.getRange(`A${26+i}`).values=[[`${i+1}. ${s}`]];
  sh.getRange(`A${26+i}:N${26+i}`).format={rowHeight:24,font:{name:'Arial',size:11,color:'#394F60'},wrapText:false};
});
const sourceStart=44;
sh.getRange(`A${sourceStart}`).values=[['数据来源与可追溯性']];
sh.getRange(`A${sourceStart}`).format.font={name:'Arial',size:12,bold:true,color:'#204B70'};
sh.getRange(`A${sourceStart+1}`).values=[[data.reference_source]];
sh.getRange(`A${sourceStart+2}`).values=[['当前与RN20参数/成绩来自各run独立审计；下列SHA绑定本次汇总读取的原始证据。']];
sh.getRange(`A${sourceStart+1}:N${sourceStart+2}`).format={rowHeight:24,font:{name:'Arial',size:11,color:'#526575'}};
const sourceRows=Object.entries(data.sources).map(([source,sha])=>[source,sha]);
const start=sourceStart+4;
sh.getRange(`A${start}`).values=[['证据文件（相对T19任务目录）']];
sh.getRange(`I${start}`).values=[['SHA256']];
sourceRows.forEach(([source,sha],i)=>{
  const row=start+1+i;
  sh.getRange(`A${row}`).values=[[source]];
  sh.getRange(`I${row}`).values=[[sha]];
  sh.getRange(`A${row}:N${row}`).format={rowHeight:24,font:{name:'Arial',size:10,color:'#526575'},wrapText:false};
});
wb.recalculate();
if(!isDeepStrictEqual(sh.getRange('A6:N22').values,values)) throw new Error('Written comparison matrix differs from audited inputs.');
const check=await wb.inspect({kind:'table',sheetId:sh.name,range:'A5:M22',include:'values,formulas',tableMaxRows:18,tableMaxCols:13,maxChars:1800});
console.log(check.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'final formula error scan',maxChars:1200});
console.log(errors.ndjson);
await fs.mkdir(outputDir,{recursive:true});
for(const [name,range] of [['comparison','A2:N22'],['model_details','A25:N41']]){
  const img=await wb.render({sheetName:sh.name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(outputDir,`${name}_preview.png`),new Uint8Array(await img.arrayBuffer()));
}
const out=await SpreadsheetFile.exportXlsx(wb);
const output=path.join(outputDir,'T19_最终基线对比_含ABO全部预算.xlsx');
await out.save(output);
console.log(JSON.stringify({output,rows:data.rows.length,sheets:1,sourceFiles:Object.keys(data.sources).length}));
