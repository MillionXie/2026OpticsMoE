// Complete-TEST scientific result workbook. Uses the desktop bundled artifact runtime only.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const [input,output,dependencies]=process.argv.slice(2);
const checkOnly=process.argv.includes('--check-only');
if(!input||!output||!dependencies)throw Error('Usage: builder results.json result.xlsx bundled_node_modules');
const require=createRequire(import.meta.url);
const entry=require.resolve('@oai/artifact-tool',{paths:[path.resolve(dependencies)]});
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(entry).href);
const data=JSON.parse(await fs.readFile(input,'utf8'));
if(data.rows.length!==500||data.examples.length!==20)throw Error('Full500 plus20 curated examples required');
const wb=Workbook.create();
const summary=wb.worksheets.add('Overall');
const examples=wb.worksheets.add('Examples20');
const appendix=wb.worksheets.add('TEST500');
const models=['Qwen','Simulation','Experiment'],metrics=['CLIP_T','DINOv2_I','CLIP_I'];
const pretty={'CLIP_T':'CLIP-T','DINOv2_I':'DINOv2-I','CLIP_I':'CLIP-I'};
const titles=['Sample ID','Category','Prompt',...models.flatMap(m=>metrics.map(k=>m+'\n'+pretty[k])),...models.map(m=>m+' image'),'Example ID'];
const values=data.rows.map(r=>[r.sample_id,r.category,r.prompt,...models.flatMap(m=>metrics.map(k=>r.scores[m][k])),...models.map(m=>r.paths[m]),r.example_id]);
appendix.getRange('A4:P504').values=[titles,...values];
appendix.getRange('A2').values=[['Complete TEST per-image results']];
appendix.getRange('A3').values=[['500 requests. Scores retain full source precision; all original PNGs are included.']];
appendix.tables.add('A4:P504',true,'TESTRecords');
appendix.freezePanes.freezeRows(4);appendix.freezePanes.freezeColumns(2);

const lookup=new Map(data.rows.map((r,i)=>[r.sample_id,i+5]));
examples.getRange('A2').values=[['Curated figure examples']];
examples.getRange('A3').values=[['20 manually selected examples. Overall statistics use all 500 requests, not this subset.']];
examples.getRange('A4:P4').values=[['Example','Sample ID','Prompt',...models.flatMap(m=>metrics.map(k=>m+'\n'+pretty[k])),...models.map(m=>m+' image'),'Prompt review']];
const exampleRows=data.examples.map(e=>{
 const r=lookup.get(e.sample_id);if(!r)throw Error('Unmatched selected request');
 return [e.example_id,`=TEST500!A${r}`,`=TEST500!C${r}`,...'DEFGHIJKL'.split('').map(c=>`=TEST500!${c}${r}`),
  ...models.map(m=>'selected20/'+e.example_id+'/'+m+'.png'),'Reviewed in all three groups'];
});
for(let i=0;i<20;i++){
 const v=exampleRows[i];examples.getRange(`A${i+5}:P${i+5}`).values=[v.map(x=>typeof x==='string'&&x.startsWith('=')?null:x)];
 examples.getRange(`B${i+5}:L${i+5}`).formulas=[v.slice(1,12)];
}
examples.tables.add('A4:P24',true,'SelectedExamples');examples.freezePanes.freezeRows(4);examples.freezePanes.freezeColumns(2);

summary.getRange('A2').values=[['T12 TEST comparison']];
summary.getRange('A3').values=[['Mean similarity scores. Simulation is the fixed pre-adaptation reference.']];
summary.getRange('A4:H4').values=[['Cohort','Metric','N','Qwen3-VL','Simulation','Experiment','EXP / SIM − 1','Direction']];
const modelColumns={Qwen:['D','E','F'],Simulation:['G','H','I'],Experiment:['J','K','L']};
const cohorts=['all','lamp','pillow'];
for(let g=0;g<3;g++)for(let k=0;k<3;k++){
 const row=5+g*3+k,cohort=cohorts[g];
 summary.getRange(`A${row}:C${row}`).values=[[(cohort==='all'?'All samples':cohort==='lamp'?'Lamp':'Pillow'),pretty[metrics[k]],null]];
 summary.getRange(`C${row}`).formulas=[[g===0?'=COUNTA(TEST500!$A$5:$A$504)':`=COUNTIFS(TEST500!$B$5:$B$504,"${cohort}")`]];
 for(let m=0;m<3;m++){
  const col=modelColumns[models[m]][k],dest='DEF'[m];
  const formula=g===0?`=AVERAGE(TEST500!${col}$5:${col}$504)`:`=AVERAGEIFS(TEST500!${col}$5:${col}$504,TEST500!$B$5:$B$504,"${cohort}")`;
  summary.getRange(`${dest}${row}`).formulas=[[formula]];
 }
 summary.getRange(`G${row}:H${row}`).formulas=[[`=F${row}/E${row}-1`,`=IF(F${row}>E${row},"Above simulation","At or below")`]];
}
summary.getRange('A16').values=[['Overall selection rule']];
summary.getRange('D16').formulas=[['=IF(AND(COUNTIFS(G5:G7,">0")<=1,MIN(G5:G7)>=-0.05,MAX(G5:G7)<=0.05),"Meets rule","Not met")']];
summary.getRange('A18').values=[['Simulation PT']];summary.getRange('C18').values=[[data.simulation_SHA256]];
summary.getRange('A19').values=[['Experiment PT']];summary.getRange('C19').values=[[data.experiment_SHA256]];
summary.getRange('A21').values=[['Higher scores indicate greater similarity, not literal prompt-compliance accuracy.']];
summary.getRange('A22').values=[['Selection rule: overall three differences within 5%; no more than one above simulation.']];
summary.getRange('A23').values=[['TEST-based candidate selection is a development result, not independent generalization.']];
summary.getRange('A24').values=[['Sources: immutable original metrics and the stated experimental run. Full SHA manifest accompanies the workbook.']];

for(const sheet of [summary,examples,appendix]){
 sheet.showGridLines=false;
 const end=sheet===appendix?504:24;sheet.getRange(`A2:P${end}`).format.font={name:'Arial',size:10,color:'#172333'};
 sheet.getRange(`A4:P${end}`).format.verticalAlignment='center';
 sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:'#172333'};
 const header=sheet.getRange(sheet===summary?'A4:H4':'A4:P4');
 header.format={fill:'#203864',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true};
 header.format.rowHeight=34;
 if(sheet!==summary){
  sheet.getRange('A:A').format.columnWidth=sheet===examples?10:49;
  sheet.getRange('B:B').format.columnWidth=sheet===examples?49:12;
  sheet.getRange('C:C').format.columnWidth=72;
  sheet.getRange('D:L').format.columnWidth=14;sheet.getRange(`D5:L${end}`).setNumberFormat('0.0000');
  sheet.getRange(`D5:L${end}`).format.horizontalAlignment='right';
  sheet.getRange('M:O').format.columnWidth=62;sheet.getRange('P:P').format.columnWidth=32;
  sheet.getRange(`C5:C${end}`).format.wrapText=true;sheet.getRange(`A5:P${end}`).format.rowHeight=36;
 }
}
summary.tabColor='#203864';examples.tabColor='#55749B';
for(const [col,width]of Object.entries({A:17,B:16,C:9,D:16,E:16,F:16,G:21,H:22}))summary.getRange(`${col}:${col}`).format.columnWidth=width;
summary.getRange('D5:F13').setNumberFormat('0.0000');summary.getRange('G5:G13').setNumberFormat('+0.00%;-0.00%;0.00%');
summary.getRange('C5:G13').format.horizontalAlignment='right';summary.getRange('A5:H13').format.rowHeight=23;
summary.getRange('G5:G13').conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0,format:{fill:'#FFF0D6',font:{color:'#8F4F0A'}}});
summary.getRange('D16').conditionalFormats.addCustom('D16="Not met"',{fill:'#FBE8E6',font:{color:'#A02B23',bold:true}});
summary.getRange('A3').format.font={name:'Arial',size:10,color:'#536171',italic:true};
examples.getRange('A5:A24').setNumberFormat('00');appendix.getRange('P5:P504').setNumberFormat('00');

wb.recalculate();
const means=summary.getRange('D5:F13').values;
for(let i=0;i<9;i++)for(let j=0;j<3;j++){
 const expected=data.overall[i][models[j]],actual=means[i][j];
 if(typeof actual!=='number'||Math.abs(actual-expected)>1e-12)throw Error(`Aggregate mismatch ${i}/${j}: ${actual} vs ${expected}`);
}
if(summary.getRange('D16').values[0][0]!== (data.rule.overall_at_most_one_above_and_within5?'Meets rule':'Not met'))throw Error('Selection-rule formula mismatch');
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:2000});
const qaDir=path.join(path.dirname(output),'_review');await fs.mkdir(qaDir,{recursive:true});
const checks={means,selectionRule:summary.getRange('D16').values,formulaScan:errors.ndjson};
await fs.writeFile(path.join(qaDir,'workbook_checks.json'),JSON.stringify(checks,null,2));
for(const [sheetName,range,file]of [['Overall','A2:H16','overall_sheet.png'],['Examples20','A2:L8','examples_sheet.png'],['TEST500','A2:L8','appendix_sheet.png']]){
 const preview=await wb.render({sheetName,range,scale:1,format:'png'});
 await fs.writeFile(path.join(qaDir,file),new Uint8Array(await preview.arrayBuffer()));
}
if(!checkOnly){const exported=await SpreadsheetFile.exportXlsx(wb);await exported.save(output);}
console.log(JSON.stringify({complete:true,output:checkOnly?null:output,checkOnly,TEST:500,examples:20,sheets:3,aggregatesReconciled:true,selectionRule:data.rule}));
