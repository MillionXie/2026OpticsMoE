/** Nine fixed-result plotting workbooks, two sheets each; no model inference. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {isDeepStrictEqual} from 'node:util';
const [inputFile,outputDir,authorDir]=process.argv.slice(2);
if(!inputFile||!outputDir||!authorDir)throw Error('Expected plot_data.json, output directory, authoring directory.');
const require=createRequire(path.join(path.resolve(authorDir),'package.json'));
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const data=JSON.parse(await fs.readFile(inputFile,'utf8'));
if(data.tasks.length!==9)throw Error('Expected human-confirmed nine tasks.');
await fs.mkdir(outputDir,{recursive:true});
const previewDir=path.join(outputDir,'verification');await fs.mkdir(previewDir,{recursive:true});
const countKeys=new Set(['samples','queries','candidates','evaluated_joints','pck_evaluated_joints','pckh_evaluated_joints']);
const metaLabels={model:'模型',pretrained_weights:'视觉预训练权重',epoch:'累计epoch',budget:'训练预算',test_samples:'测试样本／查询数',total_parameters:'总参数量',trainable_parameters:'可训练参数量',frozen_visual_parameters:'冻结视觉参数量',training_scope:'冻结与训练范围',note:'任务与版本说明',data_notes:'采用口径',code_commit:'代码Git版本',checkpoint_SHA256:'权重SHA256'};
const unit=(key,group,task)=>key==='psnr_db'?'dB':key.includes('pixel')?'px':countKeys.has(key)||group.includes('counts')?'个':(key==='mae'||key==='rmse')&&task.startsWith('lgvq_')?'MOS分': ['srcc','krcc','plcc','cc'].includes(key)?'相关系数':'无量纲';
const direction=key=>countKeys.has(key)?'—':/rmse|mae|kld|error/.test(key)?'越小越好':'越大越好';
const records=[];
for(const d of data.tasks){
  const wb=Workbook.create(),summary=wb.worksheets.add('整体指标'),samples=wb.worksheets.add('逐测试样本');
  summary.showGridLines=false;samples.showGridLines=false;summary.tabColor='#066CB1';samples.tabColor='#82B7DA';
  const groupLabel=g=>g.replace('overall/per_joint_pck_at_0.2_torso','逐关节PCK').replace('overall/per_joint_counts','逐关节样本数').replace('overall','整体').replace('mode/','编辑类型/');
  const metrics=d.overall_rows.map(([group,key,value])=>[groupLabel(group),key,value,unit(key,group,d.task),group.includes('counts')?'—':direction(key)]);
  const end=5+metrics.length;
  summary.getRange('A2').values=[[`${d.name}　ResNet-18`]];
  summary.getRange('A3').values=[['最终采用的固定末轮结果　2026-10-24']];
  summary.getRange('A5:E5').values=[['范围','指标','数值','单位','方向']];
  summary.getRange(`A6:E${end}`).values=metrics;
  summary.getRange('G5:H5').values=[['版本信息','内容']];
  const meta=d.metadata.map(([key,value])=>[metaLabels[key]??key,value]);
  meta.push(['逐样本数据说明',d.sample_notes]);
  summary.getRangeByIndexes(5,6,meta.length,2).values=meta;
  const maxRow=Math.max(end,5+meta.length);
  summary.getRange(`A2:H${maxRow}`).format={font:{name:'Arial',size:11,color:'#1C3041'},verticalAlignment:'center',rowHeight:25};
  summary.getRange('A2').format.font={name:'Arial',size:16,bold:true,color:'#164C73'};
  summary.getRange('A3').format.font={name:'Arial',size:10,color:'#5B6C78'};
  summary.getRange('A2:H3').format.rowHeight=30;
  const header={fill:'#066CB1',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',rowHeight:28};
  summary.getRange('A5:E5').format=header;summary.getRange('G5:H5').format=header;
  for(const [i,w] of [27,43,19,15,18,3,25,82].entries())summary.getRangeByIndexes(0,i,maxRow,1).format.columnWidth=w;
  summary.getRange(`C6:C${end}`).setNumberFormat('0.000000');summary.getRange(`C6:C${end}`).format.horizontalAlignment='right';
  for(let i=0;i<metrics.length;i++){
    if(i%2===0)summary.getRange(`A${i+6}:E${i+6}`).format.fill='#EEF4F8';
    if(metrics[i][3]==='个')summary.getRange(`C${i+6}`).setNumberFormat('#,##0');
    if(metrics[i][0]==='整体'&&metrics[i][1]===d.primary_metric)summary.getRange(`A${i+6}:E${i+6}`).format={fill:'#DCECF8',font:{name:'Arial',size:11,bold:true,color:'#124C75'}};
  }
  for(let i=0;i<meta.length;i++){
    if(i%2===0)summary.getRange(`G${i+6}:H${i+6}`).format.fill='#F3F6F8';
    if(typeof meta[i][1]==='number')summary.getRange(`H${i+6}`).setNumberFormat(Number.isInteger(meta[i][1])?'#,##0':'0.0000');
    else{
      const width=[...String(meta[i][1])].reduce((n,c)=>n+(c.charCodeAt(0)>255?2:1),0);
      if(width>76){summary.getRange(`H${i+6}`).format.wrapText=true;summary.getRange(`G${i+6}:H${i+6}`).format.rowHeight=Math.ceil(width/76)*20+8;}
    }
  }
  summary.tables.add(`A5:E${end}`,true,'OverallMetrics');
  // Raw observations start at row1: one sample/query per row, one variable per column.
  const cols=d.sample_headers.length,rows=d.sample_rows.length;
  samples.getRangeByIndexes(0,0,1,cols).values=[d.sample_headers];
  samples.getRangeByIndexes(1,0,rows,cols).values=d.sample_rows;
  samples.getRangeByIndexes(0,0,rows+1,cols).format={font:{name:'Arial',size:10,color:'#1C3041'},rowHeight:22,verticalAlignment:'center'};
  samples.getRangeByIndexes(0,0,1,cols).format={...header,rowHeight:45,wrapText:true,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'}};
  for(let i=0;i<cols;i++){
    const key=d.sample_headers[i],first=d.sample_rows.find(r=>r[i]!==null)?.[i];
    const text=typeof first==='string';
    const width=key==='instruction'?68:key==='sample_id'&&d.task.startsWith('lgvq_')?91:key==='sample_id'||key==='query_id'?36:text?25:key.length>25?32:24;
    samples.getRangeByIndexes(0,i,rows+1,1).format.columnWidth=width;
    if(!text){samples.getRangeByIndexes(1,i,rows,1).setNumberFormat(/rank|count|joints|_hit|^H@/.test(key)?'0':'0.000000');samples.getRangeByIndexes(1,i,rows,1).format.horizontalAlignment='right';}
  }
  const colName=n=>{let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
  const lastCol=colName(cols-1);samples.tables.add(`A1:${lastCol}${rows+1}`,true,'TestSamples');
  samples.freezePanes.freezeRows(1);samples.freezePanes.freezeColumns(1);
  wb.recalculate();
  if(!isDeepStrictEqual(summary.getRange(`A6:E${end}`).values,metrics))throw Error('Overall metrics changed: '+d.task);
  if(!isDeepStrictEqual(samples.getRangeByIndexes(1,0,rows,cols).values,d.sample_rows))throw Error('Test observations changed: '+d.task);
  const inspect=await wb.inspect({kind:'table',sheetId:summary.name,range:'A5:E12',include:'values',tableMaxRows:8,tableMaxCols:5,maxChars:600});
  const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:600});
  if(/"matchCount"\s*:\s*[1-9]/.test(errors.ndjson))throw Error('Spreadsheet error: '+errors.ndjson);
  for(const [s,range,suffix] of [[summary,`A2:H${maxRow}`,'overall'],[samples,`A1:${colName(Math.min(cols,8)-1)}9`,'samples']]){
    const preview=await wb.render({sheetName:s.name,range,scale:1,format:'png'});
    await fs.writeFile(path.join(previewDir,`${d.task}_${suffix}.png`),new Uint8Array(await preview.arrayBuffer()));
  }
  const outPath=path.join(outputDir,d.filename);const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(outPath);
  const raw=await fs.readFile(outPath);records.push({task:d.task,file:d.filename,sha256:createHash('sha256').update(raw).digest('hex'),bytes:raw.length,
    worksheets:2,test_rows:rows,test_columns:cols,all_values_readback_exact:true,primary_metric:d.primary_metric,primary_value:d.metrics[d.primary_metric]});
  console.log(JSON.stringify({task:d.task,file:outPath,samples:rows,primary:d.metrics[d.primary_metric],inspect:inspect.ndjson}));
}
await fs.writeFile(path.join(previewDir,'workbook_export_audit.json'),JSON.stringify({profile:data.profile,input_SHA256:createHash('sha256').update(await fs.readFile(inputFile)).digest('hex'),files:records},null,2));
console.log(JSON.stringify({completed_workbooks:records.length,sheets_each:2,total_test_records:records.reduce((s,r)=>s+r.test_rows,0)}));
