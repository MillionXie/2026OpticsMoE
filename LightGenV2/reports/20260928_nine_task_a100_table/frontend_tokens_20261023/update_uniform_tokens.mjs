// Narrow update: common input-workload numerator, with original costs preserved.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const dir=path.dirname(fileURLToPath(import.meta.url));
const oldName='九任务_核心与含前端_Tokens_20261023.xlsx';
const newName='九任务_统一输入Tokens_20261024.xlsx';
const auditName='frontend_tokens_audit_20261023.json';
const source=JSON.parse(await fs.readFile(path.join(dir,auditName),'utf8'));
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(dir,oldName)));
const inputs=wb.worksheets.getItem('输入依据与Bridge');
const ink='#17365D',orange='#FCE4D6',blue='#DDEBF7';
const digest=buffer=>crypto.createHash('sha256').update(buffer).digest('hex');
const value=(s,c,v)=>{s.getRange(c).values=[[v]];};
const formula=(s,c,v)=>{s.getRange(c).formulas=[[v]];};

await fs.writeFile(path.join(dir,'uniform_before_inspect_20261024.ndjson'),
 (await wb.inspect({kind:'table',range:'核心口径_历史参考!M5:O9',include:'values,formulas',tableMaxRows:5,tableMaxCols:3,maxChars:2500})).ndjson);

inputs.getRange('K5:M5').values=[['两模型共同输入计数','统一计数基准','统一计数说明']];
inputs.getRange('K5:M5').format={fill:ink,font:{name:'Microsoft YaHei',size:10,color:'#FFFFFF',bold:true},wrapText:true,rowHeight:40,verticalAlignment:'center'};
inputs.getRange('K6:M23').format.font={name:'Microsoft YaHei',size:10,color:'#222222'};
inputs.getRange('K6:M23').format.verticalAlignment='center';
inputs.getRange('K6:K23').format.fill=orange;
inputs.getRange('K6:K23').setNumberFormat('#,##0.00');
inputs.getRange('K1:K23').format.columnWidth=23;
inputs.getRange('L1:L23').format.columnWidth=31;
inputs.getRange('M1:M23').format.columnWidth=65;
inputs.getRange('L6:M23').format.wrapText=true;
inputs.getRange('L6:M23').format.fill=blue;

const rows=[];
source.rows.forEach((r,i)=>{
 const o=6+2*i,b=o+1;
 const q=r.qwen_tokens;
 const count=typeof q.native_mean==='number'?q.native_mean*r.workload:null;
 const unit=count===null?'完整输入token计数缺失':(q.native_unit==='LLM输入序列token'?'Qwen LLM输入序列参考':'Qwen视觉patch输入参考');
 formula(inputs,`K${o}`,`=IF(ISNUMBER(H${b}),H${b},"未保存")`);
 formula(inputs,`K${b}`,`=IF(ISNUMBER(K${o}),K${o},"未保存")`);
 value(inputs,`L${o}`,unit);formula(inputs,`L${b}`,`=L${o}`);
 value(inputs,`M${o}`,count===null?'基线200条缺少输入长度，不以最大长度或Ours单个样例32个token代替。':'采用该批Qwen原始输入计数，按匹配工作量折算，作为双方共同分子；不是Ours内部特征槽位数。');
 formula(inputs,`M${b}`,`=M${o}`);
 for(const sheetName of ['核心口径_历史参考','含GPU前端_待补测']){
  const s=wb.worksheets.getItem(sheetName);
  value(s,'M5','统一参考输入Tokens');value(s,'N5','共同计数基准');value(s,'O5','统一输入Tokens/s');
  value(s,'A3',sheetName==='核心口径_历史参考'?'2026-10-24：同任务双方采用相同的参考输入计数。时间、能耗、算量未改；旧版本/边界限制仍保留。':'2026-10-24：统一参考输入计数；GPU前端计时缺测仍待补，不以统一计数代替推理测量。');
  for(const n of [o,b]){
   formula(s,`M${n}`,`=IF(ISNUMBER('输入依据与Bridge'!K${o}),'输入依据与Bridge'!K${o},"未保存")`);
   formula(s,`N${n}`,`='输入依据与Bridge'!L${o}`);
   formula(s,`O${n}`,`=IF(AND(ISNUMBER(M${n}),ISNUMBER(H${n})),M${n}/H${n}*1000,"缺计时或计数")`);
  }
  value(s,'G26','统一Tokens/s倍数');value(s,'H26','与时间倍率核对');
  s.getRange('G26:H26').format={fill:ink,font:{name:'Microsoft YaHei',size:10,bold:true,color:'#FFFFFF'},wrapText:true,verticalAlignment:'center',horizontalAlignment:'center'};
  s.getRange('A26:H26').format.rowHeight=42;
  const n=27+i;
  formula(s,`G${n}`,`=IF(AND(ISNUMBER(C${n}),ISNUMBER(O${o}),ISNUMBER(O${b})),O${o}/O${b},IF(ISNUMBER(C${n}),"缺计数",C${n}))`);
  formula(s,`H${n}`,`=IF(AND(ISNUMBER(C${n}),ISNUMBER(G${n})),IF(ABS(C${n}-G${n})<0.00000001,"一致","需核查"),"待补或范围不匹配")`);
  s.getRange(`G${n}:H${n}`).format={fill:orange,font:{name:'Microsoft YaHei',size:10,color:'#222222'},wrapText:true,verticalAlignment:'center'};
  s.getRange(`G${n}`).setNumberFormat('0.000');
  s.getRange('M6:M23').setNumberFormat('#,##0.00');s.getRange('O6:O23').setNumberFormat('#,##0.0');
 }
 rows.push({id:r.id,task:r.task,workload:r.workload,common_input_tokens:count,common_unit:unit,
  ours_native_input_reference:r.ours_input_count_reference,baseline_native_input_mean:q.native_mean,
  ours_total_reference_ms:r.total_reference_ms,baseline_total_reference_ms:r.baseline_workload_ms,
  ours_uniform_input_tokens_per_second:count===null?null:count/(r.total_reference_ms/1000),
  baseline_uniform_input_tokens_per_second:count===null?null:count/(r.baseline_workload_ms/1000),
  latency_speedup_reference:r.speedup_reference,
  uniform_tokens_rate_ratio:count===null||r.id==='04'?null:r.baseline_workload_ms/r.total_reference_ms,
  comparable_scope:r.id!=='04',source:q.source??null,source_sha256:q.sha256??null,
  status:r.id==='04'?'旧Ours六次通用路径不匹配当前在线query；数学比值不作为正式任务加速':count===null?'缺共同输入计数':'统一输入工作量归一化，未改模型或计时'});
});
value(inputs,'A71','统一输入Tokens/s=N共同输入÷对应推理秒数。同任务双方使用相同分子，故其吞吐倍数等于时间加速倍数。');
value(inputs,'A72','统一参考计数不表示两模型内部序列、token维度或计算量相同；H:I仍保留原生计数作追溯，主表改用K:L。');
value(inputs,'A73','优先采用同批Qwen LLM输入长度；LLM长度未存时明确用视觉patch参考，不能把后者称为语言token；不跨任务比较绝对token速率。');
value(inputs,'A81','本次仅统一统计分子，未改时间、能耗、TOP、TOPS/W、性能、模型输入或结构；非新增GPU测速。');
wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#SPILL!',options:{useRegex:true,maxResults:20},summary:'uniform token formula scan'});
await fs.writeFile(path.join(dir,'uniform_formula_check_20261024.ndjson'),errors.ndjson);
const inspect=await wb.inspect({kind:'table',range:'核心口径_历史参考!M6:O9',include:'values,formulas',tableMaxRows:4,tableMaxCols:3,maxChars:3000});
await fs.writeFile(path.join(dir,'uniform_after_inspect_20261024.ndjson'),inspect.ndjson);
await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(dir,newName));

const metaName='uniform_input_tokens_20261024.json',csvName='uniform_input_tokens_20261024.csv';
const meta={date:'2026-10-24',definition:'共同参考输入token计数 / 同工作量推理秒数；不是更改真实模型内部序列，也不是自回归生成速度',input_workbook:oldName,input_workbook_sha256:digest(await fs.readFile(path.join(dir,oldName))),audit_sha256:digest(await fs.readFile(path.join(dir,auditName))),rows};
await fs.writeFile(path.join(dir,metaName),JSON.stringify(meta,null,2)+'\n');
const columns=Object.keys(rows[0]),quote=v=>v===null?'':`"${String(v).replaceAll('"','""')}"`;
await fs.writeFile(path.join(dir,csvName),'\uFEFF'+[columns.join(','),...rows.map(r=>columns.map(c=>quote(r[c])).join(','))].join('\r\n')+'\r\n');

// Rendering separate imports avoids sharing renderer text caches between sheets.
for(const [sheetName,range,suffix] of [['核心口径_历史参考','M2:Q23','core_tokens'],['含GPU前端_待补测','M2:Q23','expanded_tokens'],['核心口径_历史参考','A26:H35','comparison'],['输入依据与Bridge','K5:M23','inputs']]){
 const renderBook=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(dir,newName)));
 const preview=await renderBook.render({sheetName,range,scale:1,format:'png'});
 await fs.writeFile(path.join(dir,`uniform_${suffix}_20261024.png`),new Uint8Array(await preview.arrayBuffer()));
}
const files=[newName,metaName,csvName];
const manifest={date:'2026-10-24',code_commit:process.env.REPORT_CODE_COMMIT??'see source hashes',artifacts:[],source_files:[]};
for(const f of files){const bytes=await fs.readFile(path.join(dir,f));manifest.artifacts.push({file:f,bytes:bytes.length,sha256:digest(bytes)});}
for(const f of ['update_uniform_tokens.mjs','../test_uniform_input_tokens.py','../README.md']){
 const absolute=path.resolve(dir,f);manifest.source_files.push({file:f,sha256:digest(await fs.readFile(absolute))});
}
await fs.writeFile(path.join(dir,'delivery_manifest_20261024.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({output:path.join(dir,newName),formula_check:errors.ndjson,temporal:rows[0]}));
