import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const dir = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1'));
const output = dir;
const data = JSON.parse(await fs.readFile(path.join(output, 'frontend_tokens_audit_20261023.json'), 'utf8'));
const wb = Workbook.create();
const core = wb.worksheets.add('核心口径_历史参考');
const expanded = wb.worksheets.add('含GPU前端_待补测');
const inputs = wb.worksheets.add('输入依据与Bridge');
const paired = wb.worksheets.add('LGVQ同批边界');
const ink='#17365D', blue='#DDEBF7', orange='#FCE4D6', green='#E2F0D9', amber='#FFF2CC';

function base(sheet, title, headers, widths, rows=40) {
  sheet.showGridLines=false;
  sheet.tabColor=ink;
  sheet.getRangeByIndexes(0,0,rows,headers.length).format.font={name:'Microsoft YaHei',size:10,color:'#222222'};
  sheet.getRange('A2').values=[[title]];
  sheet.getRange('A2').format.font={name:'Microsoft YaHei',size:14,bold:true,color:ink};
  sheet.getRangeByIndexes(4,0,1,headers.length).values=[headers];
  sheet.getRangeByIndexes(4,0,1,headers.length).format={fill:ink,font:{bold:true,color:'#FFFFFF'},wrapText:true,verticalAlignment:'center',horizontalAlignment:'center',rowHeight:40};
  widths.forEach((width,col)=>sheet.getRangeByIndexes(0,col,rows,1).format.columnWidth=width);
  sheet.getRangeByIndexes(5,0,rows-5,headers.length).format.rowHeight=34;
  sheet.getRangeByIndexes(5,0,rows-5,headers.length).format.verticalAlignment='center';
  sheet.freezePanes.freezeRows(5);
}
function value(sheet, cell, v) { sheet.getRange(cell).values=[[v]]; }
function formula(sheet, cell, v) { sheet.getRange(cell).formulas=[[v]]; }
function band(sheet, range) { sheet.getRange(range).format={fill:ink,font:{bold:true,color:'#FFFFFF'},rowHeight:32,wrapText:true}; }

base(inputs,'输入、补测字段及可追溯依据',
 ['序号','模型','旧电子计时/ms','新增前端/ms','漏项NN补计/ms','扩展版完整TOP','旧TOP参考','工作量输入计数','计数单位','原始来源、版本与计数范围'],
 [8,23,19,30,40,20,20,21,30,110],83);
value(inputs,'A3','蓝色：已有测量/计数；绿色：统一器件参数；橙色：公式；黄色空格：尚未测得，绝不是零。原始文件不改写。');
data.rows.forEach((r,i)=>{
 const n=6+i*2;
 inputs.getRange(`A${n}:J${n+1}`).values=[
  [r.id+'\u00A0','LightGen2',r.electronic_reference_ms,null,null,null,r.ours_top_reference,r.ours_input_count_reference,r.ours_input_unit,`${r.component_source}；${r.ours_input_count_basis}；${r.operation_basis}`],
  [r.id+'\u00A0','Qwen3VL',r.baseline_workload_ms,null,0,null,r.baseline_top_reference,r.qwen_tokens.native_mean===null?null:r.qwen_tokens.native_mean*r.workload,r.qwen_tokens.native_unit,`${r.qwen_tokens.source||'formal_08_fullaudit：token长度未存'}；${r.qwen_tokens.sha256||''}；${r.operation_basis}`]
 ];
});
inputs.getRange('C6:C23').format.fill=blue;
inputs.getRange('G6:I23').format.fill=blue;
inputs.getRange('D6:F23').format.fill=amber;
inputs.getRange('C6:G23').setNumberFormat('0.000000');
inputs.getRange('H6:H23').setNumberFormat('0.00');
inputs.getRange('J6:J23').format.wrapText=true;
inputs.getRange('A6:J23').format.rowHeight=72;
inputs.getRange('A27:D27').values=[['参数','数值','单位','含义']];band(inputs,'A27:D27');
inputs.getRange('A28:D31').values=[
 ['光时延',data.per_pass_ms,'ms','单次传播=0.714+0.300+0.0307'],
 ['光段',data.powers.ours_optical_stage_w,'W','80.358器件+41.388控制机箱'],
 ['电段',data.powers.ours_electronic_stage_w,'W','Ours:41.388控制机箱+62.842 A100'],
 ['基箱',data.powers.baseline_chassis_w,'W','基线机箱基底，另加逐任务A100板卡均值，不扣空载']
];
inputs.getRange('B28:B31').format.fill=green;
inputs.getRange('A34:F34').values=[['序号','任务','基线板卡功率/W','旧光传播数','扩展版光传播数','匹配任务数']];band(inputs,'A34:F34');
data.rows.forEach((r,i)=>inputs.getRange(`A${35+i}:F${35+i}`).values=[[r.id+'\u00A0',r.task,r.baseline_board_w,r.passes,r.expanded_passes,r.workload]]);
inputs.getRange('C35:C43').format.fill=blue;
inputs.getRange('D35:F43').format.fill=green;
inputs.getRange('C35:C43').setNumberFormat('0.000000');
inputs.getRange('A46:E46').values=[['序号','任务','旧Bridge记录/ms','审计状态','实际边界与缺项']];band(inputs,'A46:E46');
data.rows.forEach((r,i)=>{
 inputs.getRange(`A${47+i}:E${47+i}`).values=[[r.id+'\u00A0',r.task,r.bridge_ms,r.bridge_status,r.bridge_explanation]];
 inputs.getRange(`D${47+i}:E${47+i}`).format.wrapText=true;
 if(['02','03','04'].includes(r.id))inputs.getRange(`C${47+i}:E${47+i}`).format.fill=amber;
});
inputs.getRange('C47:C55').setNumberFormat('0.000000');
inputs.getRange('D47:D55').format.columnWidth=30;
inputs.getRange('E47:E55').format.columnWidth=40;
inputs.getRange('A47:J55').format.rowHeight=70;
inputs.getRange('A59:J59').values=[['序号','任务','视觉token合计','LLM token合计','总计时/ms','视觉patch/s','LLM输入token/s','调用数','完整LLM缺失时不替代','来源SHA256']];band(inputs,'A59:J59');
data.rows.forEach((r,i)=>{
 const n=60+i,t=r.qwen_tokens;
 inputs.getRange(`A${n}:J${n}`).values=[[r.id+'\u00A0',r.task,t.visual_total??null,t.language_total??null,t.wall_total_ms??null,null,null,t.calls??null,t.status,t.sha256??null]];
 formula(inputs,`F${n}`,`=IF(AND(ISNUMBER(C${n}),ISNUMBER(E${n})),C${n}/E${n}*1000,"未保存")`);
 formula(inputs,`G${n}`,`=IF(AND(ISNUMBER(D${n}),ISNUMBER(E${n})),D${n}/E${n}*1000,"未保存")`);
});
inputs.getRange('C60:E68').format.fill=blue;
inputs.getRange('F60:G68').format.fill=orange;
inputs.getRange('C60:G68').setNumberFormat('#,##0.00');
inputs.getRange('I60:J68').format.wrapText=true;
inputs.getRange('A60:J68').format.rowHeight=65;
const notes=[
 'Tokens/s=累计输入token数÷累计对应推理秒数，不是逐样本速率的算术平均。',
 '序列槽位/视觉patch/Ours特征槽位粒度不同，不能用原生Tokens/s倍率冒充任务加速比；这里不是生成文本速度。',
 'LLM长度缺失时仅列已有视觉patch吞吐，不拿instruction子序列、最大长度或Ours长度代替完整LLM输入。',
 '扩展版：输入已驻GPU后计patch/position、text embedding及必要神经adapter/编码/merger；只补原来未计部分。',
 'Qwen保留synchronized wall；Ours保留CUDA Event。CPU tokenizer、解码及H2D不是这里新增的GPU前端。',
 'TOPS=TOP÷秒；TOPS/W=TOP÷J。旧TOP是MAC主项/光等效参考，非当前全部PT的完整算量，不包含所有逐元素算子。',
 'Embedding查表不是Linear/Conv MAC，但仍有显存访问、时间和能耗；扩展版本必须同步复核算量范围。',
 '缺前端/adapter/merger时间与对应OP，扩展版暂不产生完整加速/能效数字；旧值和原始证据完整保留。',
 'T12 Qwen基线计时还包含参考图VAE encode、U-Net、VAE decode；不能描述为只有语言Transformer。',
 '0.86文搜图在线title query仅language三次；图像gallery离线编码六次。旧核心六次只保留为历史通用参考。'
];
notes.forEach((t,i)=>value(inputs,`A${71+i}`,t));

function buildVersion(sheet, extended){
 base(sheet,extended?'含GPU前端版本：分段未测齐，完整结果暂缺':'核心版本：历史局部口径与输入吞吐参考',
  ['序号','任务','模型','性能','匹配任务数','电子计时/ms','光参数/ms','总时间/ms','估算能耗/J','运算量参考/TOP',extended?'新口径TOPS':'参考TOPS',extended?'新口径TOPS/W':'参考TOPS/W','输入计数参考','输入计数单位','输入Tokens/s','任务/s','状态'],
  [8,23,14,13,14,18,16,17,17,19,16,16,18,29,19,17,75],39);
 value(sheet,'A3',extended?'未测字段保留空缺，不用旧运行的前端差拼接。04在线标题query传播数已改为3。':'不是当前全模型/新PT复测。TOPS是历史形状估算；原生token粒度不同，不计算跨架构token/s倍率。');
 data.rows.forEach((r,i)=>{
  for(let model=0;model<2;model++){
   const n=6+i*2+model,src=n,p=35+i,ours=model===0;
   sheet.getRange(`A${n}:Q${n}`).values=[[r.id+'\u00A0',r.task,ours?'LightGen2':'Qwen3VL',ours?r.sim:r.baseline_performance,null,null,null,null,null,null,null,null,null,ours?r.ours_input_unit:r.qwen_tokens.native_unit,null,null,extended?'缺前端/漏项神经网络计时与匹配算量；不是完整新测结果':(ours?(r.id==='04'?'旧六次通用参考，不代表0.86在线query':r.version_status):'旧200次首block后计时；不含GPU前端；TOP参考见来源')]];
   formula(sheet,`E${n}`,`='输入依据与Bridge'!F${p}`);
   formula(sheet,`F${n}`,extended?`=IF(AND(ISNUMBER('输入依据与Bridge'!D${src}),ISNUMBER('输入依据与Bridge'!E${src})),'输入依据与Bridge'!C${src}+'输入依据与Bridge'!D${src}+'输入依据与Bridge'!E${src},"待测")`:`='输入依据与Bridge'!C${src}`);
   formula(sheet,`G${n}`,ours?`='输入依据与Bridge'!${extended?'E':'D'}${p}*'输入依据与Bridge'!$B$28`:'=0');
   formula(sheet,`H${n}`,`=IF(ISNUMBER(F${n}),F${n}+G${n},"待测")`);
   formula(sheet,`I${n}`,ours?`=IF(ISNUMBER(F${n}),(G${n}*'输入依据与Bridge'!$B$29+F${n}*'输入依据与Bridge'!$B$30)/1000,"待测")`:`=IF(ISNUMBER(F${n}),F${n}/1000*('输入依据与Bridge'!$B$31+'输入依据与Bridge'!C${p}),"待测")`);
   formula(sheet,`J${n}`,`=IF(ISNUMBER('输入依据与Bridge'!${extended?'F':'G'}${src}),'输入依据与Bridge'!${extended?'F':'G'}${src},"未统计")`);
   formula(sheet,`K${n}`,`=IF(AND(ISNUMBER(J${n}),ISNUMBER(H${n})),J${n}/H${n}*1000,"待补算")`);
   formula(sheet,`L${n}`,`=IF(AND(ISNUMBER(J${n}),ISNUMBER(I${n})),J${n}/I${n},"待补算")`);
   formula(sheet,`M${n}`,`=IF(ISNUMBER('输入依据与Bridge'!H${src}),'输入依据与Bridge'!H${src},"未保存")`);
   formula(sheet,`O${n}`,`=IF(AND(ISNUMBER(M${n}),ISNUMBER(H${n})),M${n}/H${n}*1000,"缺计时或计数")`);
   formula(sheet,`P${n}`,`=IF(ISNUMBER(H${n}),E${n}/H${n}*1000,"待测")`);
  }
 });
 sheet.getRange('D6:D23').format.fill=amber;
 sheet.getRange('F6:F23').format.fill=extended?amber:blue;
 sheet.getRange('G6:G23').format.fill=green;
 sheet.getRange('H6:L23').format.fill=orange;
 sheet.getRange('M6:M23').format.fill=blue;
 sheet.getRange('O6:P23').format.fill=orange;
 sheet.getRange('D6:D23').setNumberFormat('0.0000');sheet.getRange('D22:D23').setNumberFormat('0.00');
 sheet.getRange('F6:L23').setNumberFormat('0.0000');sheet.getRange('J6:J23').setNumberFormat('0.000000');
 sheet.getRange('M6:M23').setNumberFormat('#,##0.00');sheet.getRange('O6:P23').setNumberFormat('#,##0.0');
 sheet.getRange('Q6:Q23').format.wrapText=true;sheet.getRange('A6:Q23').format.rowHeight=48;
 sheet.getRange('A26:F26').values=[['序号','匹配任务','速度倍数','能效倍数','Ours任务/J','解释']];band(sheet,'A26:F26');
 data.rows.forEach((r,i)=>{
  const n=27+i,o=6+2*i,b=o+1;
  sheet.getRange(`A${n}:F${n}`).values=[[r.id+'\u00A0',r.task,null,null,null,extended?'前端/漏项缺测，不能给完整倍率':'历史部分边界参考；不冒称新PT完整结果']];
  formula(sheet,`C${n}`,!extended&&r.id==='04'?'="范围不匹配"':`=IF(AND(ISNUMBER(H${o}),ISNUMBER(H${b})),H${b}/H${o},"待测")`);
  formula(sheet,`D${n}`,!extended&&r.id==='04'?'="范围不匹配"':`=IF(AND(ISNUMBER(I${o}),ISNUMBER(I${b})),I${b}/I${o},"待测")`);
  formula(sheet,`E${n}`,`=IF(ISNUMBER(I${o}),E${o}/I${o},"待测")`);
 });
 sheet.getRange('C27:E35').format.fill=orange;
 sheet.getRange('C27:E35').setNumberFormat('0.000');
 sheet.getRange('F27:F35').format.wrapText=true;
 sheet.getRange('A27:F35').format.rowHeight=48;
}
buildVersion(core,false);buildVersion(expanded,true);

base(paired,'LGVQ同一批时间戳：只改变计时边界',
 ['重复','视频数','首block至GPU结果/ms','模型入口至GPU结果/ms','前端host差/ms','首block折16视频/ms','模型入口折16视频/ms','耗时比','原始依据'],
 [14,12,26,28,22,27,29,17,100],25);
value(paired,'A3','4帧448配置，cap100；三个repeat各前200视频。前端差非独立CUDA embedding时间，也不是9月表同一次运行。');
data.paired_lgvq.forEach((r,i)=>{
 const n=6+i;
 paired.getRange(`A${n}:I${n}`).values=[[r.repeat,r.calls,r.core_wall_ms,r.prepared_input_wall_ms,null,null,null,null,`${r.source}；SHA256 ${r.sha256}`]];
 formula(paired,`E${n}`,`=D${n}-C${n}`);formula(paired,`F${n}`,`=C${n}*16`);formula(paired,`G${n}`,`=D${n}*16`);formula(paired,`H${n}`,`=D${n}/C${n}`);
});
paired.getRange('C6:D9').format.fill=blue;paired.getRange('E6:H9').format.fill=orange;
paired.getRange('C6:H9').setNumberFormat('0.000000');paired.getRange('I6:I9').format.wrapText=true;paired.getRange('A6:I9').format.rowHeight=75;
value(paired,'A12','同源结果约增加7.79%的wall时间；Ours对应前端未补齐，不能据此算完整的新加速倍数。');
value(paired,'A14','核心公式：t扩展=t核心+Δt同源前端+Δt漏计的必要神经网络；已经在核心内的merger不可重复相加。');
value(paired,'A16','Tokens/s=N输入/Σt；600条新calls未记录逐样本长度，本表不套旧446来冒称600条token计数。');
value(paired,'A18','需要补测：两模型的GPU embedding/patch-position、Ours必要adapter/NN编码与漏计merger；200样本，保留首样本、无显式预热。');

wb.recalculate();
await fs.mkdir(output,{recursive:true});
const check=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#SPILL!',options:{useRegex:true,maxResults:20},summary:'formula errors'});
await fs.writeFile(path.join(output,'formula_check.ndjson'),check.ndjson);
const inspect=await wb.inspect({kind:'table',range:'核心口径_历史参考!F6:P9',include:'values,formulas',tableMaxRows:4,tableMaxCols:11,maxChars:5000});
await fs.writeFile(path.join(output,'formula_inspect.ndjson'),inspect.ndjson);
for(const [s,range,suffix] of [[core,'A2:Q23','core'],[expanded,'A2:Q23','expanded'],[inputs,'A2:J23','inputs'],[inputs,'A27:F55','bridges'],[inputs,'A59:J68','token_counts'],[paired,'A2:I18','paired']]) {
 const image=await wb.render({sheetName:s.name,range,scale:1,format:'png'});
 await fs.writeFile(path.join(output,`preview_${suffix}.png`),new Uint8Array(await image.arrayBuffer()));
}
const dest=path.join(output,'九任务_核心与含前端_Tokens_20261023.xlsx');
await(await SpreadsheetFile.exportXlsx(wb)).save(dest);
console.log(JSON.stringify({output:dest,formulaCheck:check.ndjson}));
