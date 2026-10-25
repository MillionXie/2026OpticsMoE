/** Temporal GPU scopes versus full Qwen MPS100/50, using sealed CSVs only. */
import fs from 'node:fs/promises';
import fss from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
import {stageChinese,familyChinese} from './lg_time_chinese_notes_20261025.mjs';

const HERE=path.dirname(fileURLToPath(import.meta.url));
const CORE=process.argv.includes('--core');
const OUT=path.join(HERE,'outputs/01a061e2-6f88-7913-87f5-b632000e4c83',CORE?'temporal_optical_tf16_20261025_01':'temporal_gpu16_20261025_01');
if(fss.existsSync(path.join(OUT,'delivery_manifest.json')))throw Error('Preserve sealed report');
const PY='C:/Users/Xml12/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const data=JSON.parse(execFileSync(PY,[path.join(HERE,'temporal_gpu16_report_20261025.py'),CORE?'--core-data':'--data'],{encoding:'utf8',maxBuffer:10*1024*1024}));
if(CORE){await buildCore(data);}else{
const lg=data.points.filter(r=>r.system==='LG'),st=data.stages;
const total=k=>lg.reduce((a,r)=>a+r[k],0),sum=(pred,key)=>st.filter(pred).reduce((a,r)=>a+r[key],0);
const qtotal=cap=>data.points.filter(r=>r.system==='Qwen'&&r.cap===cap).reduce((a,r)=>a+r.original_ms,0);
const near=(a,b)=>{if(!Number.isFinite(a)||Math.abs(a-b)>Math.max(1e-6,Math.abs(b)*1e-9))throw Error(a+' != '+b);};
const w=Workbook.create(),s=w.worksheets.add('16视频对照'),raw=w.worksheets.add('原始数据与来源');
const ink='#233B53',blue='#E8F1FA',calc='#EAF3EB',fixed='#FFF1D6';
for(const sheet of [s,raw]){sheet.showGridLines=false;sheet.tabColor=ink;sheet.getRange('A1:P180').format={font:{name:'Arial',size:11,color:ink},verticalAlignment:'center',rowHeight:28};}
function section(sheet,row,headers,values,widths){
 const last=String.fromCharCode(64+headers.length);
 sheet.getRange(`A${row}:${last}${row}`).values=[headers];
 sheet.getRange(`A${row}:${last}${row}`).format={fill:ink,font:{color:'#FFFFFF',bold:true},horizontalAlignment:'center',wrapText:true,rowHeight:48};
 sheet.getRange(`A${row+1}:${last}${row+values.length}`).values=values;
 widths?.forEach((v,i)=>sheet.getRange(`${String.fromCharCode(65+i)}1:${String.fromCharCode(65+i)}180`).format.columnWidth=v);
}
const f=(sheet,cell,formula)=>sheet.getRange(cell).formulas=[[formula]];
const format=(sheet,range,number='0.000')=>{sheet.getRange(range).setNumberFormat(number);sheet.getRange(range).format.horizontalAlignment='right';};
s.getRange('A2').values=[['Temporal：16视频的LG显卡时间与完整Qwen']];s.getRange('A2').format.font={bold:true,size:16};
s.getRange('A3').values=[['统一折算16个有效视频、64帧。LG资源上限100%；对照为同一40GB A100的Qwen资源上限100%与50%。']];
section(s,5,['系统及计时范围','16视频耗时 ms','相对完整Qwen\n100%时间比','相对完整Qwen\n50%时间比','具体包含什么'],[
 ['LG全部已记录GPU阶段',null,null,null,'原显卡事件合计，含电子前端、主体、仿真传播、融合与电子补偿。只代表已记录区间，不冒称完整硬件内核轨迹。'],
 ['LG计费范围内GPU电子段',null,null,null,'包含原GPU图像前端、文字词向量、特征传输、选定主体与输出头。不含CPU准备，不计原批准排除的仿真/融合/电子补偿。'],
 ['LG GPU电子段＋固定光执行',null,null,null,'上一行加固定光执行参数。表示去掉CPU准备等后的LG硬件分段对照，不能称完整输入到结果的端到端加速。'],
 ['Qwen完整流程 MPS 100%',null,null,null,'保留CPU视频解码/输入准备、传入显存、原完整模型、任务输出、结果返回与真实间隙。'],
 ['Qwen完整流程 MPS 50%',null,null,null,'同上，资源上限50%的原三轮完整计时；50%指资源配置，不代表实测利用率恒为50%。'],
 ['LG原完整输入口径（参照）',null,null,null,'保留完整CPU输入前端、选定电子同步总耗时、CPU结果返回、真实间隙及固定光执行时间，原正式结果不改。'],
 ],[44,24,27,27,110]);
s.getRange('E6:E11').format={wrapText:true,rowHeight:66};
f(s,'B6',"=SUM('原始数据与来源'!G6:I8)/600*16");
f(s,'B7','=B22');f(s,'B8',"=B7+SUM('原始数据与来源'!O6:O8)/600*16");
f(s,'B9',"=SUM('原始数据与来源'!F9:F11)/600*16");f(s,'B10',"=SUM('原始数据与来源'!F12:F14)/600*16");f(s,'B11','=B38');
for(let row=6;row<=11;row++){f(s,'C'+row,`=$B$9/B${row}`);f(s,'D'+row,`=$B$10/B${row}`);}
format(s,'B6:B11');format(s,'C6:D11','0.00"×"');s.getRange('B6:B11').format.fill=calc;
s.getRange('A13').values=[['前3行按要求使用LG局部GPU/硬件分段对完整Qwen，计时范围不同；不是同范围全流程加速。']];
section(s,15,['LG显卡电子时间花在哪里','16有效视频\n显卡事件 ms','原始平均整场\n显卡事件 ms','16有效视频\n同步总耗时 ms','中文说明'],[
 ['图像分块特征提取',null,null,null,'原图像块嵌入。当前前端按视频四帧逐个调用。'],
 ['图像位置特征构造/插值',null,null,null,'按原生规则生成图像位置特征，保留原数据类型与返回流程。'],
 ['图像块平均汇聚',null,null,null,'把图像块平均汇聚到原网格大小。'],
 ['文字词向量查表',null,null,null,'把提示词编号转换为原数值向量，不是完整语言大模型。'],
 ['整场特征送入显存',null,null,null,'把已整理的16视频整场特征从主内存送入显卡。'],
 ['主体电子网络与最终评分',null,null,null,'原适配器、可学习映射、真实四帧合并器、读出网络等。16视频整场输出。'],
 ['LG GPU电子段合计',null,null,null,'B列为主对照采用的显卡事件时间；D列为相同阶段的同步总耗时。两列是不同计时方式，不能相加。'],
 ]);
s.getRange('E16:E22').format={wrapText:true,rowHeight:56};
const selectors=[r=>r.stage==='frontend.native_patch',r=>r.stage==='frontend.native_position',r=>r.stage==='frontend.native_mean_pool',r=>r.stage==='frontend.language_embedding',r=>r.stage==='input.original_CPU_features_H2D',r=>r.charged&&r.region==='body'&&r.event_count>0&&r.stage!=='input.original_CPU_features_H2D'];
for(let i=0;i<selectors.length;i++){const row=i+16,rows=st.map((r,k)=>selectors[i](r)?k+19:null).filter(x=>x!==null);if(!rows.length)throw Error('Missing GPU part');const eventRefs=rows.map(r=>`'原始数据与来源'!G${r}`).join(','),wallRefs=rows.map(r=>`'原始数据与来源'!H${r}`).join(',');f(s,'B'+row,`=SUM(${eventRefs})/600*16`);f(s,'C'+row,`=SUM(${eventRefs})/39`);f(s,'D'+row,`=SUM(${wallRefs})/600*16`);}
for(const col of ['B','C','D'])f(s,col+'22',`=SUM(${col}16:${col}21)`);format(s,'B16:D22');
s.getRange('A24').values=[['原始每轮13场×16视频，末场8视频补齐。对照统一按600有效视频折算16视频；C列仅供核对真实39场均值。']];
section(s,27,['LG完整时间花在哪里','16有效视频耗时 ms','占LG原完整时间','是否计入上方GPU电子段','具体说明'],[
 ['CPU视频解码',null,null,'不计入','原四帧视频读取、65%裁剪和448尺寸图像生成。'],
 ['CPU 14维质量统计',null,null,'不计入','从图像计算原14维质量统计特征。'],
 ['CPU图像输入整理',null,null,'不计入','原图像处理器把图像转成网络输入数组。'],
 ['CPU文字分词',null,null,'不计入','提示词转换为词编号和有效位置标记。'],
 ['GPU前端同步总耗时',null,null,'采用对应显卡事件','图像分块、位置、汇聚和文字词向量的同步计时。不是再加一次显卡事件。'],
 ['GPU主体及传输同步总耗时',null,null,'采用对应显卡事件','整场特征传输、原主体电子网络及评分头。'],
 ['CPU结果返回',null,null,'不计入','把16个视频的质量分数送回主内存。'],
 ['前端未单独细分部分',null,null,'不冒充GPU时间','完整前端总时间减去已有独立阶段后的实际剩余，仍保留在原完整计时。'],
 ['场间真实间隙',null,null,'不计入','前一场输出到下一场输入开始的真实空档。'],
 ['固定光执行时间',null,null,'只加到GPU＋光一行','每场6次×1.0447ms，39场全部保留；折16有效视频含补齐分摊。'],
 ['LG原完整时间合计',null,null,'对应上方原完整参照','以上部分合计，与原600视频正式成本完全一致。'],
 ]);s.getRange('D28:E38').format={wrapText:true,rowHeight:46};
for(const [row,stage] of [[28,'frontend.Qwen_decode65_448'],[29,'frontend.quality14'],[30,'frontend.processor'],[31,'frontend.UTF8_tokenizer']]){const i=st.findIndex(r=>r.stage===stage)+19;f(s,'B'+row,`='原始数据与来源'!H${i}/600*16`);}
f(s,'B32','=SUM(D16:D19)');f(s,'B33','=SUM(D20:D21)');
for(const [row,col] of [[34,'L'],[35,'M'],[36,'N'],[37,'O']])f(s,'B'+row,`=SUM('原始数据与来源'!${col}6:${col}8)/600*16`);
f(s,'B38','=SUM(B28:B37)');for(let row=28;row<=38;row++)f(s,'C'+row,`=B${row}/$B$38`);format(s,'B28:B38');format(s,'C28:C38','0.0%');
section(s,42,['Qwen完整流程的组成','MPS100%\n16视频 ms','MPS50%\n16视频 ms','是否保留在Qwen完整时间','具体说明'],[
 ['CPU读文件、解码及输入准备',null,null,'保留','原视频解码、图像整理、文字处理等；没有再剔除。'],
 ['把输入送入显卡显存',null,null,'保留','开始传输到模型调用入口的原记录区间。'],
 ['主网络之前的模型前端',null,null,'保留','图像块、位置、词向量等原前端处理。'],
 ['完整主网络与评分输出头',null,null,'保留','原完整Qwen网络及视频质量评分头。'],
 ['结果送回CPU',null,null,'保留','GPU输出完成到CPU结果返回。'],
 ['真实请求间隙',null,null,'保留','两次单视频请求之间的真实空档。'],
 ['Qwen完整时间合计',null,null,'对应上方两条Qwen行','各自三轮600次单视频调用，合计后等效16视频。'],
 ]);s.getRange('D43:E49').format={wrapText:true,rowHeight:46};
const qcols=['K','Q','R','S','L','N'];
for(let i=0;i<6;i++)for(const [dst,start,end] of [['B',9,11],['C',12,14]])f(s,dst+(43+i),`=SUM('原始数据与来源'!${qcols[i]}${start}:${qcols[i]}${end})/600*16`);
f(s,'B49','=SUM(B43:B48)');f(s,'C49','=SUM(C43:C48)');format(s,'B43:C49');
s.getRange('A52').values=[['GPU电子段包含前端，不用“完整LG减CPU”冒充显卡时间。GPU事件为已有非重叠记录区间，并非新的硬件内核分析。']];
raw.getRange('A2').values=[['三轮原始汇总与精确文件来源']];raw.getRange('A2').format.font={bold:true,size:16};
const keys=['gpu_front_event','gpu_body_event','gpu_excluded_event','gpu_selected_wall','cpu_prepare','cpu_output','unspanned_front','gaps','fixed','folder','transfer','model_front','body_head'];
section(raw,5,['系统及资源上限','轮次','有效视频数','实际场或调用数','执行视频槽数','原完整时间 ms','GPU前端事件 ms','GPU主体事件 ms','GPU排除项事件 ms','选定GPU同步 ms','CPU准备 ms','CPU输出 ms','前端未细分 ms','真实间隙 ms','固定光执行 ms','原始目录','Qwen传输 ms','Qwen模型前端 ms','Qwen主体输出头 ms'],data.points.map(r=>[`${r.system} ${r.cap}%`,r.repeat,r.valid,r.fields,r.slots,r.original_ms,...keys.map(k=>r[k]??'未单独记录')]),[25,15,18,22,21,24,24,24,24,24,24,24,24,24,24,90,24,24,24]);
// raw columns G:S follow the declared keys; source paths remain distinct text.
format(raw,'F6:O14');format(raw,'Q6:S14');
section(raw,18,['原阶段代码名','中文名称','中文分类','所在部分','LG计费是否选取','三轮实际调用数','三轮显卡事件合计 ms','三轮同步总耗时 ms','有显卡事件的次数','具体说明'],st.map(r=>[r.stage,stageChinese(r.stage)[0],familyChinese[r.family],r.region==='frontend'?'输入前端':'主体及输出',r.charged?'选取':'执行但原投影不选取',r.calls,r.event_count?r.event_ms:'无显卡事件记录',r.wall_ms,r.event_count,stageChinese(r.stage)[1]]));
raw.getRange(`A19:J${st.length+18}`).format={wrapText:true,rowHeight:52};raw.getRange('A1:A180').format.columnWidth=62;raw.getRange('B1:B180').format.columnWidth=50;raw.getRange('C1:E180').format.columnWidth=30;raw.getRange('J1:J180').format.columnWidth=96;format(raw,`G19:H${st.length+18}`);
const srcRow=st.length+23;
section(raw,srcRow,['原始文件相对仓库路径','字节数','SHA256'],Object.entries(data.source_files).map(([p,v])=>[p,v.bytes,v.sha256]));raw.getRange(`A${srcRow+1}:C${srcRow+Object.keys(data.source_files).length}`).format={wrapText:true,rowHeight:56};
raw.freezePanes.freezeRows(5);raw.freezePanes.freezeColumns(1);
w.recalculate();near(s.getRange('B7').values[0][0],(total('gpu_front_event')+total('gpu_body_event'))/600*16);near(s.getRange('B6').values[0][0],(total('gpu_front_event')+total('gpu_body_event')+total('gpu_excluded_event'))/600*16);near(s.getRange('B38').values[0][0],total('original_ms')/600*16);near(s.getRange('B49').values[0][0],qtotal(100)/600*16);near(s.getRange('C49').values[0][0],qtotal(50)/600*16);
const original=raw.getRange('G6').values[0][0],initial=s.getRange('B7').values[0][0];raw.getRange('G6').values=[[original+1]];near(s.getRange('B6').values[0][0],(total('gpu_front_event')+total('gpu_body_event')+total('gpu_excluded_event')+1)/600*16);raw.getRange('G6').values=[[original]];near(s.getRange('B7').values[0][0],initial);w.recalculate();
const testCell='G'+(st.findIndex(r=>r.stage==='frontend.native_patch')+19),stageValue=raw.getRange(testCell).values[0][0];raw.getRange(testCell).values=[[stageValue+1]];near(s.getRange('B7').values[0][0],initial+16/600);near(s.getRange('C7').values[0][0],s.getRange('B9').values[0][0]/(initial+16/600));raw.getRange(testCell).values=[[stageValue]];w.recalculate();near(s.getRange('B7').values[0][0],initial);
await fs.mkdir(OUT,{recursive:true});await fs.writeFile(path.join(OUT,'analysis_data.json'),JSON.stringify(data,null,2));
const inspect=await w.inspect({kind:'region',sheetId:s.name,range:'A5:E11',maxChars:3500});await fs.writeFile(path.join(OUT,'key_values.ndjson'),inspect.ndjson);
const errors=await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:2000});await fs.writeFile(path.join(OUT,'formula_scan.ndjson'),errors.ndjson);
for(const [sheet,range,name] of [[s,'A1:E13','comparison'],[s,'A14:E24','gpu_parts'],[s,'A26:E38','LG_whole_parts'],[s,'A41:E52','Qwen_parts'],[raw,'A17:J26','raw_stages']]){const image=await w.render({sheetName:sheet.name,range,scale:1.3,format:'png'});await fs.writeFile(path.join(OUT,name+'.png'),new Uint8Array(await image.arrayBuffer()));}
const output=await SpreadsheetFile.exportXlsx(w);await output.save(path.join(OUT,'A100_Temporal_16视频_LG显卡时间对完整Qwen_20261025.xlsx'));
console.log(JSON.stringify({output:OUT,LG_GPU_selected16_ms:s.getRange('B7').values[0][0],LG_GPU_all_recorded16_ms:s.getRange('B6').values[0][0],Qwen10016_ms:s.getRange('B9').values[0][0],Qwen5016_ms:s.getRange('B10').values[0][0]}));
}

async function buildCore(data){
 const w=Workbook.create(),s=w.worksheets.add('16视频核心对照'),raw=w.worksheets.add('原值与来源');
 const ink='#233B53',calc='#EAF3EB',fixed='#FFF1D6',blue='#E8F1FA';
 for(const sheet of [s,raw]){sheet.showGridLines=false;sheet.getRange('A1:P95').format={font:{name:'Arial',size:11,color:ink},verticalAlignment:'center',rowHeight:28};}
 s.tabColor=ink;raw.tabColor='#7893A6';
 const put=(sh,cell,value)=>sh.getRange(cell).values=[[value]];
 const f=(sh,cell,formula)=>sh.getRange(cell).formulas=[[formula]];
 const fmt=(sh,range,code='0.000')=>{sh.getRange(range).setNumberFormat(code);sh.getRange(range).format.horizontalAlignment='right';};
 const table=(sh,row,headers,rows)=>{const last=String.fromCharCode(64+headers.length);sh.getRange(`A${row}:${last}${row}`).values=[headers];sh.getRange(`A${row}:${last}${row}`).format={fill:ink,font:{color:'#FFFFFF',bold:true},wrapText:true,horizontalAlignment:'center',rowHeight:48};if(rows.length)sh.getRange(`A${row+1}:${last}${row+rows.length}`).values=rows;};
 [34,36,30,25,38,22,44].forEach((v,i)=>s.getRange(`${String.fromCharCode(65+i)}1:${String.fromCharCode(65+i)}46`).format.columnWidth=v);
 put(s,'A2','Temporal：LG首个光学模块与Qwen首个Transformer块');s.getRange('A2').format.font={size:16,bold:true};
 put(s,'A3','统一折合16个有效视频、64帧。LG每场并行16视频；Qwen仍为原单视频调用，累计到相同处理量。');
 const mainRows=[
  ['LG 资源上限100%','首个视觉光学模块\n包含光场输入映射','所选电子读出及评分段',null,null,null,'固定光参数＋原显卡事件分段'],
  ['Qwen MPS100%','首个视觉Transformer块','显卡评分完成',null,null,null,'原完整Transformer及评分网络事件'],
  ['Qwen MPS50%','首个视觉Transformer块','显卡评分完成',null,null,null,'原完整Transformer及评分网络事件'],
 ];
 table(s,5,['系统','起点','终点','16有效视频\n显卡计时 ms','等效有效\n单帧 ms','相对LG\n时间倍数','时间性质'],mainRows);
 s.getRange('A6:G8').format={wrapText:true,rowHeight:62};
 f(s,'D6','=C35');f(s,'D7',"=SUM('原值与来源'!G9:G11)/SUM('原值与来源'!D9:D11)*16");f(s,'D8',"=SUM('原值与来源'!G12:G14)/SUM('原值与来源'!D12:D14)*16");
 for(let row=6;row<=8;row++){f(s,'E'+row,`=D${row}/64`);f(s,'F'+row,`=D${row}/$D$6`);}
 fmt(s,'D6:E8');fmt(s,'F6:F8','0.00"×"');s.getRange('D6:F8').format.fill=calc;
 put(s,'A10','LG硬件分段时间＝起点后选定电子段＋六次固定光执行。不是原程序连续运行时间，也不是新光路整段实测。');
 put(s,'A11','起点前的解码、图像整理、分词、图像分块及位置、特征适配、传入显存均不计；三者也不含CPU结果复制与请求间隙。');
 put(s,'A12','这里的百倍仅对应这组核心计算范围。折成单帧后，双方同除64，时间倍数不会再乘16或64。');
 table(s,14,['系统','起点','终点','16有效视频\n同步耗时 ms','等效有效\n单帧 ms','相对LG\n时间倍数','与上表的区别'],mainRows.map(r=>[...r.slice(0,6),'同步总耗时；用于与上方显卡事件交叉核对']));
 s.getRange('A15:G17').format={wrapText:true,rowHeight:62};
 f(s,'D15','=B35');f(s,'D16',"=SUM('原值与来源'!H9:H11)/SUM('原值与来源'!D9:D11)*16");f(s,'D17',"=SUM('原值与来源'!H12:H14)/SUM('原值与来源'!D12:D14)*16");
 for(let row=15;row<=17;row++){f(s,'E'+row,`=D${row}/64`);f(s,'F'+row,`=D${row}/$D$15`);}
 fmt(s,'D15:E17');fmt(s,'F15:F17','0.00"×"');
 put(s,'A19','LG起点精确落在首个视觉光学模块的输入映射，不是更晚的第一次传播。下列模块内映射、四帧合并器和评分头都保留。');
 put(s,'A20','显卡事件与同步耗时是同阶段的两种计时方式，不能相加。仿真传播、特征融合、并行电子补偿沿用原硬件投影排除规则。');
 const names=['parallel_optics.width_to_field','parallel_optics.tokens_to_field','parallel_optics.expert_readout','parallel_optics.global_readout','frame_merger','serial_optics.width_to_field','serial_router.width_to_field','serial_router.tokens_to_field','readout'];
 const explanations=[
  '视觉光模块内的可学习输入映射，把特征宽度映射成光场宽度。包含第一次和后续同模块调用。',
  '视觉光模块内的可学习输入映射，把图像位置映射成光场结构。',
  '专家光路径结果的电子读出。不是额外再拍一次相机。',
  '全局光路径结果的电子读出。',
  '把每视频四帧的特征合并，得到后续网络所需表示。它是实际神经网络，仍计入。',
  '后半段光模块的输入映射，处理图像与提示词合并后的特征。',
  '后半段路由光模块的特征宽度映射。',
  '后半段路由光模块的特征位置映射。',
  '原评分网络，将16个视频的各自特征读出为质量分数。',
 ];
 table(s,23,['LG从此起点之后的组成','16视频\n同步耗时 ms','16视频\n显卡事件 ms','占上方LG\n显卡计时','中文说明'],names.map((n,i)=>[stageChinese(n)[0],null,null,null,explanations[i]]));
 for(let i=0;i<names.length;i++){const target=24+i,j=data.stages.findIndex(r=>r.stage===names[i]);if(j<0)throw Error('Missing post-start stage');const r=j+19;f(s,'B'+target,`='原值与来源'!F${r}/SUM('原值与来源'!D6:D8)*16`);f(s,'C'+target,`='原值与来源'!E${r}/SUM('原值与来源'!D6:D8)*16`);f(s,'D'+target,`=C${target}/$C$35`);}
 s.getRange('A24:E32').format={wrapText:true,rowHeight:78};
 put(s,'A33','起点后电子段小计');f(s,'B33','=SUM(B24:B32)');f(s,'C33','=SUM(C24:C32)');f(s,'D33','=C33/$C$35');put(s,'E33','仅计上方九项实际电子阶段；起点之前的适配器不在此范围。');
 put(s,'A34','六次固定光执行');f(s,'B34',"=SUM('原值与来源'!I6:I8)/SUM('原值与来源'!D6:D8)*16");f(s,'C34','=B34');f(s,'D34','=C34/$C$35');put(s,'E34','每实际整场6×1.0447ms。三轮39场含末场补齐，折16有效视频为6.518928ms；真实一整场仍6.2682ms。');
 put(s,'A35','LG此范围合计');f(s,'B35','=SUM(B33:B34)');f(s,'C35','=SUM(C33:C34)');f(s,'D35','=SUM(D33:D34)');put(s,'E35','本表主结果的分母。原完整输入到结果的LG时间不作改写。');
 s.getRange('A33:E35').format={wrapText:true,rowHeight:98};s.getRange('B34:C34').format.fill=fixed;s.getRange('B35:D35').format.fill=calc;fmt(s,'B24:C35');fmt(s,'D24:D35','0.0%');
 put(s,'A37','补齐处理：每轮13场执行208视频槽、200有效视频。按三轮合计÷600×16折算，不删首场、不丢补齐成本。');
 table(s,39,['同起点连续软件运行参照','LG\n16视频 ms','Qwen100%\n16视频 ms','Qwen50%\n16视频 ms','为什么与硬件分段时间不同'],[
  ['从上述起点到CPU评分返回',null,null,null,'这一行是实际原程序的连续区间。LG包含数值光传播仿真、融合、电子补偿及未独立细分的软件段，不再另加固定光执行。'],
 ]);s.getRange('A40:E40').format={wrapText:true,rowHeight:80};
 for(const [col,a,b] of [['B',6,8],['C',9,11],['D',12,14]])f(s,col+'40',`=SUM('原值与来源'!J${a}:J${b})/SUM('原值与来源'!D${a}:D${b})*16`);fmt(s,'B40:D40');
 put(s,'A42','主对照终点为显卡侧选定计算完成；上面的软件参照为了保留原连续区间，终点延至CPU评分返回。两者不混用。');
 put(s,'A43','MPS100%/50%是同一40GB A100的资源上限设置，不是实际利用率，也不是把50%结果简单乘2。');
 put(raw,'A2','原三轮记录与阶段来源');raw.getRange('A2').format.font={size:16,bold:true};
 put(raw,'A3','原输入ID、四帧选择和16视频整场形状均沿用封存记录。源代码以Git原字节核SHA，不受Windows换行符影响。');
 table(raw,5,['系统','资源上限 %','轮次','有效视频数','实际场或调用数','执行视频槽数','起点后电子\n显卡事件 ms','起点后电子\n同步耗时 ms','固定光执行 ms','到CPU评分\n连续软件 ms','结果复制到CPU ms','起点前选定\nGPU事件 ms','起点前选定\nGPU同步 ms','原投影排除\nGPU事件 ms','原投影排除\nGPU同步 ms','原始运行目录'],data.points.map(p=>[p.system,p.cap,p.repeat,p.valid,p.fields,p.slots,p.event_ms,p.wall_ms,p.fixed_ms??'不适用',p.software_to_cpu_ms,p.cpu_output_ms,p.before_selected_event_ms??'未单独记录',p.before_selected_wall_ms??'未单独记录',p.excluded_event_ms??'不适用',p.excluded_wall_ms??'不适用',p.folder]));
 [25,18,12,18,24,20,26,26,24,28,24,26,26,26,26,96].forEach((v,i)=>raw.getRange(`${String.fromCharCode(65+i)}1:${String.fromCharCode(65+i)}95`).format.columnWidth=v);fmt(raw,'G6:O14');
 table(raw,18,['起点后原阶段代码','中文名称','硬件分段是否计入','实际调用数','原显卡事件合计 ms','原同步耗时合计 ms','说明'],data.stages.map(r=>[r.stage,stageChinese(r.stage)[0],r.charged?(r.event_count?'计入电子段':'CPU复制，不计上方显卡段'):'原投影排除',r.calls,r.event_count?r.event_ms:'无显卡事件',r.wall_ms,stageChinese(r.stage)[1]]));
 raw.getRange(`A19:G${18+data.stages.length}`).format={wrapText:true,rowHeight:56};fmt(raw,`E19:F${18+data.stages.length}`);
 raw.getRange('A1:A95').format.columnWidth=66;raw.getRange('B1:B95').format.columnWidth=52;raw.getRange('C1:C95').format.columnWidth=38;raw.getRange('G19:G40').format.columnWidth=92;
 const fileRows=[...Object.entries(data.source_files).map(([p,v])=>[p,v.bytes,v.sha256,'原封存文件，字节数与SHA一致']),...data.source_code_evidence.map(v=>[v.path,v.bytes,v.sha256,'Git原字节，与封存运行源码相同'])];
 table(raw,45,['原文件路径（相对仓库）','字节数','SHA256','来源核对方式'],fileRows);raw.getRange(`A46:D${45+fileRows.length}`).format={wrapText:true,rowHeight:64};
 raw.getRange('C46:C75').format.columnWidth=72;raw.freezePanes.freezeRows(5);raw.freezePanes.freezeColumns(1);
 const near=(a,b)=>{if(typeof a!=='number'||Math.abs(a-b)>Math.max(1e-6,Math.abs(b)*1e-9))throw Error(`${a} != ${b}`);};
 const total=(system,cap,key)=>data.points.filter(p=>p.system===system&&p.cap===cap).reduce((sum,p)=>sum+p[key],0)*16/600;
 w.recalculate();near(s.getRange('D6').values[0][0],total('LG',100,'event_ms')+total('LG',100,'fixed_ms'));near(s.getRange('D15').values[0][0],total('LG',100,'wall_ms')+total('LG',100,'fixed_ms'));near(s.getRange('D7').values[0][0],total('Qwen',100,'event_ms'));near(s.getRange('D8').values[0][0],total('Qwen',50,'event_ms'));
 // Verify live formula propagation by perturbing a stage, then restore it.
 const test='E'+(19+data.stages.findIndex(r=>r.stage==='frame_merger')),original=raw.getRange(test).values[0][0],baseline=s.getRange('D6').values[0][0];raw.getRange(test).values=[[original+1]];near(s.getRange('D6').values[0][0],baseline+16/600);near(s.getRange('F7').values[0][0],s.getRange('D7').values[0][0]/(baseline+16/600));raw.getRange(test).values=[[original]];w.recalculate();near(s.getRange('D6').values[0][0],baseline);
 await fs.mkdir(OUT,{recursive:true});await fs.writeFile(path.join(OUT,'analysis_data.json'),JSON.stringify(data,null,2));
 const scan=await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:2000});await fs.writeFile(path.join(OUT,'formula_scan.ndjson'),scan.ndjson);
 const keys=await w.inspect({kind:'region',sheetId:s.name,range:'A5:G17',maxChars:4500});await fs.writeFile(path.join(OUT,'key_values.ndjson'),keys.ndjson);
 for(const [sheet,range,name] of [[s,'A1:G12','comparison'],[s,'A14:G20','synchronized'],[s,'A23:E37','LG_parts'],[s,'A39:E43','software_reference'],[raw,'A18:G26','raw_stage_sample'],[raw,'A45:D50','source_sample']]){const image=await w.render({sheetName:sheet.name,range,scale:1.2,format:'png'});await fs.writeFile(path.join(OUT,name+'.png'),new Uint8Array(await image.arrayBuffer()));}
 const output=await SpreadsheetFile.exportXlsx(w);await output.save(path.join(OUT,'A100_Temporal_16视频_LG首光模块对Qwen首TF_20261025.xlsx'));
 console.log(JSON.stringify({output:OUT,LG16_event_ms:s.getRange('D6').values[0][0],LG16_wall_ms:s.getRange('D15').values[0][0],Qwen100_event_ms:s.getRange('D7').values[0][0],Qwen50_event_ms:s.getRange('D8').values[0][0],ratios:s.getRange('F7:F8').values}));
}
