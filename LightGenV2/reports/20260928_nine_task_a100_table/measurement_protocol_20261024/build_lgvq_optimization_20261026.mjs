import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const output=process.argv[2];if(!output)throw Error('New output directory required');
const data=JSON.parse(await fs.readFile(path.join(output,'independent_results.json'),'utf8'));
if(data.status!=='passed')throw Error('Independent review required');
const file=path.join(output,'A100_LG指令图与诊断同步优化_三轮2048次_20261026.xlsx');
try{await fs.access(file);throw Error('Preserve existing workbook')}catch(e){if(e.code!=='ENOENT')throw e}
const wb=Workbook.create(),summary=wb.worksheets.add('16视频结果与口径'),raw=wb.worksheets.add('逐轮原值与前缀');
const kinds=['hoisted','graph',...(data.all_points.some(p=>p.kind==='split')?['split']:[])];
const labels={hoisted:'输入适配器前移',graph:'诊断延后＋完整指令图',split:'诊断延后＋双缓冲前后端图'};
const mean=k=>data.all_points.filter(p=>p.kind===k).reduce((a,p)=>a+p.optimized_GPU_total_ms/128,0)/3;
const selected=kinds.reduce((a,b)=>mean(a)<mean(b)?a:b),selectedRow=12+kinds.indexOf(selected);
const qHeader=5+data.all_points.length+2,qFirst=qHeader+1,qLast=qHeader+3,prefixHeader=qLast+3;
const block=(s,row,v)=>s.getRangeByIndexes(row-1,0,v.length,v[0].length).values=v;
const head=(s,r)=>s.getRange(r).format={fill:'#243B53',font:{name:'Arial',size:11,color:'#FFFFFF',bold:true},rowHeight:40,wrapText:true,verticalAlignment:'center'};
const note=(s,row,text)=>{s.getRange(`A${row}:H${row}`).merge();s.getRange(`A${row}`).values=[[text]];s.getRange(`A${row}:H${row}`).format.wrapText=true;s.getRange(`A${row}:H${row}`).format.rowHeight=36;};
for(const s of [summary,raw]){s.showGridLines=false;s.getRange('A1:O100').format.font={name:'Arial',size:11,color:'#182B3A'};s.getRange('A1:O100').format.rowHeight=25;s.getRange('A1:O100').format.verticalAlignment='center';s.getRange('A:A').format.columnWidth=29;s.getRange('B:H').format.columnWidth=19;}
note(summary,2,'LG Temporal：16视频 / 64帧，新的实现优化');summary.getRange('A2').format.font={size:17,bold:true};
note(summary,3,'每版三个独立加载进程，每轮2048次输入＝512个原TEST视频循环4次；Qwen不改。');
block(summary,5,[['对照口径','Qwen50 / ms每16','LG新版 / ms每16','Qwen÷LG','是否文件端到端','解释','','']]);head(summary,'A5:H5');
block(summary,6,[['Qwen完整串行窗口',null,null,null,'否','按指定策略保留Qwen视频间准备；LG图像原输入驻留','',''],['Qwen GPU模型窗口之和',null,null,null,'否','两边均含GPU模型前端；不含Qwen视频间CPU准备','','']]);
summary.getRange('B6').formulas=[[`=AVERAGE('逐轮原值与前缀'!F${qFirst}:F${qLast})`]];summary.getRange('B7').values=[[data.Qwen_GPU_model_window_sum16_ms]];
for(const r of [6,7]){summary.getRange(`C${r}`).formulas=[[`=C${selectedRow}`]];summary.getRange(`D${r}`).formulas=[[`=B${r}/C${r}`]];}
summary.getRange('B6:C7').setNumberFormat('0.000');summary.getRange('D6:D7').setNumberFormat('0.00"×"');summary.getRange('E6:H7').format.wrapText=true;summary.getRange('A6:H7').format.rowHeight=62;summary.getRange('F6:H6').merge();summary.getRange('F7:H7').merge();
summary.getRange('D6:E7').format.horizontalAlignment='center';
note(summary,9,'重要：共卡、不同时间段测量；完整串行窗口的倍率不是对称视频文件端到端或独占硬件加速结论。');summary.getRange('A9:H9').format.fill='#FFF0DD';
block(summary,11,[['LG配对实验','原版 / ms每16','候选 / ms每16','原版÷候选','候选等效帧/ms','三轮全部保留','','']]);head(summary,'A11:H11');
kinds.forEach((k,i)=>{const r=12+i,start=5+3*i,end=start+2;block(summary,r,[[labels[k],null,null,null,null,k==='hoisted'?'实际CPU分词；前移本身未证明收益':k==='graph'?'首48视频＋完整图建图＋后2000视频全保留':'首48视频＋四图建图＋双缓冲前后端全保留','','']]);summary.getRange(`B${r}`).formulas=[[`=AVERAGE('逐轮原值与前缀'!F${start}:F${end})`]];summary.getRange(`C${r}`).formulas=[[`=AVERAGE('逐轮原值与前缀'!G${start}:G${end})`]];summary.getRange(`D${r}`).formulas=[[`=B${r}/C${r}`]];summary.getRange(`E${r}`).formulas=[[`=C${r}/64`]];summary.getRange(`F${r}:H${r}`).merge();});
summary.getRange('B12:E14').setNumberFormat('0.000');summary.getRange('D12:D14').setNumberFormat('0.00"×"');summary.getRange('F12:H14').format.wrapText=true;summary.getRange('A12:H14').format.rowHeight=45;
note(summary,15,'LG主起点：第一组图像卷积/文字embedding的GPU事件；终点：最后一组16个评分在GPU就绪。');
note(summary,16,'新版实际CPU分词每组都执行；含首分词的“主机调度至末评分”另列，不将分词再重复相加。');
note(summary,17,'视频解码、裁剪、图像整理和原始输入传输在LG主窗外；这些没有因为本优化就自动被覆盖。');
note(summary,18,'完整软件GPU执行含快速傅里叶变换光路模拟、原残差、融合映射和任务头，不冒称物理光路实测。');
note(summary,19,`上方采用三轮整体均值最低的方案：${labels[selected]}；不是最快单轮。MPS50%指Qwen实际使用54个计算单元。`);
block(summary,20,[['实际改了什么','原实现','新实现','','','','','']]);head(summary,'A20:H20');
const defs=[
 ['GPU指令提交','每组由Python反复发起许多小操作','复用录下的GPU指令图；每组仍重新输入、重新计算'],
 ['诊断CPU读回','四融合层每组16个诊断量立即读回，强制等待','GPU统计量全部保留；评分后集中读回，128组全部保存'],
 ['任务评分输出','完整16视频相干场、六次传播、原评分','没有删层/改精度/改权重/缓存神经特征；完整输出逐位同'],
 ['建图与首批','无额外dummy预热','首3组48视频正常执行、结果保留；建图开销也留在总窗'],
 ['tokenizer / 分词','文字变成整数编号，在CPU上执行','新版每组实际执行，不是embedding；结果和原输入完全同'],
 ['embedding / 词向量查表','GPU根据文字编号取得向量','每组重新执行；图像卷积embedding、位置、池化也保留'],
 ['Qwen 50%','每次一个视频：准备→传输→完整模型→CPU返回','完全不改、不加预取、不并行；精确复用已封存三轮'],
 ['固定光周期','另有固定参数估算，不是本表软件计时','本表不拿固定光周期替换FFT，也不把两者相加']
];
defs.forEach((v,i)=>{const r=21+i;block(summary,r,[[...v,'','','','','']]);summary.getRange(`C${r}:H${r}`).merge();summary.getRange(`A${r}:H${r}`).format.wrapText=true;summary.getRange(`A${r}:H${r}`).format.rowHeight=58;});
note(summary,30,'等效单帧＝16视频整场时间÷64；Qwen也统一除64。两边同时换单帧不会改变加速倍数。');
note(summary,31,'各实现均按同新进程配对、三个独立进程全部保留；不把最快LG轮和最慢Qwen轮搭配。');
note(summary,32,'两次早期建图捕获失败已保全，不计合格成本：01常量分配不支持；02诊断float(Tensor)同步不支持。');
note(summary,33,'“稳态”只是重复输入吞吐观察，不能把512个独立视频循环4次说成2048个不同视频。');
block(raw,4,[['LG实验','轮次','视频输入次数','对照累计/ms','候选累计/ms','对照每16/ms','候选每16/ms','候选等效帧/ms','候选主机累计/ms','建图开销/ms']]);head(raw,'A4:J4');
data.all_points.forEach((p,i)=>{const r=5+i;block(raw,r,[[labels[p.kind],p.repeat,p.video_calls,p.control_GPU_total_ms,p.optimized_GPU_total_ms,null,null,null,p.optimized_host_total_ms,p.capture?.host_ms??'不适用']]);raw.getRange(`F${r}`).formulas=[[`=D${r}*16/C${r}`]];raw.getRange(`G${r}`).formulas=[[`=E${r}*16/C${r}`]];raw.getRange(`H${r}`).formulas=[[`=G${r}/64`]];raw.getRange(`K${r}`).values=[[p.original_path]];const source=data.source_files.find(f=>f.path===p.original_path);if(!source)throw Error('Missing raw SHA');raw.getRange(`N${r}:O${r}`).values=[[source.bytes,source.sha256]];});
raw.getRange('K4').values=[['原始LG时间记录绝对路径']];raw.getRange('N4:O4').values=[['原文件字节','原文件SHA256']];head(raw,'K4:O4');
block(raw,qHeader,[['Qwen三轮严格串行','轮次','视频输入次数','累计连续/ms','模型窗口累计/ms','每16连续/ms','每16模型/ms','等效帧连续/ms']]);head(raw,`A${qHeader}:H${qHeader}`);
data.Qwen_retained.forEach((p,i)=>{const r=qFirst+i;block(raw,r,[["Qwen50严格串行",p.repeat,p.video_calls,p.elapsed_ms,p.GPU_model_window_sum16_ms*128,null,null,null]]);raw.getRange(`F${r}`).formulas=[[`=D${r}*16/C${r}`]];raw.getRange(`G${r}`).formulas=[[`=E${r}*16/C${r}`]];raw.getRange(`H${r}`).formulas=[[`=F${r}/64`]];raw.getRange(`K${r}`).values=[[p.original_path]];});
block(raw,prefixHeader,[['图优化前缀（包含首批/建图）','轮次','前缀视频次数','每16视频/ms','等效帧/ms']]);head(raw,`A${prefixHeader}:E${prefixHeader}`);
let pr=prefixHeader+1;for(const p of data.all_points.filter(p=>p.kind!=='hoisted'))for(const pre of p.prefixes){block(raw,pr,[[labels[p.kind],p.repeat,pre.video_calls,pre.time16_ms,null]]);raw.getRange(`E${pr}`).formulas=[[`=D${pr}/64`]];pr++;}
raw.getRange(`D5:J${pr}`).setNumberFormat('0.000');raw.getRange('J5:J7').setNumberFormat('General');raw.getRange('K:K').format.columnWidth=95;raw.getRange('L:L').format.columnWidth=3;raw.getRange('M:M').format.columnWidth=3;raw.getRange('N:N').format.columnWidth=18;raw.getRange('O:O').format.columnWidth=72;raw.getRange('I:J').format.columnWidth=23;raw.freezePanes.freezeRows(4);raw.getRange(`A5:A${4+data.all_points.length}`).format.wrapText=true;raw.getRange(`A5:J${4+data.all_points.length}`).format.rowHeight=40;
summary.freezePanes.freezeRows(5);summary.getRange('A:A').format.columnWidth=30;summary.getRange('B:B').format.columnWidth=26;summary.getRange('C:H').format.columnWidth=18;
wb.recalculate();const original=raw.getRange('E8').values[0][0],old=summary.getRange('C13').values[0][0];raw.getRange('E8').values=[[original+128]];wb.recalculate();if(Math.abs(summary.getRange('C13').values[0][0]-old-1/3)>1e-5)throw Error('Formula did not recalculate');raw.getRange('E8').values=[[original]];wb.recalculate();
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:2000})).ndjson);
for(const [s,range,name] of [[summary,'A1:H19','results'],[summary,'A20:H33','explanation'],[raw,`A1:J${qLast+1}`,'rounds']]){const b=await wb.render({sheetName:s.name,range,scale:1,format:'png'});await fs.writeFile(path.join(output,`${name}.png`),new Uint8Array(await b.arrayBuffer()));}
await (await SpreadsheetFile.exportXlsx(wb)).save(file);await fs.writeFile(path.join(output,'formula_review.json'),JSON.stringify({input_change_recalculation_checked:true,restored:true,selected_kind:selected,comparison:summary.getRange('A5:H7').values,paired:summary.getRange(`A11:H${11+kinds.length}`).values},null,2));console.log(file);
