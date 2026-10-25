import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const output=process.argv[2];
if(!output)throw new Error('Pass the new reviewed output directory');
const data=JSON.parse(await fs.readFile(path.join(output,'independent_results.json'),'utf8'));
if(data.status!=='passed')throw new Error('Raw interval review required');
const file=path.join(output,'A100_LG流水线与Qwen50严格串行_2048次对照_20261026.xlsx');
try{await fs.access(file);throw new Error('Preserve existing workbook')}catch(e){if(e.code!=='ENOENT')throw e}
const wb=Workbook.create();const summary=wb.worksheets.add('对照结果'),raw=wb.worksheets.add('逐轮与来源'),nodes=wb.worksheets.add('节点与分项'),parts=wb.worksheets.add('时间花在哪里');
const label=p=>p.system==='Qwen'?'Qwen 50%原串行':p.mode==='pipeline'?'LG GPU流水线':'LG GPU顺序';
const keys=['LG GPU顺序','LG GPU流水线','Qwen 50%原串行'];
function block(sh,row,values){sh.getRangeByIndexes(row-1,0,values.length,values[0].length).values=values}
function head(sh,address){sh.getRange(address).format={fill:'#243B53',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},rowHeight:32,wrapText:true,horizontalAlignment:'center',verticalAlignment:'center'}}
for(const s of [summary,raw,nodes,parts]){s.showGridLines=false;s.getRange('A1:H80').format.font={name:'Arial',size:11,color:'#182B3A'};s.getRange('A1:H80').format.rowHeight=24;s.getRange('A1:H80').format.verticalAlignment='center';s.getRange('A:A').format.columnWidth=25;s.getRange('B:H').format.columnWidth=18;}
summary.tabColor='#243B53';summary.getRange('A2').values=[['LG流水线与Qwen逐视频严格串行']];summary.getRange('A2').format.font={size:16,bold:true};
summary.getRange('A3').values=[['每轮512个视频循环4遍；Qwen每次实际重新准备，无预取，各三轮全保留。']];
block(raw,2,[['每轮原始连续耗时与来源','','','','','','','','']]);
block(raw,4,[['执行方式','轮次','输入次数','原总毫秒','每16视频/ms','等效单帧/ms','LG后续前端重叠/ms','Qwen视频间隙/ms','稳定判据首次满足N']]);head(raw,'A4:I4');
const pointRows=[];
data.all_points.forEach((p,i)=>{const r=5+i;pointRows.push({r,p});block(raw,r,[[label(p),p.repeat,p.video_calls,p.elapsed_ms,null,null,p.system==='LG'?p.observed_later_front_body_overlap_ms:null,p.system==='Qwen'?p.inter_video_gaps16_ms*128:null,p.system==='Qwen'?'不适用：逐视频串行':p.first_qualified_steady_videos??'未观察到']]);raw.getRange(`E${r}`).formulas=[[`=D${r}*16/C${r}`]];raw.getRange(`F${r}`).formulas=[[`=E${r}/64`]];const src=data.source_files.find(f=>f.path===p.original_path);if(!src)throw new Error('Original raw SHA absent');raw.getRange(`K${r}:L${r}`).values=[[p.original_path,src.sha256]];});
raw.getRange('K4:L4').values=[['原始记录绝对路径','原始文件SHA256']];head(raw,'K4:L4');raw.getRange('K:K').format.columnWidth=85;raw.getRange('L:L').format.columnWidth=70;raw.getRange('D5:H13').setNumberFormat('0.000');raw.freezePanes.freezeRows(4);
block(summary,5,[['执行方式','三轮总输入次数','三轮总秒数','每16视频/ms','等效单帧/ms','Qwen50÷本行','稳定N：三轮']]);head(summary,'A5:G5');
keys.forEach((k,i)=>{const r=6+i;summary.getRange(`A${r}`).values=[[k]];summary.getRange(`B${r}`).formulas=[[`=SUMIFS('逐轮与来源'!$C$5:$C$13,'逐轮与来源'!$A$5:$A$13,A${r})`]];summary.getRange(`C${r}`).formulas=[[`=SUMIFS('逐轮与来源'!$D$5:$D$13,'逐轮与来源'!$A$5:$A$13,A${r})/1000`]];summary.getRange(`D${r}`).formulas=[[`=C${r}*1000*16/B${r}`]];summary.getRange(`E${r}`).formulas=[[`=D${r}/64`]];summary.getRange(`F${r}`).formulas=[[`=$D$8/D${r}`]];summary.getRange(`G${r}`).values=[[i===2?'不适用：逐视频串行':data.all_points.filter(p=>label(p)===k).map(p=>p.first_qualified_steady_videos??'未观察到').join(' / ')]];});
summary.getRange('C6:F8').setNumberFormat('0.000');summary.getRange('F6:F8').setNumberFormat('0.00"×"');summary.getRange('G:G').format.columnWidth=28;summary.getRange('A10').values=[['起点：首GPU模型前端；终点：最后评分在GPU就绪。含全部内部间隙。']];
summary.getRange('A11').values=[['完整软件包含FFT、融合和残差；不是新物理光路端到端测量。']];
block(raw,17,[['前缀连续窗口','','','','']]);block(raw,18,[['执行方式','轮次','前缀输入次数','前缀累计毫秒','每16视频/ms']]);head(raw,'A18:E18');
let pr=19;for(const p of data.all_points)for(const pre of p.prefixes){block(raw,pr,[[label(p),p.repeat,pre.video_calls,pre.time16_ms*pre.video_calls/16,null]]);raw.getRange(`E${pr}`).formulas=[[`=D${pr}*16/C${pr}`]];pr++}
raw.getRange('D19:E54').setNumberFormat('0.000');
block(summary,14,[['前缀输入次数','LG顺序/ms每16','LG流水线/ms每16','Qwen50/ms每16','Qwen50÷LG流水线']]);head(summary,'A14:E14');
[256,512,1024,2048].forEach((n,i)=>{let r=15+i;summary.getRange(`A${r}`).values=[[n]];keys.forEach((k,j)=>{let c=String.fromCharCode(66+j);summary.getRange(`${c}${r}`).formulas=[[`=SUMIFS('逐轮与来源'!$D$19:$D$54,'逐轮与来源'!$A$19:$A$54,"${k}",'逐轮与来源'!$C$19:$C$54,$A${r})*16/($A${r}*3)`]]});summary.getRange(`E${r}`).formulas=[[`=D${r}/C${r}`]];});summary.getRange('B15:E18').setNumberFormat('0.000');summary.getRange('E15:E18').setNumberFormat('0.00"×"');
block(summary,21,[['LG流水线实际重叠','','','','']]);block(summary,22,[['轮次','后续前端重叠/ms','前端区间总ms','已重叠比例','解释']]);head(summary,'A22:E22');
data.all_points.filter(p=>p.system==='LG'&&p.mode==='pipeline').forEach((p,i)=>{let r=23+i;block(summary,r,[[p.repeat,p.observed_later_front_body_overlap_ms,p.front_envelope_ms,null,'后续组前端与当前主体区间交集']]);summary.getRange(`D${r}`).formulas=[[`=B${r}/C${r}`]];});summary.getRange('B23:C25').setNumberFormat('0.000');summary.getRange('D23:D25').setNumberFormat('0.0%');summary.getRange('E23:G25').format.wrapText=true;summary.getRange('A23:G25').format.rowHeight=40;
summary.getRange('A28').values=[['稳态：连续三个8组窗口，各CV≤5%，三窗口均值max/min−1≤3%。']];summary.getRange('A29').values=[['完整2048次始终保留；未达准则则写“未观察到”，不选择最快稳定窗口。']];
summary.getRange('A30').values=[['“稳定N”是重复输入吞吐的判据，不是N个不同视频的独立泛化测试。']];
block(summary,33,[['另列：仅模型窗口之和','Qwen50/ms每16','除以LG顺序','除以LG流水线']]);head(summary,'A33:D33');
block(summary,34,[['不含视频间CPU准备',data.Qwen_GPU_model_window_sum16_ms,null,null]]);summary.getRange('C34').formulas=[['=B34/D6']];summary.getRange('D34').formulas=[['=B34/D7']];summary.getRange('B34:D34').setNumberFormat('0.000');
summary.getRange('A36').values=[['主比较按请求保留Qwen视频间CPU准备；LG的文件准备在主窗外，非对称文件端到端。']];
summary.getRange('A37').values=[['原约1030ms/16来自提前准备全部Qwen输入的旧新测方案，已弃作当前正式对照，原件保留。']];
if(data.Qwen_nonexclusive_authorized){
 summary.getRange('A11').values=[['注意：Qwen为共享GPU应急测量；LG为独占原件，倍率不作公平硬件加速结论。']];
 summary.getRange('A11:G11').format.fill='#FFF0DD';summary.getRange('A11:G11').format.font.color='#9A3412';
 summary.getRange('F5').values=[['Qwen共卡÷本行（参考）']];
}
nodes.getRange('A2').values=[['起止节点和分项含义']];nodes.getRange('A2').format.font={size:16,bold:true};
block(nodes,4,[['节点','在新GPU主窗内？','LG Temporal','Qwen 50%']]);head(nodes,'A4:D4');
const definitions=[
 ['视频文件解码与裁剪','LG窗外；Qwen中间计入','原四帧、65%裁剪及图像整理，512次预备','每一条重新执行；不预取下一条'],
 ['CPU文字分词 tokenizer','LG窗外；Qwen中间计入','文字变整数编号，不是词向量查表','每视频实际分词，不能提前覆盖'],
 ['CPU质量特征及其他整理','LG窗外；Qwen中间计入','原14维质量特征；不是可删网络层','原processor需要的其他输入，单视频执行'],
 ['CPU→GPU输入传输','LG窗外；Qwen中间计入','原始模型输入驻留，不存embedding','本视频准备好后才传输；无提前驻留输入'],
 ['首GPU模型前端','是：共同起点','16视频/64帧卷积图像embedding、位置、池化','原生B1的图像映射和实际位置操作'],
 ['GPU文本embedding','是','每组重新查表，独立流与图像前端并行','保持原架构调用顺序'],
 ['主体入口输入适配','是','原视觉/语言/质量适配仍在主体流执行','各自原架构，不额外假造同一个adapter'],
 ['第一光学模块或第一TF','是','完整六传播、原融合/残差、merger和读出','完整视觉TF、语言网络及原完整词表/质量头'],
 ['视频或组之间的等待','是','全部内部调度和等待保留','串行视频之间CPU返回、同步和调度保留'],
 ['最后GPU评分就绪','是：共同终点','最后一组16评分就绪','第2048次评分就绪'],
 ['最后结果回CPU','否，终点之后','最终数组原件保留','最终一次CPU拷贝不加尾部空等']];
block(nodes,5,definitions);nodes.getRange('A5:D15').format.wrapText=true;nodes.getRange('A5:D15').format.rowHeight=46;nodes.getRange('B:B').format.columnWidth=25;nodes.getRange('C:D').format.columnWidth=54;
block(nodes,18,[['LG详细轮分段诊断','每16视频/ms','说明']]);head(nodes,'A18:C18');
const diag=data.LG_dense_diagnostics.filter(p=>p.mode==='pipeline');const stages=[['frontend.conv_embedding','图像卷积embedding'],['frontend.image_position','图像位置相加'],['frontend.mean_pool','无参数均值池化'],['frontend.text_embedding','文本词向量查表']];
stages.forEach(([k,n],i)=>block(nodes,19+i,[[n,diag.reduce((a,p)=>a+p.stages_per16_ms[k],0)/3,'来自独立保留的详细轮，不与主窗口强行加总']]));nodes.getRange('B19:B22').setNumberFormat('0.000');
nodes.getRange('A25').values=[['光电结构：四个主阶段＋两个光学router＝六次传播；不是六个相同OEO块。']];
nodes.getRange('A26').values=[['视觉与语言之间的merger、最后电子读出都保留。']];
nodes.getRange('A27').values=[['可提前的是文本分词/embedding；依赖视觉融合结果的语言主体不能整体提前。']];
nodes.getRange('A28').values=[['实际CPU分词记录：LG准备512原输入一次；Qwen逐条准备2048次']];
block(nodes,29,[['系统','轮次','记录到的分词调用数','分词累计毫秒']]);head(nodes,'A29:D29');
data.extra_preparation.forEach((p,i)=>block(nodes,30+i,[[p.system==='lg'?'LG':'Qwen 50%',p.repeat,p.tokenizer_call_count||'未单独捕获',p.tokenizer_call_count?p.recorded_CPU_tokenizer_ms:'未核定']]));nodes.getRange('D30:D35').setNumberFormat('0.000');
const gaps=data.extra_preparation.filter(p=>p.guard_platform_annotation_gaps?.length);if(gaps.length)nodes.getRange('A38').values=[['原守卫个别末尾记录缺平台注释，邻接记录一致；原缺项未补造，详见独立审查JSON。']];
parts.getRange('A2').values=[['时间花在哪里：主窗内与主窗外分开']];parts.getRange('A2').format.font={size:16,bold:true};
block(parts,4,[['方式 / 每16视频','图像＋文字前端区间/ms','主体入口适配区间/ms','第一光学/TF到评分/ms','视频间隙/ms','连续总时间/ms']]);head(parts,'A4:F4');
for(let i=0;i<3;i++){
 const k=keys[i],pts=data.all_points.filter(p=>label(p)===k),mean=field=>pts.reduce((n,p)=>n+p[field],0)/3;
 block(parts,5+i,[[k,i<2?mean('front_envelope16_ms'):mean('model_front16_ms'),i<2?mean('input_adapters_body_envelope16_ms'):'含在原架构窗口',i<2?mean('first_optic_to_score_software_envelope16_ms'):mean('first_TF_to_score16_ms'),i<2?'含在连续总时间':mean('inter_video_gaps16_ms'),null]]);
 parts.getRange(`F${5+i}`).formulas=[[`='对照结果'!D${6+i}`]];
}
parts.getRange('B5:F7').setNumberFormat('0.000');parts.getRange('A9').values=[['LG分项区间会重叠，不能横向相加；连续总时间才是实际比较分母。']];
parts.getRange('A10').values=[['“第一光学到评分”是完整GPU软件执行，含FFT模拟/融合/残差，不是物理光传播耗时。']];
parts.getRange('A11').values=[['前端区间含每组文字embedding的依赖等待；未单独计时的操作不虚构细分值。']];
block(parts,14,[['实际CPU准备与传输','轮次','CPU准备跨度或累计/ms','CPU→GPU等窗口/ms','说明']]);head(parts,'A14:E14');
data.extra_preparation.forEach((p,i)=>block(parts,15+i,[[p.system==='lg'?'LG：512次，窗外':'Qwen：2048次',p.repeat,p.system==='lg'?p.CPU_preparation_span_ms:p.CPU_prepare_total_ms,p.system==='lg'?p.packing_transfer_tokenizer_window_sum_ms:p.H2D_total_ms,p.system==='lg'?'CPU跨度含异步间隔；右列含打包、传输、分词，非纯传输':'逐条CPU准备含分词；除第一条外均在主连续窗内']]));
parts.getRange('C15:D20').setNumberFormat('0.000');parts.getRange('E15:E20').format.wrapText=true;parts.getRange('A15:F20').format.rowHeight=48;parts.getRange('E:E').format.columnWidth=48;
parts.getRange('A23').values=[['LG循环原始GPU输入，不复用神经特征；Qwen每条重新解码/分词/传输/推理。']];
parts.getRange('A24').values=[['因此主比较是用户指定执行策略，不声称对称的视频文件端到端加速。']];
parts.getRange('B:F').format.columnWidth=24;parts.getRange('E:E').format.columnWidth=48;
wb.recalculate();
const original=raw.getRange('D5').values[0][0],old=summary.getRange('D6').values[0][0];raw.getRange('D5').values=[[original+128]];wb.recalculate();const changed=summary.getRange('D6').values[0][0];if(Math.abs(changed-old-1/3)>1e-5)throw new Error('Summary formula did not recalculate');raw.getRange('D5').values=[[original]];wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:2000});console.log(errors.ndjson);
for(const [sh,range,name] of [[summary,'A1:G37','summary'],[raw,'A1:I14','raw'],[nodes,'A1:D38','nodes'],[parts,'A1:F24','parts']]){const b=await wb.render({sheetName:sh.name,range,scale:1,format:'png'});await fs.writeFile(path.join(output,`${name}.png`),new Uint8Array(await b.arrayBuffer()));}
const x=await SpreadsheetFile.exportXlsx(wb);await x.save(file);
await fs.writeFile(path.join(output,'workbook_formula_review.json'),JSON.stringify({status:'passed',input_change_recalculation_checked:true,restored:true,summary_values:summary.getRange('A5:G8').values},null,2));
console.log(file);
