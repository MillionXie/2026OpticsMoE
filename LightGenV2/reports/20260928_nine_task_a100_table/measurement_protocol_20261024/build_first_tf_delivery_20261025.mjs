/** Two first-optics / first-Transformer workbooks, independent old inputs retained. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
import {stageChinese,familyChinese} from './lg_time_chinese_notes_20261025.mjs';
const HERE=path.dirname(fileURLToPath(import.meta.url));
const OUT=path.join(HERE,'outputs/01a061e2-6f88-7913-87f5-b632000e4c83/first_tf_complete_20261025_01');
const d=JSON.parse(await fs.readFile(path.join(OUT,'core_data.json'),'utf8'));
const TEMP_ONLY=process.argv.includes('--temporal-only'),MAIN_ONLY=process.argv.includes('--main-only');
if((!TEMP_ONLY&&d.mainRecords.some(x=>x.pending))||(!MAIN_ONLY&&d.tempRecords.some(x=>x.pending)))throw Error('Do not export incomplete supplement as complete');
try{await fs.access(path.join(OUT,'delivery_manifest.json'));throw Error('Preserve sealed delivery');}catch(e){if(e.code!=='ENOENT')throw e;}
const ink='#243A50',blue='#EAF2F9',green='#EDF5EE',amber='#FFF0D6';
const col=n=>{let s='';while(n){n--;s=String.fromCharCode(65+n%26)+s;n=Math.floor(n/26);}return s;};
function sheet(w,name,title,rows=100,cols=12){
 const s=w.worksheets.add(name);s.showGridLines=false;
 s.getRange(`A1:${col(cols)}${rows}`).format={font:{name:'Arial',size:11,color:ink},rowHeight:27,verticalAlignment:'center'};
 s.getRange('A2').values=[[title]];s.getRange('A2').format.font={bold:true,size:16,color:ink};return s;
}
function table(s,row,headers,values,widths){
 const end=col(headers.length);s.getRange(`A${row}:${end}${row}`).values=[headers];
 s.getRange(`A${row}:${end}${row}`).format={fill:ink,font:{bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:46,horizontalAlignment:'center'};
 if(values.length)s.getRange(`A${row+1}:${end}${row+values.length}`).values=values;
 widths.forEach((v,i)=>s.getRange(`${col(i+1)}1:${col(i+1)}${Math.max(100,row+values.length)}`).format.columnWidth=v);
 for(let i=0;i<values.length;i++)if(i%2===1)s.getRange(`A${row+i+1}:${end}${row+i+1}`).format.fill='#F5F7FA';
}
const form=(s,cell,f)=>{s.getRange(cell).formulas=[[f]];s.getRange(cell).format.fill=green;};
const num=(s,range,f='0.000')=>{s.getRange(range).setNumberFormat(f);s.getRange(range).format.horizontalAlignment='right';};
const title=r=>r.system==='ours'?'LG':r.system==='qwen'?'Qwen':r.system==='glm'?'GLM':r.system==='clip'?'CLIP':r.system==='yolo'?'YOLO':r.system==='deepseek'?'DeepSeek':'Kimi';
function inputs(w,records,runs){
 const s=sheet(w,'三轮原值与来源','原始三轮区间汇总',runs.length+10,13);
 table(s,5,['记录编号','模型','轮次','有效输入数','计入电子或核心秒','固定光执行秒','核心合计秒','计入的光路或显卡能量 J','匹配区间总能量 J','同批有效词编号数','区间起点','原数据路径','取值类型'],
 runs.map(r=>[r.key,title(r),r.repeat,200,r.system==='ours'?r.wall_s-r.optical_s:r.wall_s,r.optical_s,null,r.system==='ours'?r.optical_s*121.746:r.board_j,r.system_j,(records.find(x=>x.key===r.key)?.tokens??0)/3,r.system==='ours'?stageChinese(r.start_description)[0]:r.start_description,r.folder,r.new_supplement?'新补测实测':'封存原始区间重切']),[24,15,10,16,24,21,23,29,24,23,55,115,25]);
 s.getRange(`D6:J${runs.length+5}`).format.fill=blue;s.getRange(`F6:F${runs.length+5}`).format.fill=amber;
 runs.forEach((r,i)=>form(s,'G'+(i+6),`=E${i+6}+F${i+6}`));num(s,`D6:J${runs.length+5}`,'0.000000');
 num(s,`D6:D${runs.length+5}`,'#,##0');num(s,`J6:J${runs.length+5}`,'#,##0');
 s.freezePanes.freezeRows(5);return s;
}
function totals(w,records,runs){
 const s=sheet(w,'核心计算','三轮汇总和单位换算',records.length+14,13);
 table(s,5,['记录编号','模型','三轮有效输入','核心同步总秒','固定光执行总秒','计入电子总秒','匹配总能量 J','每输入同步 ms','每输入能量 J','每输入电子 ms','同批有效词编号数','共同词编号每秒','完整TEST指标'],records.map(r=>[r.key,title(r),600,null,null,null,null,null,null,null,r.tokens??null,null,r.accuracy]),[25,16,18,25,24,29,24,23,21,26,23,25,23]);
 records.forEach((r,i)=>{const n=i+6;const rr=runs.flatMap((v,j)=>v.key===r.key?[j+6]:[]);for(const [dest,source] of [['D','G'],['E','F'],['G','I']])form(s,dest+n,'='+rr.map(x=>`'三轮原值与来源'!${source}${x}`).join('+'));
 form(s,'F'+n,`=D${n}-E${n}`);
 form(s,'H'+n,`=D${n}/C${n}*1000`);form(s,'I'+n,`=G${n}/C${n}`);form(s,'J'+n,`=F${n}/C${n}*1000`);
 if(r.tokens!=null)form(s,'L'+n,`=K${n}/D${n}`);else s.getRange('L'+n).values=[['不适用']];});
 num(s,`C6:M${records.length+5}`,'0.000000');num(s,`C6:C${records.length+5}`,'#,##0');num(s,`K6:K${records.length+5}`,'#,##0');s.freezePanes.freezeRows(5);return s;
}
function LGdetail(w){
 const source=d.lgStages.filter(r=>!r.key.startsWith('40GB')&&r.charged);
 const groups=new Map();for(const r of source){const k=[r.key,r.stage,r.family,r.charged].join('|');if(!groups.has(k))groups.set(k,{...r,wall_s:0,event_s:0,calls:0});let g=groups.get(k);g.wall_s+=r.wall_s;g.event_s+=r.event_s;g.calls+=r.calls;}
 const values=[...groups.values()].map(r=>{const chinese=stageChinese(r.stage);return[d.mainRecords.find(x=>x.key===r.key).task,chinese[0]??r.stage,r.wall_s*1000/600,r.calls,chinese[1]??familyChinese[r.family],r.stage];});
 const s=sheet(w,'LG阶段中文说明','LG实际计入结果的阶段明细',values.length+8,6);
 table(s,5,['任务','中文阶段名称','计入 ms每输入','三轮执行次数','这一步具体在做什么','对应原记录名称'],values,[30,36,25,21,115,65]);
 num(s,`C6:C${values.length+5}`,'0.000000');s.freezePanes.freezeRows(5);
 s.getRange(`B6:B${values.length+5}`).format.wrapText=true;s.getRange(`E6:E${values.length+5}`).format.wrapText=true;
 s.getRange(`A6:F${values.length+5}`).format.rowHeight=48;
 s.getRange('A3').values=[['仅列实际计入阶段。Temporal每视频值乘16得到16视频64帧，除4得到等效单帧。']];return s;
}
function LGparts(w){
 const records=d.mainRecords.filter(r=>r.system==='ours'),s=sheet(w,'LG时间组成','LG核心时间到底由哪些部分组成',35,9);
 table(s,5,['任务','计时单位','光模块及后续电子 ms','空间光调制器更新 ms','曝光 ms','额外相机读出 ms','核心合计 ms'],records.map((r,i)=>[r.task,i===0?'16视频64帧':i===1?'1视频4帧':'每输入',null,null,null,null,null]),[30,23,31,25,25,28,27]);
 records.forEach((r,i)=>{const n=i+6,a=d.mainRecords.indexOf(r)+6,scale=i===0?16:1;
 form(s,'C'+n,`=('核心计算'!D${a}-'核心计算'!E${a})/'核心计算'!C${a}*1000*${scale}`);
 for(const [column,component] of [['D',.714],['E',.300],['F',.0307]])form(s,column+n,`='核心计算'!E${a}/'核心计算'!C${a}*1000*${scale}*${component}/1.0447`);
 form(s,'G'+n,`=SUM(C${n}:F${n})`);});num(s,'C6:H14','0.000000');
 s.getRange('A17').values=[['核心合计＝电子＋SLM更新＋曝光＋额外相机读出。电子已包含原任务头、检索排序和输出，不能重复相加。']];
 s.getRange('A19').values=[['SLM是空间光调制器。单周期三个固定分量依次为0.714、0.300、0.0307 ms，总计1.0447 ms。']];
 s.getRange('A21').values=[['Temporal按三轮600有效视频折16视频，保留补齐槽成本。Spatial仍是原模型1视频4帧。']];return s;
}
function notes(s,row){
 const texts=[
 '同步耗时指等待显卡确实算完再读钟的实际经过时间；表内仅保留计入时间及同一区间能量。',
 'Qwen从原生第一个Transformer块进入，到CPU收到任务结果。YOLO没有Transformer，保留首个原生卷积块起点。',
 'LG从首个光学模块的输入映射开始。保留后续原任务头、真实合并器、检索排序和CPU输出。',
 '进入起点前的读取、解码、分词、词向量、图像分块和位置准备不计。进入起点后原网络的后续分支仍照常执行并计入。',
 'LG使用已批准的硬件分段投影：选定电子实测同步耗时加固定光执行。GPU光传播仿真、布局、融合和并行补偿保留原记录但不加入投影。',
 '蓝色为原记录或新实测输入，绿色为表内推导，浅橙为固定参数。前置耗时不再用同一估计值加给各模型。',
 '本范围的完整运算量尚未独立核定，不把原全图运算量套到新时间上，也不以容量或FFT代替物理光学运算量。'];
 texts.forEach((x,i)=>s.getRange('A'+(row+i*2)).values=[[x]]);
}
async function finish(w,file,previews){
 w.recalculate();console.log((await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
 for(const [name,range] of previews){const png=await w.render({sheetName:name,range,scale:1.3});await fs.writeFile(path.join(OUT,file+'_'+name+'_'+range.replace(':','-')+'.png'),new Uint8Array(await png.arrayBuffer()));}
 const output=await SpreadsheetFile.exportXlsx(w);await output.save(path.join(OUT,file+'.xlsx'));
}
if(!TEMP_ONLY){
 const w=Workbook.create(),s=sheet(w,'九任务对照','A100九任务核心时间和能量',65,10);s.tabColor=ink;
 const records=d.mainRecords,runs=d.mainRuns;
 // Create all referenced worksheets before writing cross-sheet formulas.
 inputs(w,records,runs);totals(w,records,runs);LGparts(w);LGdetail(w);
 table(s,5,['任务','指标','LG完整TEST','Qwen完整TEST','计时单位','LG同步 ms','Qwen同步 ms','Qwen除LG倍数'],Array.from({length:9},(_,i)=>{let a=records[i*2],b=records[i*2+1];return[a.task,a.metric,a.accuracy,b.accuracy,i<2?'等效单帧':'每输入',null,null,null];}),[30,24,21,23,18,22,24,26]);
 for(let i=0;i<9;i++){let n=i+6,a=i*2+6,b=a+1,divide=i<2?4:1;for(const [dest,source,row] of [['F','H',a],['G','H',b]])form(s,dest+n,`='核心计算'!${source}${row}/${divide}`);form(s,'H'+n,`=G${n}/F${n}`);}
 num(s,'C6:D14','0.0000');num(s,'F6:J14','0.000');num(s,'H6:H14','0.00"×"');
 table(s,18,['任务','LG能量 J\n同上计时单位','Qwen能量 J\n同上计时单位','Qwen除LG能耗比','节能率','能量时延积改善','LG共同词编号每秒','Qwen共同词编号每秒'],Array.from({length:9},(_,i)=>[records[i*2].task,null,null,null,null,null,null,null]),[30,24,24,27,19,27,30,30]);
 for(let i=0;i<9;i++){let n=i+19,a=i*2+6,b=a+1,div=i<2?4:1;form(s,'B'+n,`='核心计算'!I${a}/${div}`);form(s,'C'+n,`='核心计算'!I${b}/${div}`);form(s,'D'+n,`=C${n}/B${n}`);form(s,'E'+n,`=1-B${n}/C${n}`);form(s,'F'+n,`=D${n}*H${i+6}`);form(s,'G'+n,`='核心计算'!L${a}`);form(s,'H'+n,`='核心计算'!L${b}`);}
 num(s,'B19:H27');num(s,'E19:E27','0.0%');num(s,'D19:D27','0.00"×"');num(s,'F19:F27','0.00"×"');
 notes(s,31);s.getRange('A3').values=[['LG从首个光学模块开始，Qwen从首个Transformer块开始。两类视频任务均折算等效单帧，其余按一次输入。']];
 s.getRange('A29').values=[['能量按上表同一单位展示：LGVQ两项为J/等效单帧，其余为J/输入。']];
 await finish(w,'A100_九任务_首计算模块核心对照_20261025',[['九任务对照','A1:H28'],['核心计算','A4:M13'],['三轮原值与来源','A4:K13'],['LG时间组成','A1:G15'],['LG阶段中文说明','A4:F13']]);
}
if(!MAIN_ONLY){
 const w=Workbook.create(),s=sheet(w,'Temporal对照','Temporal十六视频六十四帧核心对照',90,10);s.tabColor=ink;
 const records=[...d.tempRecords].sort((a,b)=>(a.system==='ours'?-1:b.system==='ours'?1:0)),runs=d.tempRuns;
 inputs(w,records,runs);totals(w,records,runs);
 const lg=records.findIndex(x=>x.system==='ours')+6;
 table(s,5,['模型','A100平台及功率上限','资源上限','完整TEST SRCC','16视频同步 ms','等效单帧 ms','相对LG时间比','16视频能量 J','相对LG能耗比','同平台Qwen100相对时间比'],records.map(r=>[title(r),r.power_limit===400?'80GB 400W':'40GB 250W',r.cap??100,r.accuracy,null,null,null,null,null,null]),[20,28,16,24,26,25,24,26,25,31]);
 for(let i=0;i<records.length;i++){let r=records[i],n=i+6,c=i+6;form(s,'E'+n,`='核心计算'!H${c}*16`);form(s,'F'+n,`=E${n}/64`);form(s,'G'+n,`=E${n}/E${lg}`);form(s,'H'+n,`='核心计算'!I${c}*16`);form(s,'I'+n,`=H${n}/H${lg}`);const anchor=records.findIndex(x=>x.system==='qwen'&&x.cap===100&&x.power_limit===r.power_limit)+6;form(s,'J'+n,`=E${anchor}/E${n}`);}
 num(s,`C6:C${records.length+5}`,'0');num(s,`D6:D${records.length+5}`,'0.0000');num(s,`E6:J${records.length+5}`);for(const c of ['G','I','J'])num(s,`${c}6:${c}${records.length+5}`,'0.00"×"');
 s.getRange('A3').values=[['LG真实每场16视频64帧。三轮39场624槽、600有效视频；包含尾补齐。基线单视频四帧累计后同样折16视频。']];
 table(s,24,['模型','资源上限','共同词编号每秒','LG相对本行节能率','本行除LG能量时延积比','比较范围'],records.map(r=>[title(r),r.cap??100,null,null,null,r.power_limit===400?'不同平台系统级对照':'同40GB平台']),[20,20,29,28,31,38]);
 for(let i=0;i<records.length;i++){let n=i+25,a=i+6;form(s,'C'+n,`='核心计算'!L${a}`);form(s,'D'+n,`=1-1/I${a}`);form(s,'E'+n,`=G${a}*I${a}`);}
 num(s,'C25:E38');num(s,'D25:D38','0.00%');
 s.getRange('A41').values=[['80GB对LG倍率只表示两套系统的已记录比较，不归因为同一硬件上的纯模型收益；同平台Qwen100比较另列。']];
 s.getRange('A43').values=[['折16视频＝三轮合计÷600×16。等效单帧＝16视频耗时÷64。双方同除64，倍数不会额外变大。']];
 s.getRange('A45').values=[['资源上限100%/50%是MPS配置，不表示实测利用率恒定。LG仅沿用100%，没有另测LG50%。']];
 notes(s,46);
 await finish(w,'A100_Temporal多基线_首计算模块核心对照_20261025',[['Temporal对照','A1:J20'],['Temporal对照','A23:F39'],['核心计算','A4:M13'],['三轮原值与来源','A4:K13']]);
}
