/** Baselines only: preserved continuous file-to-CPU timing, no model execution. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const HERE=path.dirname(fileURLToPath(import.meta.url)),REPO=path.resolve(HERE,'../../../..');
const BASE=path.join(HERE,'outputs/01a061e2-6f88-7913-87f5-b632000e4c83');
const OUT=path.join(BASE,'baseline_e2e_fps_20261025_01');
const source=path.join(BASE,'complete_20261025_01/collected_data_and_audit.json');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const sourceBytes=await fs.readFile(source),sourceSHA=hash(sourceBytes);
if(sourceSHA!=='f0cd430b99aae15c69ddbeeeee1aa2d2b1334a6e804d7a91097a79c6659d2ffe')throw Error('Sealed source differs');
try{await fs.access(path.join(OUT,'delivery_manifest.json'));throw Error('Preserve sealed delivery');}catch(e){if(e.code!=='ENOENT')throw e;}
const d=JSON.parse(sourceBytes),records=d.tempRecords.filter(x=>x.system!=='ours');
const runs=d.tempRuns.filter(x=>x.system!=='ours');
if(records.length!==13||runs.length!==39)throw Error('Expected 13 baseline configurations and 39 original rounds');
const pins=new Map(d.raw_files.map(x=>[x.file.replaceAll('\\','/'),x]));
const names={qwen:'Qwen',glm:'GLM',clip:'CLIP',yolo:'YOLO',deepseek:'DeepSeek',kimi:'Kimi'};
const platform=r=>r.key.startsWith('80GB')?'A100 80GB / 400W':'A100 40GB / 250W';
const provenance=[];
for(const r of runs){
 const file=r.folder+'/'+r.reportName,pin=pins.get(file),bytes=await fs.readFile(path.join(REPO,file));
 if(!pin||hash(bytes)!==pin.sha256||bytes.length!==pin.bytes)throw Error('Original report SHA/size differs: '+file);
 const report=JSON.parse(bytes);if(report.status!=='complete')throw Error('Incomplete original report');
 const ms=report.cost?.sample_ms??report.cost?.ms_per_video??report.sample_ms??report.continuous_wall_s*5;
 if(Math.abs(ms*r.n/1000-r.seconds200)>1e-7)throw Error('Original continuous time differs');
 provenance.push({file,bytes:bytes.length,sha256:pin.sha256});
}
const w=Workbook.create(),s=w.worksheets.add('端到端帧率'),raw=w.worksheets.add('三轮原始耗时');
const ink='#243A50';
function base(sheet,range){sheet.showGridLines=false;sheet.getRange(range).format={font:{name:'Arial',size:11,color:ink},rowHeight:27,verticalAlignment:'center'};}
function header(sheet,row,labels,widths){
 const last=String.fromCharCode(64+labels.length);sheet.getRange(`A${row}:${last}${row}`).values=[labels];
 sheet.getRange(`A${row}:${last}${row}`).format={fill:ink,font:{bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:46,horizontalAlignment:'center'};
 widths.forEach((width,i)=>sheet.getRange(`${String.fromCharCode(65+i)}1:${String.fromCharCode(65+i)}${sheet===s?22:44}`).format.columnWidth=width);
}
function formula(sheet,cell,value){sheet.getRange(cell).formulas=[[value]];sheet.getRange(cell).format.fill='#EDF5EE';}
base(s,'A1:J22');base(raw,'A1:J44');s.tabColor=ink;
s.getRange('A2').values=[['Temporal 多基线端到端帧率']];s.getRange('A2').format.font={size:16,bold:true};
s.getRange('A3').values=[['每视频输入帧数']];s.getRange('B3').values=[[4]];s.getRange('B3').format.fill='#FFF0D6';s.getRange('B3').setNumberFormat('0');
header(s,5,['模型','平台及功率上限','MPS资源上限','端到端 ms/视频','等效 ms/帧','端到端 frame/s'],[18,29,20,24,23,25]);
for(let i=0;i<records.length;i++){
 const r=records[i],n=i+6,rr=runs.flatMap((v,j)=>v.key===r.key?[j+6]:[]);
 if(rr.length!==3)throw Error('Three qualified rounds required');
 s.getRange(`A${n}:C${n}`).values=[[names[r.system],platform(r),r.cap/100]];
 formula(s,'D'+n,`=(${rr.map(x=>`'三轮原始耗时'!F${x}`).join('+')})/(${rr.map(x=>`'三轮原始耗时'!E${x}`).join('+')})*1000`);
 formula(s,'E'+n,`=D${n}/$B$3`);formula(s,'F'+n,`=1000/E${n}`);
 if(i%2===1)s.getRange(`A${n}:C${n}`).format.fill='#F5F7FA';
}
s.getRange('C6:C18').setNumberFormat('0%');s.getRange('D6:E18').setNumberFormat('0.000');s.getRange('F6:F18').setNumberFormat('0.00');s.getRange('C6:F18').format.horizontalAlignment='right';
s.getRange('H5').values=[['端到端计时范围']];s.getRange('H5').format.font.bold=true;
s.getRange('H7').values=[['视频读取、解码、预处理、分词、显存传输、完整网络、评分结果返回。']];
s.getRange('H9').values=[['使用原连续窗口，包含200条请求之间的199段间隙。']];
s.getRange('H11').values=[['frame/s＝三轮总视频数×4÷三轮总秒数，不平均三轮帧率。']];
s.getRange('H13').values=[['这里的“帧”指每视频实际输入的4个抽样帧，不是原视频全部帧。']];
s.getRange('H15').values=[['MPS百分比是资源配置上限，不是实测利用率。40GB与80GB分别标明。']];
s.getRange('H5:J18').format.columnWidth=45;s.getRange('H7:J15').format.wrapText=false;
raw.getRange('A2').values=[['原封存连续时间：每配置三轮，每轮200视频']];raw.getRange('A2').format.font={size:15,bold:true};
header(raw,5,['模型','平台及功率上限','MPS资源上限','轮次','有效视频数','原连续总耗时 s','等效输入帧数','本轮 frame/s','原报告路径','原报告 SHA256'],[18,29,20,10,18,24,22,24,105,72]);
runs.forEach((r,i)=>{
 const n=i+6,record=records.find(x=>x.key===r.key),p=provenance[i];
 raw.getRange(`A${n}:J${n}`).values=[[names[r.system],platform(r),record.cap/100,r.repeat,r.n,r.seconds200,null,null,p.file,p.sha256]];
 raw.getRange(`E${n}:F${n}`).format.fill='#EAF2F9';
 formula(raw,'G'+n,`=E${n}*'端到端帧率'!$B$3`);formula(raw,'H'+n,`=G${n}/F${n}`);
});
raw.getRange('C6:C44').setNumberFormat('0%');raw.getRange('D6:E44').setNumberFormat('0');raw.getRange('F6:F44').setNumberFormat('0.000000');raw.getRange('G6:G44').setNumberFormat('#,##0');raw.getRange('H6:H44').setNumberFormat('0.00');raw.getRange('C6:H44').format.horizontalAlignment='right';raw.freezePanes.freezeRows(5);
w.recalculate();
const results=records.map((r,i)=>{
 const own=runs.filter(x=>x.key===r.key),seconds=own.reduce((a,x)=>a+x.seconds200,0),videos=own.reduce((a,x)=>a+x.n,0),fps=videos*4/seconds;
 if(Math.abs(s.getRange('F'+(i+6)).values[0][0]-fps)>1e-9)throw Error('Frame/s formula differs');
 if(Math.abs(seconds/videos*1000-r.ms)>1e-7)throw Error('Sealed aggregate differs');
 return {model:names[r.system],platform:platform(r),MPS_percent:r.cap,ms_per_video:r.ms,frame_per_s:fps};
});
console.log((await w.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:10},maxChars:500})).ndjson);
await fs.mkdir(OUT,{recursive:true});
for(const [sheetName,range,file] of [['端到端帧率','A1:F19','summary.png'],['三轮原始耗时','A4:H14','rounds.png']]){
 const png=await w.render({sheetName,range,scale:1.5});await fs.writeFile(path.join(OUT,file),new Uint8Array(await png.arrayBuffer()));
}
const filename='A100_Temporal多基线_端到端帧率_20261025.xlsx',xlsx=await SpreadsheetFile.exportXlsx(w);await xlsx.save(path.join(OUT,filename));
const outputBytes=await fs.readFile(path.join(OUT,filename));
await fs.writeFile(path.join(OUT,'delivery_manifest.json'),JSON.stringify({status:'complete',scope:'original continuous file input to CPU result, all199 gaps included',frames_per_video:4,models:6,configurations:13,qualified_rounds:39,no_LightGen:true,no_model_or_GPU_execution:true,source:{file:path.relative(REPO,source),sha256:sourceSHA},raw_reports:provenance,output:{file:filename,bytes:outputBytes.length,sha256:hash(outputBytes)},results},null,2)+'\n');
console.log(JSON.stringify({output:path.join(OUT,filename),bytes:outputBytes.length,sha256:hash(outputBytes),results}));
