"""Compose existing, hash-verified outputs; never generate or retouch model images."""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont

TASK = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
DEV = TASK / 'runs/hardware/scene_F9_DEV128_registered_20261031'
BASE = TASK / 'runs/simulation/scene_natural_electronic30_fused_DEV128_20261030'
SIM = TASK / 'runs/simulation/scene_natural_joint12_adapted_fused_DEV128_20261030'
EXP = DEV / 'EXP_adapt1'
SAMPLES = [
    'lamp-B075X1T517-02-scene1-draw2',
    'lamp-B07B4VYKPR-11-scene0-draw2',
    'pillow-B07C8MSZ5T-09-scene1-draw2',
    'pillow-B07JL5PLPD-09-scene3-draw2',
]
CN = [
    '阅读角：浅色石质边桌上的台灯；场景包含扶手椅与书架。',
    '现代书房：橡木书桌上的台灯；场景包含书架与绿植。',
    '阅读角：扶手椅上的靠垫；场景包含石质边桌与书架。',
    '温馨客厅：布艺沙发上的靠垫；场景包含木质茶几与地毯。',
]

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc', size)

canvas = Image.new('RGB', (2160, 3110), 'white')
draw = ImageDraw.Draw(canvas)
INK, GRAY, BLUE = '#17212b', '#576471', '#205575'

def text(x, y, value, size=26, fill=INK, bold=False):
    draw.text((x, y), value, font=font(size, bold), fill=fill)

def wrap(value, width, size):
    lines, line = [], ''
    # Word wrapping for English; character wrapping for Chinese.
    tokens = value.split(' ') if value.isascii() else list(value)
    sep = ' ' if value.isascii() else ''
    for token in tokens:
        candidate = line + (sep if line else '') + token
        if line and draw.textlength(candidate, font=font(size)) > width:
            lines.append(line)
            line = token
        else:
            line = candidate
    return lines + ([line] if line else [])

def paragraph(x, y, value, width, size=24, fill=INK, leading=35):
    for line in wrap(value, width, size):
        text(x, y, line, size, fill)
        y += leading
    return y

records = [{r['sample_id']: r for r in read(folder / 'images.json')} for folder in (BASE, SIM, EXP)]
assert all(len(r) == 128 for r in records)
manifest = {'scope': 'Selected examples, not a success-rate estimate', 'split': 'DEV128',
            'experiment': 'EXP_adapt1', 'head_sha256': 'dc151164a4e7222a9183cf43e24ed0e832377f5f8e940300eb32a2a2283702dc',
            'examples': []}

text(48, 30, 'LightGen-II | 图文条件生成结果', 48, bold=True)
text(48, 100, '四例选取展示 · 同一参考图、提示词、随机种子与初始噪声', 27, GRAY)
text(48, 165, '参考图与完整提示词', 29, bold=True)
for x, label, detail in zip((480, 1024, 1568),
                           ('Baseline · 电子基线', 'Simulation · 光电仿真', 'Experiment · 光电实验'),
                           ('E662 | 30 sampling steps', 'F9 | 12 sampling steps', 'EXP_adapt1 | 12 sampling steps')):
    text(x, 161, label, 30, BLUE, True)
    text(x, 205, detail, 23, GRAY)

for i, sample_id in enumerate(SAMPLES):
    y = 263 + i * 568
    trio = [r[sample_id] for r in records]
    for key in ('sample_id', 'prompt', 'seed', 'input_sha256', 'initial_latent_sha256'):
        assert len({str(r[key]) for r in trio}) == 1, (sample_id, key)
    reference = BASE / Path(trio[0]['reference']).name
    assert sha(reference) == trio[0]['reference_sha256']
    text(48, y, chr(65+i), 36, BLUE, True)
    with Image.open(reference) as im:
        im = im.convert('RGB')
        im.thumbnail((126,126), Image.Resampling.LANCZOS)
        canvas.paste(im, (104, y))
    paragraph(48, y+145, CN[i], 385, 26, leading=38)
    paragraph(48, y+285, trio[0]['prompt'], 385, 23, GRAY, 33)
    entry = {'sample_id': sample_id, 'prompt': trio[0]['prompt'], 'seed': trio[0]['seed'], 'files': []}
    for x, folder, rec in zip((480,1024,1568), (BASE,SIM,EXP), trio):
        path = folder / rec['generated_PNG']
        digest = sha(path)
        assert digest == rec['generated_sha256'], path
        with Image.open(path) as im:
            assert im.size == (512,512)
            canvas.paste(im.convert('RGB'), (x,y))
        entry['files'].append({'path': str(path.relative_to(TASK)), 'sha256': digest})
    manifest['examples'].append(entry)
    if i < 3:
        draw.line((48,y+540,2112,y+540), fill='#dee4e9', width=2)

summary = read(TASK / 'runs/simulation/scene_natural_joint12_adapted_metrics256_20261030/summary.json')
physical = read(DEV / 'quality_adapt1/summary.json')
rows = [summary['scene_natural_electronic30_fused_DEV128_20261030/all'],
        summary['scene_natural_joint12_adapted_fused_DEV128_20261030/all'], physical['EXP_adapt1/all']]
keys = ['CLIP_T', 'DINOv2_I', 'CLIP_I', 'ImageReward_compact']
manifest['metrics'] = {label: {k: row['scores'][k]['mean'] for k in keys}
                       for label,row in zip(('Baseline_E662','Simulation_F9','Experiment_adapt1'),rows)}
assert all(row['n']==128 for row in rows)
text(48, 2550, '全量 DEV128 指标', 34, bold=True)
text(420, 2558, '台灯 64 + 靠垫 64；所有指标均为均值，越高越好', 25, GRAY)
xs = (70, 700, 1040, 1380, 1730)
draw.rectangle((48,2614,2112,2671),fill='#ecf1f5')
for x,label in zip(xs, ('模型','CLIP-T ↑','DINOv2-I ↑','CLIP-I ↑','ImageReward ↑')):
    text(x,2624,label,27,bold=True)
for i,(label,row) in enumerate(zip(('Baseline · E662','Simulation · F9','Experiment · adapt1'),rows)):
    y=2686+i*56
    text(xs[0],y,label,27)
    for x,k in zip(xs[1:],keys):
        text(x,y,f"{row['scores'][k]['mean']:.4f}",27)

notes = [
    '选例不等于总体成功率：主要物体与位置关系可辨，但背景物件、材质及商品外观仍有偏差。',
    '实验：实测光学输出 + 原有电子模块适配（adapt1）；上表非四例均值，亦非独立 TEST500 结果。',
    '电子基线为 30 步，光电仿真 / 实验为 12 步；这是系统对照而非单因素消融。',
    '基线 / 仿真在 A100 生成，实验在 RTX 4090 生成；图像保持原始完整画面，无重绘、裁切或增强。',
    'ImageReward 使用固定简短描述评分，不表示完整提示词的遵循率；实验版本仍为开发候选。',
]
for i,note in enumerate(notes):
    text(48,2895+i*35,note,23,GRAY)

output = REPORT / 'HARDWARE_BASELINE_SIM_EXP_4EXAMPLES_20261031.png'
canvas.save(output, dpi=(200,200))
manifest['figure'] = str(output.relative_to(TASK))
manifest['figure_sha256'] = sha(output)
manifest['metric_sources'] = ['runs/simulation/scene_natural_joint12_adapted_metrics256_20261030/summary.json',
                              'runs/hardware/scene_F9_DEV128_registered_20261031/quality_adapt1/summary.json']
(DEV / 'teacher_four_examples_20261031.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
canvas.thumbnail((1080,1555),Image.Resampling.LANCZOS)
canvas.save(DEV / 'teacher_four_examples_preview_20261031.png')
print(json.dumps({'output':str(output), 'sha256':manifest['figure_sha256'], 'metrics':manifest['metrics']},ensure_ascii=False,indent=2))
