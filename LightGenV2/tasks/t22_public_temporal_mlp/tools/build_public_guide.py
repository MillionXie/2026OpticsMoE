"""Build the public README and editable guide from the same concise content."""
from pathlib import Path
import hashlib
import json

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PUBLIC = Path(__file__).resolve().parents[4] / 'LightGen2'
TITLE = 'Codes for temporal video quality assessment with LightGen2'
INTRODUCTION = (
    'This is the code to assess temporal video quality with LightGen2 in Python '
    'using PyTorch. We provide pretrained weights and video frames from the LGVQ '
    'dataset to evaluate 558 videos, with 16 videos processed in parallel. '
    'The code predicts a mean opinion score (MOS) for each video and provides '
    'examples of the optical wave propagation for reference.'
)
ENVIRONMENT = (
    'We have tested the code with Python 3.11.15 and PyTorch 2.6.0 on an NVIDIA '
    'A100 GPU. For the required Python packages, please refer to requirements.txt.'
)
INSTALLATION = (
    'Please follow the PyTorch installation guide to install a build compatible '
    'with your system. Then install the dependent packages from this directory.'
)
EXECUTION = (
    'The code first reads '
    'the video frames from data/input_for_review.npz and loads the model weights '
    'from weights/LightGen2.pt. It extracts visual features with the Qwen-derived '
    'frontend and processes them with routed experts and optical propagation '
    'in the vision branch. The language branch combines these features with '
    'the prompt features and applies its routed experts and optical propagation. '
    'The two-layer readout predicts '
    'one MOS for each video. Higher MOS indicates better temporal video quality. '
    'We then use the human MOS in data/reference_mos.csv to calculate SRCC.'
)
OPTICAL_EXAMPLE = (
    'The same execution saves the input video frames and captures the input '
    'field, learned mask and output field for one group of 16 videos. '
    'The examples below show the vision branch in the top row and the language '
    'branch in the bottom row, ordered from input to mask to output. '
    'Each optical result is saved as numerical data in an NPZ file and as a PNG image.'
)
FILES = (
    ('data/input_for_review.npz', 'The input video frames from 558 LGVQ videos.'),
    ('data/reference_mos.csv', 'The human temporal MOS labels used to calculate SRCC from the predictions.'),
    ('weights/LightGen2.pt', 'Model weights, including the frozen Qwen frontend and prompt features.'),
    ('model.py', 'The network definition and optical propagation based on the angular spectrum method.'),
    ('infer.py', 'The script to run the complete video quality assessment.'),
    ('visualize.py', 'The functions to save predictions and visualize the input frames and optical fields.'),
    ('requirements.txt', 'The required Python packages.'),
    ('README.md and README.docx', 'Instructions for installing and running the code. The Word version is editable.'),
    ('LICENSE, NOTICE and licenses/', 'The Apache-2.0 license and attribution for upstream components and data.'),
)
RESULT_FILES = (
    ('01_predictions.csv', 'The video ID, human MOS and predicted MOS for each of the 558 videos.'),
    ('02_metrics.json', 'SRCC, number of videos, complete program runtime and computing device.'),
    ('03_prediction_vs_mos.png', 'The predicted MOS compared with the human MOS for all videos.'),
    ('04_input_frames.png', 'The input frames for the 16 videos used in the optical example.'),
    ('05_vision_input_field.npz and .png', 'The input optical field of the vision branch.'),
    ('06_vision_mask.npz and .png', 'The complete learned phase mask of the vision branch.'),
    ('07_vision_output_field.npz and .png', 'The output optical field of the vision branch.'),
    ('08_language_input_field.npz and .png', 'The input optical field of the language branch.'),
    ('09_language_mask.npz and .png', 'The complete learned phase mask of the language branch.'),
    ('10_language_output_field.npz and .png', 'The output optical field of the language branch.'),
)
OPTICAL_IMAGES = tuple(
    f'results/{number:02d}_{branch}_{kind}.png'
    for branch, first in (('vision', 5), ('language', 8))
    for number, kind in enumerate(('input_field', 'mask', 'output_field'), first)
)


def runtime_reference(reference):
    task = Path(__file__).resolve().parents[1]
    profile = json.loads((task / 'configs/selected.json').read_text('utf-8'))
    record = profile['runtime_benchmark']
    directory = task / record['directory']
    summary = json.loads((directory / 'summary.json').read_text('utf-8'))
    protocol = json.loads((directory / 'protocol.json').read_text('utf-8'))
    timing = summary['projects'][record['project']]['program_seconds']
    for name, expected in protocol['input_and_source_sha256'].items():
        with (PUBLIC / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Runtime measurement identity differs: ' + name)
    if (reference['total_runtime_seconds'] != timing['median']
            or reference['device_name'] != summary['hardware']['device']
            or reference['videos'] != summary['videos_per_process']):
        raise ValueError('Reference results differ from the selected runtime measurement')
    return summary['measured_processes']


def results_description(reference, repeats):
    return (
        'Predictions and figures are saved in results/. For the supplied 558 videos, '
        f"we obtained a Spearman rank correlation coefficient (SRCC) of {reference['SRCC']:.6f} "
        'between the predicted and human MOS. '
        f"The median complete program runtime was {reference['total_runtime_seconds']:.3f} seconds "
        f"over {repeats} runs on a shared {reference['device_name']}. "
        'Each execution records its own runtime in 02_metrics.json.'
    )


def markdown_table(rows):
    lines = ['| File | Description |', '| --- | --- |']
    return '\n'.join(lines + [f'| `{name}` | {description} |' for name, description in rows])


def build_readme(reference, repeats):
    images = [f'![{Path(name).stem[3:].replace("_", " ")}]({name})' for name in OPTICAL_IMAGES]
    text = f'''# {TITLE}

## Introduction

{INTRODUCTION}

## Environment

{ENVIRONMENT}

## Installation guide

{INSTALLATION} [PyTorch installation guide](https://pytorch.org/get-started/locally/)

```bash
python -m pip install -r requirements.txt
```

## Files

{markdown_table(FILES)}

The following files are generated in `results/`.

{markdown_table(RESULT_FILES)}

## How to execute

Run the following command in the LightGen2 directory.

```bash
python infer.py
```

{EXECUTION}

{results_description(reference, repeats)}

![Predicted and human temporal MOS](results/03_prediction_vs_mos.png)

{OPTICAL_EXAMPLE}

| Input | Mask | Output |
| --- | --- | --- |
| {images[0]} | {images[1]} | {images[2]} |
| {images[3]} | {images[4]} | {images[5]} |

The Qwen components and LGVQ data retain their upstream attribution. See `NOTICE`
and `licenses/QWEN_LICENSE.txt`. [Qwen](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [LGVQ](https://github.com/zczhang-sjtu/UGVQ)
'''
    (PUBLIC / 'README.md').write_text(text, encoding='utf-8', newline='\n')


def file_table(document, rows):
    table = document.add_table(rows=0, cols=2)
    table.autofit = False
    widths = (2.65, 4.35)
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    for name, description in rows:
        cells = table.add_row().cells
        for cell, width, text in zip(cells, widths, (name, description)):
            cell.width = Inches(width)
            cell.text = text
            properties = cell._tc.get_or_add_tcPr()
            borders = OxmlElement('w:tcBorders')
            for edge in ('top', 'left', 'bottom', 'right'):
                border = OxmlElement('w:' + edge)
                for key, value in (('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')):
                    border.set(qn('w:' + key), value)
                borders.append(border)
            properties.append(borders)
            margins = OxmlElement('w:tcMar')
            for edge in ('top', 'left', 'bottom', 'right'):
                margin = OxmlElement('w:' + edge)
                margin.set(qn('w:w'), '70')
                margin.set(qn('w:type'), 'dxa')
                margins.append(margin)
            properties.append(margins)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(1)
                paragraph.paragraph_format.line_spacing = 1.05
                for run in paragraph.runs:
                    run.font.size = Pt(10.5)
        table.rows[-1]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
    return table


def command(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(.15)
    run = paragraph.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(10.5)


def build_guide():
    reference = json.loads((PUBLIC / 'results/02_metrics.json').read_text('utf-8'))
    repeats = runtime_reference(reference)
    build_readme(reference, repeats)
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(.65)
    section.left_margin = section.right_margin = Inches(.75)
    normal = document.styles['Normal']
    normal.font.name, normal.font.size = 'Times New Roman', Pt(11.5)
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.space_after = Pt(6)
    for name, size in (('Title', 17), ('Heading 1', 14)):
        style = document.styles[name]
        style.font.name, style.font.size = 'Times New Roman', Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(6)
    document.styles['Title'].font.bold = True
    for name in ('Normal', 'Title', 'Heading 1'):
        fonts = document.styles[name].element.get_or_add_rPr().get_or_add_rFonts()
        for attribute in tuple(fonts.attrib):
            if attribute.endswith('Theme'):
                del fonts.attrib[attribute]
        for attribute in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
            fonts.set(qn('w:' + attribute), 'Times New Roman')
    document.core_properties.title = TITLE
    document.core_properties.subject = 'Installation and execution of LightGen2 video quality assessment'
    document.core_properties.author = 'LightGen2 contributors'
    document.add_heading(TITLE, 0)
    document.add_heading('Introduction', 1)
    document.add_paragraph(INTRODUCTION)
    document.add_heading('Environment', 1)
    document.add_paragraph(ENVIRONMENT)
    document.add_heading('Installation guide', 1)
    document.add_paragraph(INSTALLATION + ' https://pytorch.org/get-started/locally/')
    command(document, 'python -m pip install -r requirements.txt')
    document.add_heading('Files', 1)
    file_table(document, FILES)
    results_heading = document.add_paragraph('The following files are generated in results/.')
    results_heading.paragraph_format.page_break_before = True
    file_table(document, RESULT_FILES)
    document.add_heading('How to execute', 1)
    document.add_paragraph('Run the following command in the LightGen2 directory.')
    command(document, 'python infer.py')
    document.add_paragraph(EXECUTION)
    document.add_paragraph(results_description(reference, repeats))
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(PUBLIC / 'results/03_prediction_vs_mos.png'), width=Inches(2.75))
    optical_description = document.add_paragraph(OPTICAL_EXAMPLE)
    optical_description.paragraph_format.keep_with_next = True
    optical_description.paragraph_format.keep_together = True
    for start in (0, 3):
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = start == 0
        for name in OPTICAL_IMAGES[start:start + 3]:
            paragraph.add_run().add_picture(str(PUBLIC / name), width=Inches(2.3))
    for root in (document.styles.element, document.element):
        for border in list(root.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
    document.save(PUBLIC / 'README.docx')
    print(json.dumps({'guide': str(PUBLIC / 'README.docx'), 'SRCC': reference['SRCC']}))


if __name__ == '__main__':
    build_guide()
