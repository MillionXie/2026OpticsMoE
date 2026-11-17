"""Adapt the existing LightGen-II Word guide for the MLP inference project."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import posixpath
import zipfile

from lxml import etree
from PIL import Image


REPO = Path(__file__).resolve().parents[4]
PROJECT = REPO/'LightGen2_MLP'
REFERENCE = REPO/'LightGen2/README.docx'
REFERENCE_SHA256 = '301c6d8df887f130d03eab50a627d83f97cc1f3b67fece3f9e810e06cb082f33'
NAMESPACES = {
    'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'rel':'http://schemas.openxmlformats.org/package/2006/relationships',
}


def tag(name):
    prefix, local = name.split(':')
    return '{'+NAMESPACES[prefix]+'}'+local


def replace_paragraph(paragraph, text):
    properties = paragraph.find('w:pPr', NAMESPACES)
    template = paragraph.find('w:r/w:rPr', NAMESPACES)
    for child in list(paragraph):
        if child is not properties:
            paragraph.remove(child)
    run = etree.SubElement(paragraph, tag('w:r'))
    if template is not None:
        run.append(deepcopy(template))
    value = etree.SubElement(run, tag('w:t'))
    value.text = text


def populate_table(table, rows):
    template = deepcopy(table.find('w:tr', NAMESPACES))
    for row in table.findall('w:tr', NAMESPACES):
        table.remove(row)
    for values in rows:
        row = deepcopy(template)
        for element in row.iter():
            for attribute in list(element.attrib):
                if etree.QName(attribute).localname in ('paraId','textId'):
                    del element.attrib[attribute]
        for cell, text in zip(row.findall('w:tc', NAMESPACES), values):
            paragraphs = cell.findall('w:p', NAMESPACES)
            replace_paragraph(paragraphs[0], text)
            for paragraph in paragraphs[1:]:
                cell.remove(paragraph)
        table.append(row)


def serialize(element):
    return etree.tostring(element, xml_declaration=True, encoding='UTF-8', standalone=True)


def build():
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError('The retained Word template has changed')
    metrics = json.loads((PROJECT/'results/02_metrics.json').read_text('utf-8'))
    with zipfile.ZipFile(REFERENCE) as source:
        parts = {name:source.read(name) for name in source.namelist()}
        entries = source.infolist()
    root = etree.fromstring(parts['word/document.xml'])
    body = root.find('w:body', NAMESPACES)
    elements = list(body)
    texts = {
        0:'Codes for LightGen-II MLP version',
        2:'This is the code to assess temporal video quality with the MLP version of LightGen-II '
          'in Python using PyTorch. We provide pretrained weights and video frames from the LGVQ '
          'dataset to evaluate 558 videos. The code predicts a mean opinion score (MOS) for each video.',
        4:'We have tested the code with Python 3.11.15 and PyTorch 2.6.0, and with Python 3.8 '
          'and PyTorch 2.4.1. For the required Python packages, please refer to requirements.txt.',
        13:'Run the following command in the LightGen2_MLP directory.',
        15:'The code first reads the video frames from data/input_for_review.npz and loads the model '
           'weights from weights/LightGen2.pt. It extracts visual features with the frozen frontend '
           'and processes them with routed experts and a two-layer global MLP in the vision branch. '
           'The language branch combines these features with the prompt features and applies its '
           'routed experts and global MLP. The two-layer readout predicts one MOS for each video. '
           'Higher MOS indicates better temporal video quality. We then use the human MOS in '
           'data/reference_mos.csv to calculate SRCC.',
        16:'Predictions and figures are saved in results/. For the supplied 558 videos, we obtained '
           f'a Spearman rank correlation coefficient (SRCC) of {metrics["SRCC"]:.2f} between '
           'the predicted and human MOS.',
        19:'The same execution processes all 558 videos in batches of 16 and saves an input-frame '
           'preview for the first batch. Each panel below shows the four frames used for one video. '
           'The preview and the quality predictions are generated from the same inference run.',
        21:'Input frames for the first 16 LGVQ videos (04_input_frames.png).',
        24:f'In summary, this MLP version achieved an SRCC of {metrics["SRCC"]:.2f} on 558 LGVQ '
           'videos, comparable to the original LightGen-II software reference (SRCC 0.80). '
           'The supplied weights, video frames and per-video predictions support reproduction '
           'of this temporal video quality assessment.',
    }
    for index, text in texts.items():
        replace_paragraph(elements[index], text)
    source_rows = [
        ('data/input_for_review.npz', 'The input video frames from 558 LGVQ videos.'),
        ('data/reference_mos.csv', 'The human temporal MOS labels used to calculate SRCC from the predictions.'),
        ('weights/LightGen2.pt', 'Model weights.'),
        ('model.py', 'The network definition, including the vision and language global MLPs.'),
        ('infer.py', 'The script to run the complete video quality assessment.'),
        ('data.py', 'The functions to read video frames and reference scores and prepare batches.'),
        ('visualize.py', 'The functions to save predictions and visualize the input frames and quality scores.'),
        ('requirements.txt', 'The required Python packages.'),
        ('README.pdf', 'Instructions for installing and running the code.'),
        ('LICENSE and licenses/', 'The project license and third-party notices and licenses.'),
    ]
    result_rows = [
        ('01_predictions.csv', 'The video ID, human MOS and predicted MOS for each of the 558 videos.'),
        ('02_metrics.json', 'Number of videos and SRCC.'),
        ('03_prediction_vs_mos.png', 'The predicted MOS compared with the human MOS for all videos.'),
        ('04_input_frames.png', 'The four input frames for each of the first 16 videos.'),
    ]
    populate_table(elements[9], source_rows)
    populate_table(elements[11], result_rows)

    image_paragraph = elements[20]
    runs = image_paragraph.findall('w:r', NAMESPACES)
    retained_run = next(run for run in runs if run.find('.//w:drawing', NAMESPACES) is not None)
    for child in list(image_paragraph):
        if child is not retained_run and child.tag != tag('w:pPr'):
            image_paragraph.remove(child)
    drawing = retained_run.find('w:drawing', NAMESPACES)
    blip = drawing.find('.//a:blip', NAMESPACES)
    image_id = blip.get(tag('r:embed'))
    preview = PROJECT/'results/04_input_frames.png'
    with Image.open(preview) as picture:
        width, height = picture.size
    extent_width = int(4.35*914400)
    extent_height = round(extent_width*height/width)
    for extent in drawing.xpath('.//wp:extent | .//a:xfrm/a:ext', namespaces=NAMESPACES):
        extent.set('cx', str(extent_width))
        extent.set('cy', str(extent_height))
    for properties in drawing.findall('.//wp:docPr', NAMESPACES):
        properties.set('descr', 'Input video frames for 16 LGVQ videos')
        properties.set('name', 'Input video frames')
    body.remove(elements[22])
    body.remove(elements[23])

    relations = etree.fromstring(parts['word/_rels/document.xml.rels'])
    scatter_id = elements[17].find('.//a:blip', NAMESPACES).get(tag('r:embed'))
    images = {scatter_id:PROJECT/'results/03_prediction_vs_mos.png', image_id:preview}
    removed = set()
    for relation in list(relations):
        if not relation.get('Type','').endswith('/image'):
            continue
        target = posixpath.normpath('word/'+relation.get('Target'))
        identity = relation.get('Id')
        if identity in images:
            parts[target] = images[identity].read_bytes()
        else:
            relations.remove(relation)
            removed.add(target)
    for name in removed:
        parts.pop(name, None)
    parts['word/document.xml'] = serialize(root)
    parts['word/_rels/document.xml.rels'] = serialize(relations)
    settings = etree.fromstring(parts['word/settings.xml'])
    update = settings.find('w:updateFields', NAMESPACES)
    if update is None:
        update = etree.SubElement(settings, tag('w:updateFields'))
    update.set(tag('w:val'), 'true')
    parts['word/settings.xml'] = serialize(settings)
    core = etree.fromstring(parts['docProps/core.xml'])
    for element in core:
        local = etree.QName(element).localname
        if local == 'title':
            element.text = texts[0]
        elif local in ('creator','lastModifiedBy'):
            element.text = ''
    parts['docProps/core.xml'] = serialize(core)

    destination = PROJECT/'README.docx'
    with zipfile.ZipFile(destination, 'w') as output:
        for entry in entries:
            if entry.filename in parts:
                output.writestr(entry, parts[entry.filename])
    print(str(destination))


if __name__ == '__main__':
    build()
