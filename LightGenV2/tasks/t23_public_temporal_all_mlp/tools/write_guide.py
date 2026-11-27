"""Apply the finalized guide format to the author's MLP inference description."""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import posixpath
import zipfile

from lxml import etree
from PIL import Image


REPO = Path(__file__).resolve().parents[4]
PROJECT = REPO/'LightGen-II-MLP'
NS = {
    'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'rel':'http://schemas.openxmlformats.org/package/2006/relationships',
}


def tag(name):
    prefix, local = name.split(':')
    return '{'+NS[prefix]+'}'+local


def text(element):
    return ''.join(element.xpath('.//w:t/text()', namespaces=NS))


def replace_text(element, value):
    nodes = element.xpath('.//w:t', namespaces=NS)
    if not nodes:
        raise ValueError('The retained source text is missing')
    nodes[0].text = value
    nodes[0].set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    for node in nodes[1:]:
        node.text = ''


def property_value(paragraph, name, value):
    properties = paragraph.find('w:pPr', NS)
    if properties is None:
        properties = etree.Element(tag('w:pPr'))
        paragraph.insert(0, properties)
    element = properties.find('w:'+name, NS)
    if element is None:
        element = etree.SubElement(properties, tag('w:'+name))
    element.set(tag('w:val'), value)


def serialize(element):
    return etree.tostring(element, xml_declaration=True, encoding='UTF-8', standalone=True)


def build(args):
    reference = args.reference.resolve()
    author = args.author.resolve()
    if hashlib.sha256(reference.read_bytes()).hexdigest() != args.reference_sha256:
        raise ValueError('The retained finalized reference has changed')
    with zipfile.ZipFile(reference) as archive:
        entries = archive.infolist()
        parts = {entry.filename:archive.read(entry) for entry in entries}
    with zipfile.ZipFile(author) as archive:
        author_root = etree.fromstring(archive.read('word/document.xml'))
    author_elements = list(author_root.find('w:body', NS))
    if text(author_elements[12]) != 'How to execute':
        raise ValueError('The author-edited MLP guide structure has changed')
    root = etree.fromstring(parts['word/document.xml'])
    body = root.find('w:body', NS)
    elements = list(body)
    if text(elements[0]) != 'Codes for LightGen-II' or len(elements) != 28:
        raise ValueError('The finalized reference structure has changed')
    metrics = json.loads((PROJECT/'reference_results/02_metrics.json').read_text('utf-8'))
    if metrics['videos'] != 558 or not .795 <= metrics['SRCC'] <= .799:
        raise ValueError('The finalized MLP score must be verified before publishing')
    score = format(metrics['SRCC'], '.2f')
    changes = {
        2:text(author_elements[2]),
        13:'Run the following command in the LightGen-II-MLP directory.',
        15:text(author_elements[15]).replace('frozen frontend', 'Qwen-derived frontend'),
        16:text(elements[16]).replace('SRCC) of 0.80', 'SRCC) of '+score),
        17:text(author_elements[19]),
        24:text(elements[24]).replace('approximately 0.80', 'approximately '+score),
    }
    for index, value in changes.items():
        replace_text(elements[index], value)
    rows = elements[9].findall('w:tr', NS)
    if len(rows) != 9:
        raise ValueError('Expected the nine finalized source-file rows')
    replace_text(rows[3].findall('w:tc', NS)[1], 'The network definition.')
    replace_text(rows[5].findall('w:tc', NS)[1],
                 'The functions to save predictions and visualize the input frames.')
    result_rows = elements[11].findall('w:tr', NS)
    replace_text(result_rows[2].findall('w:tc', NS)[0], '03_input_frames.png')
    replace_text(result_rows[2].findall('w:tc', NS)[1],
                 'The four input frames for each of the first 16 videos.')
    for row in result_rows[3:]:
        elements[11].remove(row)

    image_paragraph = deepcopy(elements[19])
    for attribute in list(image_paragraph.attrib):
        if etree.QName(attribute).localname in ('paraId', 'textId'):
            del image_paragraph.attrib[attribute]
    property_value(image_paragraph, 'pageBreakBefore', '0')
    property_value(image_paragraph, 'keepNext', '1')
    image = image_paragraph.find('.//w:drawing', NS)
    blip = image.find('.//a:blip', NS)
    image_id = blip.get(tag('r:embed'))
    preview = PROJECT/'reference_results/03_input_frames.png'
    with Image.open(preview) as picture:
        width, height = picture.size
    extent_width = int(5.2*914400)
    extent_height = round(extent_width*height/width)
    for extent in image.xpath('.//wp:extent | .//a:xfrm/a:ext', namespaces=NS):
        extent.set('cx', str(extent_width))
        extent.set('cy', str(extent_height))
    for properties in image.findall('.//wp:docPr', NS):
        properties.set('name', 'Input video frames')
        properties.set('descr', 'The four input frames for each of the first 16 LGVQ videos')
    caption = deepcopy(elements[20])
    replace_text(caption, text(author_elements[21]).replace('04_input_frames.png', '03_input_frames.png'))
    for element in elements[18:24]:
        body.remove(element)
    body.insert(18, image_paragraph)
    body.insert(19, caption)

    relations = etree.fromstring(parts['word/_rels/document.xml.rels'])
    for relation in list(relations):
        if not relation.get('Type', '').endswith('/image'):
            continue
        target = posixpath.normpath('word/'+relation.get('Target'))
        if relation.get('Id') == image_id:
            parts[target] = preview.read_bytes()
        else:
            relations.remove(relation)
            parts.pop(target, None)
    parts['word/document.xml'] = serialize(root)
    parts['word/_rels/document.xml.rels'] = serialize(relations)
    destination = PROJECT/'README.docx'
    temporary = destination.with_suffix('.docx.building')
    with zipfile.ZipFile(temporary, 'w') as output:
        for entry in entries:
            if entry.filename in parts:
                output.writestr(entry, parts[entry.filename])
    temporary.replace(destination)
    print(str(destination))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--author', type=Path, required=True)
    build(parser.parse_args())
