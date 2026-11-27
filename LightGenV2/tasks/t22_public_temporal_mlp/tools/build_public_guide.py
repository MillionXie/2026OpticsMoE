"""Arrange the author's existing optical figures while preserving Word formatting."""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import zipfile

from lxml import etree


PUBLIC = Path(__file__).resolve().parents[4] / 'LightGen-II'
NAMESPACES = {
    'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}


def qualified(name):
    prefix, local = name.split(':')
    return '{' + NAMESPACES[prefix] + '}' + local


def paragraph_text(element):
    return ''.join(element.xpath('.//w:t/text()', namespaces=NAMESPACES))


def replace_text(element, text):
    nodes = element.xpath('.//w:t', namespaces=NAMESPACES)
    if not nodes:
        raise ValueError('The existing caption has no text')
    nodes[0].text = text
    for node in nodes[1:]:
        node.text = ''


def property_value(paragraph, name, value):
    properties = paragraph.find('w:pPr', NAMESPACES)
    if properties is None:
        properties = etree.Element(qualified('w:pPr'))
        paragraph.insert(0, properties)
    element = properties.find('w:' + name, NAMESPACES)
    if element is None:
        element = etree.SubElement(properties, qualified('w:' + name))
    element.set(qualified('w:val'), value)


def image_paragraph(template, drawings, width_inches=3.3, page_break=False):
    paragraph = deepcopy(template)
    for attribute in list(paragraph.attrib):
        if etree.QName(attribute).localname in ('paraId', 'textId'):
            del paragraph.attrib[attribute]
    for element in list(paragraph):
        if element.tag != qualified('w:pPr'):
            paragraph.remove(element)
    property_value(paragraph, 'jc', 'center')
    property_value(paragraph, 'keepNext', '1')
    property_value(paragraph, 'keepLines', '1')
    property_value(paragraph, 'pageBreakBefore', '1' if page_break else '0')
    for index, original in enumerate(drawings):
        if index:
            run = etree.SubElement(paragraph, qualified('w:r'))
            space = etree.SubElement(run, qualified('w:t'))
            space.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            space.text = ' '
        drawing = deepcopy(original)
        extent = drawing.find('.//wp:extent', NAMESPACES)
        ratio = int(extent.get('cy')) / int(extent.get('cx'))
        width = round(width_inches * 914400)
        height = round(width * ratio)
        for size in drawing.xpath('.//wp:extent | .//a:xfrm/a:ext', namespaces=NAMESPACES):
            size.set('cx', str(width))
            size.set('cy', str(height))
        run = etree.SubElement(paragraph, qualified('w:r'))
        run.append(drawing)
    return paragraph


def build_guide(path, project=PUBLIC):
    path = Path(path)
    original = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        parts = {name:archive.read(name) for name in archive.namelist()}
    root = etree.fromstring(parts['word/document.xml'])
    body = root.find('w:body', NAMESPACES)
    elements = list(body)
    relations = etree.fromstring(parts['word/_rels/document.xml.rels'])
    targets = {relation.get('Id'):'word/' + relation.get('Target') for relation in relations
               if relation.get('Type', '').endswith('/image')}
    drawings = {}
    image_elements = []
    for element in elements:
        figures = element.xpath('.//w:drawing', namespaces=NAMESPACES)
        if figures:
            image_elements.append(element)
        for drawing in figures:
            identity = drawing.find('.//a:blip', NAMESPACES).get(qualified('r:embed'))
            digest = hashlib.sha256(parts[targets[identity]]).hexdigest()
            if digest in drawings:
                raise ValueError('A repeated figure requires manual identification')
            drawings[digest] = drawing
    if len(drawings) != 6:
        raise ValueError('Expected the six existing optical figures')
    captions = {branch:next(element for element in elements
                if paragraph_text(element).startswith(branch.capitalize() + ' input field'))
                for branch in ('vision','language')}
    first = elements.index(image_elements[0])
    last = max(elements.index(caption) for caption in captions.values())
    allowed = {id(element) for element in image_elements + list(captions.values())}
    if any(id(element) not in allowed and paragraph_text(element) for element in elements[first:last+1]):
        raise ValueError('Unexpected text between the optical figures')
    replacement = []
    for branch, number in (('vision',5),('language',8)):
        ordered = []
        for offset, kind in enumerate(('input_field','mask','output_field')):
            source = project/'reference_results'/f'{number+offset:02d}_{branch}_{kind}.png'
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest not in drawings:
                raise ValueError('The embedded figure differs from its source ' + source.name)
            ordered.append(drawings[digest])
        upper = image_paragraph(image_elements[0], ordered[:2], page_break=branch == 'language')
        lower = image_paragraph(image_elements[0], ordered[2:])
        caption = deepcopy(captions[branch])
        replace_text(caption, branch.capitalize() + ' input field and phase mask (top row), and output field (bottom).')
        property_value(caption, 'keepNext', '0')
        property_value(caption, 'keepLines', '1')
        replacement.extend((upper, lower, caption))
    for element in elements[first:last+1]:
        body.remove(element)
    for offset, element in enumerate(replacement):
        body.insert(first+offset, element)
    introduction = next(element for element in body
                        if paragraph_text(element).startswith('The same execution saves the input video frames'))
    text = paragraph_text(introduction).replace('in the top row','on the first figure page')
    text = text.replace('in the bottom row','on the second figure page')
    if text != paragraph_text(introduction):
        replace_text(introduction, text)
    property_value(introduction, 'keepNext', '1')
    for table in body.findall('w:tbl', NAMESPACES):
        for row in table.findall('w:tr', NAMESPACES):
            if '03_prediction_vs_mos.png' in paragraph_text(row):
                table.remove(row)
    parts['word/document.xml'] = etree.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
    temporary = path.with_name('.' + path.name + '.editing')
    with zipfile.ZipFile(temporary, 'w') as archive:
        for entry in entries:
            archive.writestr(entry, parts[entry.filename])
    if path.read_bytes() != original:
        temporary.unlink()
        raise RuntimeError('The author edited the guide during figure arrangement')
    temporary.replace(path)
    print('Updated ' + str(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document', type=Path, default=PUBLIC/'README.docx')
    parser.add_argument('--project', type=Path, default=PUBLIC)
    args = parser.parse_args()
    build_guide(args.document, args.project)
