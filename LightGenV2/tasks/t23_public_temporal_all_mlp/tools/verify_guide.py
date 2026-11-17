"""Verify that the Word guide lists the actual standalone files and results."""
from pathlib import Path
import json
import zipfile

from docx import Document
from lxml import etree
from pypdf import PdfReader


REPO = Path(__file__).resolve().parents[4]
PROJECT = REPO/'LightGen2_MLP'
REFERENCE = REPO/'LightGen2/README.docx'


def verify():
    document = Document(PROJECT/'README.docx')
    expected = [
        'data/input_for_review.npz', 'data/reference_mos.csv', 'weights/LightGen2.pt',
        'model.py', 'infer.py', 'data.py', 'visualize.py', 'requirements.txt',
        'README.pdf', 'LICENSE and licenses/',
    ]
    first = [row.cells[0].text for row in document.tables[0].rows]
    if first != expected:
        raise ValueError('The Files table differs from the actual project')
    for name in first[:-1]:
        if not (PROJECT/name).is_file():
            raise FileNotFoundError(name)
    if not (PROJECT/'LICENSE').is_file() or not (PROJECT/'licenses').is_dir():
        raise ValueError('License locations differ')
    results = [row.cells[0].text for row in document.tables[1].rows]
    actual = sorted(path.name for path in (PROJECT/'results').iterdir())
    if results != actual:
        raise ValueError('The generated Files table differs from the result folder')
    metrics = json.loads((PROJECT/'results/02_metrics.json').read_text('utf-8'))
    reader = PdfReader(PROJECT/'README.pdf')
    text = '\n'.join(page.extract_text() for page in reader.pages)
    for name in expected[:-1]+results:
        if name not in text:
            raise ValueError('A listed file is missing from the PDF '+name)
    if 'SRCC of '+format(metrics['SRCC'], '.2f') not in text:
        raise ValueError('The guide score differs from the saved metrics')
    with zipfile.ZipFile(REFERENCE) as original, zipfile.ZipFile(PROJECT/'README.docx') as current:
        editable = {'word/document.xml', 'word/_rels/document.xml.rels',
                    'word/settings.xml', 'docProps/core.xml'}
        for name in original.namelist():
            if name in editable or name.startswith('word/media/'):
                continue
            if original.read(name) != current.read(name):
                raise ValueError('A preserved template part differs '+name)
        namespace = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        original_body = etree.fromstring(original.read('word/document.xml'))
        current_body = etree.fromstring(current.read('word/document.xml'))
        if etree.tostring(original_body.find('.//w:sectPr', namespace)) != etree.tostring(current_body.find('.//w:sectPr', namespace)):
            raise ValueError('The page geometry differs from the reference')
        original_images = [name for name in original.namelist() if name.startswith('word/media/')]
        current_images = [name for name in current.namelist() if name.startswith('word/media/')]
        if len(current_images) != 2:
            raise ValueError('Expected the scatter and input-frame preview in the guide')
    report = dict(PDF_pages=len(reader.pages), source_file_rows=len(first),
                  result_file_rows=len(results), template_parts_preserved=True,
                  template_images=len(original_images), guide_images=len(current_images),
                  SRCC=metrics['SRCC'])
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    verify()
