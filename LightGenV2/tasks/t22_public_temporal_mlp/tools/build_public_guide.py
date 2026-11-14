"""Apply release formatting to the author's current editable Word guide."""
from pathlib import Path
import argparse
import hashlib

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph


PUBLIC = Path(__file__).resolve().parents[4] / 'LightGen2'
CAPTIONS = (
    'Predicted versus human temporal MOS for the 558 LGVQ videos (03_prediction_vs_mos.png).',
    'Vision input field, phase mask and output field, from left to right.',
    'Language input field, phase mask and output field, from left to right.',
)
LICENSE_STATEMENT = 'This project is covered under the GNU General Public License v3.0.'


def replace_text(paragraph, text):
    if paragraph.runs:
        paragraph.runs[0].text = text
        for run in paragraph.runs[1:]:
            run.text = ''
    else:
        paragraph.add_run(text)


def replace_cell(cell, text):
    replace_text(cell.paragraphs[0], text)
    for paragraph in cell.paragraphs[1:]:
        paragraph._p.getparent().remove(paragraph._p)


def format_command(paragraph):
    properties = paragraph._p.get_or_add_pPr()
    shading = properties.find(qn('w:shd'))
    if shading is None:
        shading = OxmlElement('w:shd')
        properties.append(shading)
    shading.set(qn('w:val'), 'clear')
    shading.set(qn('w:color'), 'auto')
    shading.set(qn('w:fill'), 'F2F2F2')
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(6)


def add_caption(paragraph, text, *, keep_with_next=False):
    following = paragraph._p.getnext()
    if following is not None and following.tag == qn('w:p'):
        existing = Paragraph(following, paragraph._parent)
        if existing.style.name == 'Caption':
            caption = existing
        else:
            caption = None
    else:
        caption = None
    if caption is None:
        element = OxmlElement('w:p')
        paragraph._p.addnext(element)
        caption = Paragraph(element, paragraph._parent)
        caption.style = 'Caption'
        caption.add_run(text)
    paragraph.paragraph_format.keep_with_next = True
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.keep_together = True
    caption.paragraph_format.keep_with_next = keep_with_next
    caption.paragraph_format.space_before = Pt(3)
    caption.paragraph_format.space_after = Pt(8)
    for run in caption.runs:
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10.5)
        run.font.italic = False
        run.font.bold = False
        run.font.color.rgb = RGBColor(0, 0, 0)


def add_page_numbers(document):
    for section in document.sections:
        section.footer_distance = Inches(.3)
        footer = section.footer
        fields = footer._element.xpath('.//w:fldSimple')
        if any(field.get(qn('w:instr'), '').strip().upper() == 'PAGE' for field in fields):
            continue
        paragraph = footer.paragraphs[0] if not any(p.text for p in footer.paragraphs) else footer.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        field = OxmlElement('w:fldSimple')
        field.set(qn('w:instr'), 'PAGE')
        field.set(qn('w:dirty'), 'true')
        run = OxmlElement('w:r')
        properties = OxmlElement('w:rPr')
        fonts = OxmlElement('w:rFonts')
        fonts.set(qn('w:ascii'), 'Times New Roman')
        fonts.set(qn('w:hAnsi'), 'Times New Roman')
        size = OxmlElement('w:sz')
        size.set(qn('w:val'), '18')
        properties.extend((fonts, size))
        run.append(properties)
        text = OxmlElement('w:t')
        text.text = '1'
        run.append(text)
        field.append(run)
        paragraph._p.append(field)


def build_guide(path):
    original = path.read_bytes()
    document = Document(path)
    for table in document.tables:
        for row in list(table.rows):
            name = row.cells[0].text
            if name == 'weights/LightGen2.pt':
                replace_cell(row.cells[1], 'Model weights.')
            elif name in ('README.md and README.docx', 'README.pdf'):
                replace_cell(row.cells[0], 'README.pdf')
                replace_cell(row.cells[1], 'Instructions for installing and running the code.')
            elif name == 'LICENSE, NOTICE and licenses/':
                table._tbl.remove(row._tr)
    commands = [paragraph for paragraph in document.paragraphs
                if paragraph.text.strip() in ('python -m pip install -r requirements.txt', 'python infer.py')]
    if len(commands) != 2:
        raise ValueError('The current guide requires the installation and inference commands')
    for paragraph in commands:
        format_command(paragraph)
    figures = [paragraph for paragraph in document.paragraphs if paragraph._p.xpath('.//w:drawing')]
    if len(figures) != len(CAPTIONS):
        raise ValueError('The current guide has an unexpected number of image groups')
    for number, (paragraph, caption) in enumerate(zip(figures, CAPTIONS)):
        add_caption(paragraph, caption, keep_with_next=number == 1)
    if not any(paragraph.text == 'License' and paragraph.style.name == 'Heading 1'
               for paragraph in document.paragraphs):
        document.add_heading('License', 1)
        document.add_paragraph(LICENSE_STATEMENT)
    add_page_numbers(document)
    temporary = path.with_name('.' + path.name + '.editing')
    document.save(temporary)
    if path.read_bytes() != original:
        temporary.unlink()
        raise RuntimeError('The editable guide changed during formatting')
    temporary.replace(path)
    print('Updated ' + str(path))
    print('SHA256 ' + hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document', type=Path, default=PUBLIC / 'README.docx')
    build_guide(parser.parse_args().document)
