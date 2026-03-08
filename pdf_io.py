# pdf_io.py
# PDF-Zugriff via PyMuPDF (fitz): Volltext und Layout-Extraktion.
# extract_text_from_pdf() liefert den Rohtext mit Seitenmarkern.
# extract_layout_from_pdf() teilt die Seite in Zonen (Header/Footer/Links/Rechts)
# – diese Zonen nutzt main.py um Absender/Empfänger besser zu lokalisieren.
# Wird nur von main.py aufgerufen.

import fitz


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    doc = fitz.open(file_path)
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if page_text:
            text += f"\n--- Seite {page_num} ---\n{page_text}\n"
    doc.close()
    return text


def extract_layout_from_pdf(file_path: str) -> dict:
    result = {
        "left_blocks": [],
        "right_blocks": [],
        "header_blocks": [],
        "footer_blocks": []
    }
    doc = fitz.open(file_path)
    for page in doc:
        w = page.rect.width
        h = page.rect.height
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, content = b[0], b[1], b[2], b[3], b[4]
            content = content.strip()
            if not content:
                continue
            # Obere 25% = Header, untere 20% = Footer, Rest nach links/rechts Seitenmitte
            if y0 < h * 0.25:
                result["header_blocks"].append(content)
            elif y1 > h * 0.80:
                result["footer_blocks"].append(content)
            elif x0 < w * 0.5:
                result["left_blocks"].append(content)
            else:
                result["right_blocks"].append(content)
    doc.close()
    return result
