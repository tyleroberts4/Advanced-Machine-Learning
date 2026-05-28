"""Convert Rec_Center_Final_Report.md to PDF using fpdf2."""

import re
import urllib.request
import tempfile
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos

REPORT_DIR = Path(__file__).parent
FIGURES_DIR = REPORT_DIR.parent / "figures"
MD_FILE = REPORT_DIR / "Rec_Center_Final_Report.md"
OUT_FILE = REPORT_DIR / "Rec_Center_Final_Report.pdf"

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BLACK = (30, 30, 30)
DARK_BLUE = (31, 73, 125)
MID_BLUE = (52, 120, 190)
LIGHT_GRAY = (245, 245, 245)
RULE_COLOR = (180, 180, 180)

# Unicode replacements so we can use built-in Helvetica (latin-1 range)
UNICODE_MAP = {
    "—": "-",   # em dash
    "–": "-",   # en dash
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "é": "e",
    "≈": "~",
}

def sanitize(text: str) -> str:
    for ch, rep in UNICODE_MAP.items():
        text = text.replace(ch, rep)
    return text

MARGIN = 20
PAGE_W = 210  # A4 mm
CONTENT_W = PAGE_W - 2 * MARGIN


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*RULE_COLOR)
        self.cell(0, 8, "Cal Poly Rec Center Usage Prediction - Final Report", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*RULE_COLOR)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(2)

    def footer(self):
        pass


def add_cover(pdf: ReportPDF):
    """Full-page styled cover."""
    pdf.add_page()
    # Top accent bar
    pdf.set_fill_color(*DARK_BLUE)
    pdf.rect(0, 0, PAGE_W, 12, "F")

    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*DARK_BLUE)
    pdf.multi_cell(CONTENT_W, 12, "Cal Poly Rec Center\nUsage Prediction", align="C",  # noqa
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_draw_color(*MID_BLUE)
    pdf.set_line_width(0.8)
    pdf.line(MARGIN + 20, pdf.get_y(), PAGE_W - MARGIN - 20, pdf.get_y())
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(*MID_BLUE)
    pdf.cell(CONTENT_W, 10, "Final Report", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(*BLACK)
    pdf.cell(CONTENT_W, 8, "Advanced Machine Learning Final Project", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(60)

    # Bottom bar
    pdf.set_fill_color(*DARK_BLUE)
    pdf.rect(0, 277, PAGE_W, 20, "F")


def render_inline(pdf: FPDF, text: str, base_font_size: float, color):
    """Render a line that may contain **bold** and/or *italic* spans."""
    segments = re.split(r'(\*\*.*?\*\*|\*.*?\*)', text)
    pdf.set_text_color(*color)
    for seg in segments:
        if seg.startswith("**") and seg.endswith("**"):
            pdf.set_font("Helvetica", "B", base_font_size)
            pdf.write(base_font_size * 0.45, seg[2:-2])
        elif seg.startswith("*") and seg.endswith("*"):
            pdf.set_font("Helvetica", "I", base_font_size)
            pdf.write(base_font_size * 0.45, seg[1:-1])
        else:
            pdf.set_font("Helvetica", "", base_font_size)
            pdf.write(base_font_size * 0.45, seg)


def add_rule(pdf: FPDF):
    pdf.set_draw_color(*RULE_COLOR)
    pdf.set_line_width(0.3)
    pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
    pdf.ln(4)


def parse_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows) from markdown table lines."""
    headers: list[str] = []
    rows: list[list[str]] = []
    for line in lines:
        if re.match(r"^\s*\|[-:| ]+\|\s*$", line):
            continue  # separator
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not headers:
            headers = cells
        else:
            rows.append(cells)
    return headers, rows


def draw_table(pdf: FPDF, headers: list[str], rows: list[list[str]]):
    col_count = len(headers)
    col_w = CONTENT_W / col_count

    # Header row
    pdf.set_fill_color(*DARK_BLUE)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    for h in headers:
        clean = re.sub(r"\*\*|\*", "", h)
        pdf.cell(col_w, 7, clean, border=0, align="C", fill=True)
    pdf.ln()

    # Data rows
    for i, row in enumerate(rows):
        fill = i % 2 == 0
        pdf.set_fill_color(*LIGHT_GRAY) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(*BLACK)
        pdf.set_font("Helvetica", "", 9)
        for cell in row:
            clean = re.sub(r"\*\*|\*", "", cell)
            pdf.cell(col_w, 6, clean, border=0, align="C", fill=True)
        pdf.ln()
    pdf.ln(3)


def add_figure(pdf: FPDF, img_ref: str):
    """Embed a figure if the path resolves."""
    match = re.search(r'\(([^)]+\.png)\)', img_ref)
    if not match:
        return
    rel = match.group(1).replace("../figures/", "")
    img_path = FIGURES_DIR / rel
    if not img_path.exists():
        return
    max_w = CONTENT_W - 10
    pdf.ln(2)
    # Check remaining space; add page if < 60 mm
    if pdf.get_y() > 230:
        pdf.add_page()
    pdf.image(str(img_path), x=MARGIN + 5, w=max_w)
    pdf.ln(4)


def build_pdf(md_text: str):
    pdf = ReportPDF("P", "mm", "A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(MARGIN, 15, MARGIN)

    add_cover(pdf)
    pdf.add_page()

    md_text = sanitize(md_text)
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Horizontal rule
        if re.match(r"^---+\s*$", line):
            pdf.ln(2)
            add_rule(pdf)
            i += 1
            continue

        # H1
        if line.startswith("# ") and not line.startswith("## "):
            # Already on cover; skip the title line in body
            i += 1
            continue

        # H2
        if line.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(*DARK_BLUE)
            pdf.set_fill_color(*LIGHT_GRAY)
            text = line[3:].strip()
            pdf.cell(CONTENT_W, 9, text, align="L", fill=True,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
            i += 1
            continue

        # H3
        if line.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*MID_BLUE)
            pdf.cell(CONTENT_W, 7, line[4:].strip(),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            i += 1
            continue

        # Table — collect all consecutive table lines
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            headers, rows = parse_table(table_lines)
            draw_table(pdf, headers, rows)
            continue

        # Figure
        if line.startswith("!["):
            add_figure(pdf, line)
            i += 1
            continue

        # Numbered list item
        m = re.match(r"^(\d+)\.\s+(.*)", line)
        if m:
            num = m.group(1)
            text = m.group(2)
            pdf.set_x(MARGIN + 4)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*DARK_BLUE)
            pdf.cell(6, 5.5, f"{num}.")
            render_inline(pdf, text, 10, BLACK)
            pdf.ln(5.5)
            i += 1
            continue

        # Bullet list item
        if line.startswith("- "):
            text = line[2:].strip()
            pdf.set_x(MARGIN + 4)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*DARK_BLUE)
            pdf.cell(4, 5.5, "-")
            render_inline(pdf, text, 10, BLACK)
            pdf.ln(5.5)
            i += 1
            continue

        # Sub-bullet
        if line.startswith("  - "):
            text = line[4:].strip()
            pdf.set_x(MARGIN + 10)
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_text_color(*MID_BLUE)
            pdf.cell(4, 5, "–")
            render_inline(pdf, text, 9.5, BLACK)
            pdf.ln(5)
            i += 1
            continue

        # Bold-only line (e.g. **Advanced Machine Learning Final Project**)
        if line.startswith("**") and line.endswith("**") and "\n" not in line:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*BLACK)
            pdf.cell(CONTENT_W, 6, line[2:-2], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            i += 1
            continue

        # Italic note at bottom (starts with *)
        if line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            clean = re.sub(r'\[.*?\]\(.*?\)', lambda m: m.group(0).split(']')[0][1:], line[1:-1])
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(120, 120, 120)
            pdf.multi_cell(CONTENT_W, 5, clean, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
            i += 1
            continue

        # Blank line
        if line.strip() == "":
            pdf.ln(2)
            i += 1
            continue

        # Normal paragraph text
        pdf.set_x(MARGIN)
        render_inline(pdf, line.strip(), 10, BLACK)
        pdf.ln(5.5)
        i += 1

    pdf.output(str(OUT_FILE))
    print(f"PDF written to: {OUT_FILE}")


if __name__ == "__main__":
    md_text = MD_FILE.read_text(encoding="utf-8")
    build_pdf(md_text)
