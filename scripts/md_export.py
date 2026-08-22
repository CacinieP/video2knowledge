#!/usr/bin/env python3
"""md_export.py — Markdown → DOCX / PDF exporter for video2knowledge artifacts.

Converts the pipeline's own generated Markdown (knowledge.md, notes.md — a
known, stable subset: headings, tables, lists, blockquotes, bold/italic/code
spans, links, standalone images, horizontal rules, fenced code) into shareable
office/print formats:

    python3 scripts/md_export.py --md run/knowledge.md --out run/knowledge.pdf
    python3 scripts/md_export.py --md run/notes.md --out run/notes.docx

Backends are pure-python and pip/uv-installable (no LaTeX, no Word, no native
libs): python-docx for .docx, fpdf2 for .pdf. CJK output needs a Unicode font
for PDF (auto-detected per OS; override with V2K_PDF_FONT=/path/font.ttf).
Images referenced by relative path are resolved against the .md file's
directory (base_dir) and embedded. Missing optional deps degrade with a clear
install hint; used by build_knowledge.py (--format docx/pdf) and
build_notes.py (--docx/--pdf).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# --- markdown block parser -----------------------------------------------------

_BLOCK_TYPES = ("h", "hr", "quote", "ul", "ol", "code", "table", "p", "img")


def parse_markdown(md: str) -> list[dict]:
    """Parse the pipeline's Markdown subset into typed blocks.

    Inline markup inside text blocks is kept verbatim (each renderer applies
    its own inline handling). Images are only recognized as STANDALONE lines
    (how build_notes emits them); inline images inside paragraphs are dropped.
    """
    blocks: list[dict] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            blocks.append({"t": "h", "level": len(m.group(1)), "text": m.group(2)})
            i += 1
            continue
        if re.match(r"^-{3,}$|^\*{3,}$|^_{3,}$", s):
            blocks.append({"t": "hr"})
            i += 1
            continue
        if s.startswith("```"):
            lang = s[3:].strip()
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # closing fence
            blocks.append({"t": "code", "lang": lang, "lines": buf})
            continue
        if s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            blocks.append({"t": "quote", "lines": buf})
            continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(
                r"^\s*\|[\s:|-]+\|?\s*$", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            blocks.append({"t": "table", "rows": [rows[0]] + rows[2:]})  # keep header, drop separator
            continue
        if re.match(r"^[-*]\s+", s):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            blocks.append({"t": "ul", "items": items})
            continue
        if re.match(r"^\d+[.)]\s+", s):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+[.)]\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+[.)]\s+", "", lines[i]))
                i += 1
            blocks.append({"t": "ol", "items": items})
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", s)
        if m:
            blocks.append({"t": "img", "alt": m.group(1), "path": m.group(2)})
            i += 1
            continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{1,6}\s|>|-{3,}$|\||[-*]\s|\d+[.)]\s|```|!\[)", lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        blocks.append({"t": "p", "text": " ".join(buf)})
    return blocks


# --- inline spans: [(text, bold, italic, code), ...] ---------------------------

_INLINE_RE = re.compile(r"(\*\*(?=\S)(.+?)(?<=\S)\*\*|\*(?=\S)([^*]+?)(?<=\S)\*|`([^`]+)`|\[(?=[^\]]+\]\()[^\]]+\]\(([^)]+)\))")


def inline_spans(text: str) -> list[tuple[str, bool, bool, bool]]:
    """Split a line into (text, bold, italic, code) spans.

    Links render as their label (URLs would clutter print output); the URL of
    bare links `[label](url)` is dropped when label != url, kept otherwise.
    """
    spans: list[tuple[str, bool, bool, bool]] = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            spans.append((text[pos:m.start()], False, False, False))
        g = m.group(0)
        if g.startswith("**"):
            spans.append((m.group(2), True, False, False))
        elif g.startswith("`"):
            spans.append((m.group(4), False, False, True))
        elif g.startswith("["):
            label = g[g.index("[") + 1:g.index("](")]
            url = m.group(5)
            spans.append((label if label and label != url else url, False, False, False))
        else:
            spans.append((m.group(3), False, True, False))
        pos = m.end()
    if pos < len(text):
        spans.append((text[pos:], False, False, False))
    return spans or [(text, False, False, False)]


# --- DOCX backend ---------------------------------------------------------------

def _jpeg_strip_comments(data: bytes) -> bytes:
    """Remove COM (0xFFFE) segments from a JPEG byte stream.

    ffmpeg 8.x writes an encoder-version comment right after SOI. Comments are
    non-semantic — safe to drop.
    """
    if data[:2] != b"\xff\xd8":
        return data
    out = bytearray(data[:2])
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            out += data[pos:]  # not a marker boundary — keep the rest as-is
            return bytes(out)
        marker = data[pos + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            out += data[pos:pos + 2]
            pos += 2
            continue
        seglen = (data[pos + 2] << 8) | data[pos + 3]
        end = pos + 2 + seglen
        if marker == 0xDA or end > len(data):  # SOS: entropy data follows
            out += data[pos:]
            return bytes(out)
        if marker != 0xFE:  # keep everything except comments
            out += data[pos:end]
        pos = end
    out += data[pos:]
    return bytes(out)


# minimal JFIF APP0 segment (version 1.1, no density, no thumbnail)
_JFIF_APP0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"


def _jpeg_normalize(data: bytes) -> bytes:
    """Make an ffmpeg-extracted JPEG acceptable to python-docx:

    1. strip COM (0xFFFE) comment segments (encoder version string);
    2. if no JFIF APP0 follows SOI, insert a minimal one — ffmpeg 8's mjpeg
       encoder emits no APP0, and python-docx's signature sniffing requires
       the literal 'JFIF'/'Exif' at offset 6 or it raises
       UnrecognizedImageError. A standard APP0 is valid for every consumer.
    """
    data = _jpeg_strip_comments(data)
    if data[:2] != b"\xff\xd8" or data[2:4] in (b"\xff\xe0", b"\xff\xe1"):
        return data
    return data[:2] + _JFIF_APP0 + data[2:]


def md_to_docx(md: str, out_path: Path, base_dir: Path | None = None) -> Path:
    """Render markdown to .docx via python-docx. base_dir resolves image paths."""
    try:
        import docx  # noqa: F401  (python-docx)
        from docx import Document
        from docx.shared import Inches, Pt
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
    except ImportError:
        raise RuntimeError("python-docx not installed — run "
                           "`pip install python-docx` (or the uv equivalent)")

    doc = Document()
    base_dir = base_dir or Path(".")

    def add_runs(par, text: str):
        for txt, bold, italic, code in inline_spans(text):
            r = par.add_run(txt)
            r.bold, r.italic = bold, italic
            if code:
                r.font.name = "Consolas"
                r._element.rPr.rFonts.set(qn("w:eastAsia"), "等线")

    def add_hr(par):
        pPr = par._p.get_or_add_pPr()
        pBdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single"), bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:space"), "1"), bottom.set(qn("w:color"), "BBBBBB")
        pBdr.append(bottom)
        pPr.append(pBdr)

    for b in parse_markdown(md):
        t = b["t"]
        if t == "h":
            h = doc.add_heading("", level=min(b["level"], 4))
            add_runs(h, b["text"])
        elif t == "hr":
            add_hr(doc.add_paragraph())
        elif t == "quote":
            p = doc.add_paragraph(style="Intense Quote")
            add_runs(p, "\n".join(b["lines"]))
        elif t in ("ul", "ol"):
            for item in b["items"]:
                p = doc.add_paragraph(style="List Bullet" if t == "ul" else "List Number")
                add_runs(p, item)
        elif t == "code":
            for ln in b["lines"]:
                p = doc.add_paragraph()
                r = p.add_run(ln if ln else " ")
                r.font.name = "Consolas"
                r.font.size = Pt(9)
        elif t == "table":
            rows = b["rows"]
            if not rows:
                continue
            ncols = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=ncols)
            table.style = "Table Grid"
            for ri, row in enumerate(rows):
                for ci in range(ncols):
                    cell = table.cell(ri, ci)
                    par = cell.paragraphs[0]
                    txt = row[ci] if ci < len(row) else ""
                    if ri == 0:  # header row bold
                        for _txt, _b, _i, _c in inline_spans(txt):
                            r = par.add_run(_txt)
                            r.bold = True
                    else:
                        add_runs(par, txt)
            doc.add_paragraph()  # breathing room after tables
        elif t == "img":
            p = (base_dir / b["path"]) if not Path(b["path"]).is_absolute() else Path(b["path"])
            if p.is_file():
                import io
                blob = _jpeg_normalize(p.read_bytes())
                doc.add_picture(io.BytesIO(blob), width=Inches(5.9))
            else:
                doc.add_paragraph(f"(图片缺失: {b['path']})")
        else:
            add_runs(doc.add_paragraph(), b["text"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path


# --- PDF backend -----------------------------------------------------------------

_FONT_CANDIDATES = [
    # Windows
    r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\deng.ttf", r"C:\Windows\Fonts\simsun.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    # Linux (Noto CJK / WenQuanYi)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
]
_BOLD_HINTS = {"msyh.ttc": "msyhbd.ttc", "simhei.ttf": None,
               "simsun.ttc": None, "deng.ttf": "dengb.ttf",
               "PingFang.ttc": None, "Songti.ttc": None}


def find_cjk_font() -> tuple[str | None, str | None]:
    """Locate (regular, bold) CJK-capable font files. V2K_PDF_FONT overrides."""
    env = os.environ.get("V2K_PDF_FONT")
    cands = [env] if env else _FONT_CANDIDATES
    for c in cands:
        if c and Path(c).is_file():
            bold = None
            hint = _BOLD_HINTS.get(Path(c).name)
            if hint:
                bp = Path(c).parent / hint
                bold = str(bp) if bp.is_file() else None
            return c, bold
    return None, None


def md_to_pdf(md: str, out_path: Path, base_dir: Path | None = None,
              title: str = "知识文档") -> Path:
    """Render markdown to .pdf via fpdf2 (CJK font auto-detected & embedded)."""
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError("fpdf2 not installed — run "
                           "`pip install fpdf2` (or the uv equivalent)")

    reg, bold = find_cjk_font()
    if not reg:
        raise SystemExit("[err] no CJK font found for PDF export. Set "
                         "V2K_PDF_FONT=/path/to/font.ttf|ttc (e.g. msyh.ttc)")
    base_dir = base_dir or Path(".")
    pdf = FPDF(format="A4")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.add_font("cjk", "", reg)
    pdf.add_font("cjk", "B", bold or reg)  # faux-bold fallback: same file
    pdf.add_font("cjk", "I", bold or reg)

    def emit(text: str, size=10.5, style="", color=(33, 37, 41), indent=0,
             lspace=2.2, align="L"):
        pdf.set_font("cjk", style, size)
        pdf.set_text_color(*color)
        if indent:
            pdf.set_x(pdf.l_margin + indent)
        # markdown=True lets fpdf2 honor **bold** spans with the B font
        pdf.multi_cell(pdf.epw - indent, size * 0.52 + lspace * 0.5, text,
                       align=align, markdown=True)
        pdf.ln(lspace)

    for b in parse_markdown(md):
        t = b["t"]
        if t == "h":
            lvl = b["level"]
            emit(b["text"], size={1: 17, 2: 13.5, 3: 12, 4: 11}.get(lvl, 10.5),
                 style="B", color=(10, 37, 64), lspace=3)
        elif t == "hr":
            pdf.ln(2)
            y = pdf.get_y()
            pdf.set_draw_color(187, 187, 187)
            pdf.line(pdf.l_margin, y, pdf.l_margin + pdf.epw, y)
            pdf.ln(4)
        elif t == "quote":
            for ln in b["lines"]:
                emit(ln, size=9.5, color=(85, 95, 105), indent=5)
        elif t in ("ul", "ol"):
            for k, item in enumerate(b["items"], 1):
                marker = "•  " if t == "ul" else f"{k}. "
                emit(marker + item, indent=3)
        elif t == "code":
            for ln in b["lines"]:
                pdf.set_font("cjk", "", 9)
                pdf.set_text_color(43, 43, 43)
                pdf.set_fill_color(243, 244, 246)
                pdf.multi_cell(pdf.epw - 4, 5, " " + (ln or " "), fill=True)
            pdf.ln(2)
        elif t == "table":
            rows = b["rows"]
            if not rows:
                continue
            pdf.set_font("cjk", "", 9)
            pdf.set_text_color(33, 37, 41)
            pdf.set_draw_color(208, 212, 216)
            with pdf.table(line_height=5.4, padding=1.2, num_heading_rows=1) as table:
                for row in rows:
                    tr = table.row()
                    for cell in row:
                        tr.cell(cell)
            pdf.ln(3)
        elif t == "img":
            p = (base_dir / b["path"]) if not Path(b["path"]).is_absolute() else Path(b["path"])
            if p.is_file():
                try:
                    pdf.image(str(p), w=pdf.epw)
                    pdf.ln(3)
                except Exception as e:  # unsupported/corrupt image — keep going
                    emit(f"(图片无法嵌入: {b['path']} — {e})", size=9, color=(160, 60, 60))
            else:
                emit(f"(图片缺失: {b['path']})", size=9, color=(160, 60, 60))
        else:
            emit(b["text"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    return out_path


# --- CLI ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Markdown -> DOCX/PDF exporter")
    ap.add_argument("--md", required=True, type=Path, help="input .md file")
    ap.add_argument("--out", required=True, type=Path,
                    help="output path (.docx or .pdf — extension picks the backend)")
    ap.add_argument("--title", default=None, help="PDF document title (default 知识文档)")
    args = ap.parse_args()
    if not args.md.is_file():
        print(f"[err] not found: {args.md}", file=sys.stderr)
        return 2
    md = args.md.read_text(encoding="utf-8")
    try:
        if args.out.suffix.lower() == ".docx":
            p = md_to_docx(md, args.out, base_dir=args.md.parent)
        elif args.out.suffix.lower() == ".pdf":
            p = md_to_pdf(md, args.out, base_dir=args.md.parent, title=args.title or "知识文档")
        else:
            print("[err] --out must end in .docx or .pdf", file=sys.stderr)
            return 2
    except RuntimeError as e:
        print(f"[err] {e}", file=sys.stderr)
        return 3
    print(f"[ok] exported -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
