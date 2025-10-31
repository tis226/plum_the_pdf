#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qa_gui_margins.py

GUI preview to pick default margins from a chosen page, then parse QA:
- Two-column crop (top/bottom/gutter fractions)
- Detect leftmost x within each column (red guide lines)
- Button: "Use this page's margins for ALL pages"
- Optional: Export margins.json
- Parse: question_text ends at first '①'; options split by circled numerals
- Detect leading question number in stem -> use as 'question_number'
- Strip circled index (①/②/…) from option 'text' (kept only in 'index')
- NEW: All text normalization collapses any whitespace (incl. newlines) to single spaces.
- NEW: Optional automatic subject detection from page headers (use --subject auto).
- NEW: Optional JPG previews for each extracted chunk (--chunk-preview-dir).

Usage:
  python qa_gui_margins.py --pdf "C:\\path\\file.pdf" --out "C:\\path\\qa.json" --subject auto --year 2025 --target default
  # optional knobs:
  --dpi 180 --tol 1.5 --top-frac 0.04 --bottom-frac 0.96 --gutter-frac 0.005
  --subject-map header_map.json
  --chunk-preview-dir chunk_previews --chunk-preview-dpi 180 --chunk-preview-pad 2.0
"""

import os, re, json, argparse, tempfile
from typing import List, Dict, Any, Optional

import pdfplumber

# GUI deps
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# ---------- Parsing regex/config ----------
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
OPT_SPLIT_RE = re.compile(rf"(?=([{CIRCLED}]))", re.UNICODE)
DISPUTE_RE   = re.compile(r"\(다툼이\s*있는\s*경우\s*판례에\s*의함\)", re.UNICODE)

# Detect a leading question number in the stem (before the first ①)
# Matches: "16.", "16 )", "(16)", "16번", "16 ." etc.
QUESTION_NUM_RE = re.compile(
    r"^\s*(?:\(\s*(\d{1,3})\s*\)|(\d{1,3})\s*번|(\d{1,3}))\s*[.)]?\s*",
    re.UNICODE
)

# For noisy sources, make sure option text never keeps the circled index
CIRCLED_STRIP_RE = re.compile(rf"^[{CIRCLED}]\s*")

# ---------- Header detection (for subject auto-detection) ----------
TARGET_SUBJECTS = [
    "경찰학개론",
    "헌법",
    "형사법",
    "경찰학",
    "형법",
    "형사소송법",
    "경찰학개론",
]
TARGET_SUBJECTS_CASEFOLD = [s.casefold() for s in TARGET_SUBJECTS if s]

HEADER_TOL = 1.0          # tolerance to consider a line horizontal
HEADER_MIN_FRAC = 0.5     # keep horizontals >= 50% of page width
HEADER_PAD_DOWN = 3.0     # include a tiny band *below* the rule so boundary glyphs aren't dropped
HEADER_X_TOL, HEADER_Y_TOL = 2, 4
HEADER_LINE_MERGE_TOL = 2.0


def _is_horizontal(line, tol=HEADER_TOL):
    return abs(line.get("y0", 0.0) - line.get("y1", 0.0)) <= tol


def words_in_bbox_toporigin(words, crop_rect_toporigin):
    """Filter words (already in top-origin coords) by intersection with a top-origin rect."""

    cx0, ctop, cx1, cbottom = crop_rect_toporigin
    keep = []
    for w in words:
        top = w.get("top")
        bottom = w.get("bottom")
        x0 = w.get("x0")
        x1 = w.get("x1")
        if None in (top, bottom, x0, x1):
            continue
        if not (bottom < ctop or top > cbottom or x1 < cx0 or x0 > cx1):
            keep.append(w)
    return keep


def words_to_text_in_reading_order(words, line_tol=HEADER_LINE_MERGE_TOL):
    """Sort words by line (top) then x0, and join them into lines."""

    if not words:
        return ""

    words = sorted(words, key=lambda w: (round(w.get("top", 0.0), 1), w.get("x0", 0.0)))
    lines = []
    current = []
    prev_top = None
    for w in words:
        top = w.get("top")
        if top is None:
            continue
        if prev_top is None or abs(top - prev_top) <= line_tol:
            current.append(w.get("text", ""))
        else:
            lines.append(" ".join(current))
            current = [w.get("text", "")]
        prev_top = top
    if current:
        lines.append(" ".join(current))
    return "\n".join(s.strip() for s in lines if s.strip())


def extract_header_text(page,
                        tol=HEADER_TOL,
                        min_frac=HEADER_MIN_FRAC,
                        pad_down=HEADER_PAD_DOWN,
                        x_tol=HEADER_X_TOL,
                        y_tol=HEADER_Y_TOL,
                        line_merge_tol=HEADER_LINE_MERGE_TOL):
    """Return text located above the topmost long horizontal rule on the page."""

    working_page = page.rotate(-page.rotation) if page.rotation else page
    width, height = float(working_page.width), float(working_page.height)

    candidates = [
        (max(line.get("y0", 0.0), line.get("y1", 0.0)), line)
        for line in (working_page.lines or [])
        if _is_horizontal(line, tol=tol) and abs(line.get("x1", 0.0) - line.get("x0", 0.0)) >= min_frac * width
    ]
    if not candidates:
        return ""

    y_div, _ = max(candidates, key=lambda t: t[0])
    y0_pdf = max(0.0, y_div - pad_down)
    bbox_pdf = (0, y0_pdf, width, height)

    words = working_page.extract_words(x_tolerance=x_tol, y_tolerance=y_tol, keep_blank_chars=False) or []
    crop_toporigin = {
        "x0": bbox_pdf[0],
        "top": height - bbox_pdf[3],
        "x1": bbox_pdf[2],
        "bottom": height - bbox_pdf[1],
    }

    words_crop = words_in_bbox_toporigin(
        words,
        (crop_toporigin["x0"], crop_toporigin["top"], crop_toporigin["x1"], crop_toporigin["bottom"])
    )

    return words_to_text_in_reading_order(words_crop, line_tol=line_merge_tol)


def detect_page_subject(page,
                        subject_keywords: Optional[Dict[str, str]] = None,
                        target_subjects: Optional[List[str]] = None):
    """Detect a page subject using the header text and optional keyword mapping.

    Returns (subject or None, raw_header_text).
    """

    header_text = norm_space(extract_header_text(page))
    if not header_text:
        return None, ""

    lowered = header_text.casefold()
    if subject_keywords:
        for key, value in subject_keywords.items():
            if not key:
                continue
            if key.casefold() in lowered:
                return value, header_text

    targets = target_subjects or TARGET_SUBJECTS
    target_lower = (
        [s.casefold() for s in targets if s]
        if target_subjects is not None
        else TARGET_SUBJECTS_CASEFOLD
    )
    for subj, subj_lower in zip(targets, target_lower):
        if not subj:
            continue
        if subj_lower in lowered:
            return subj, header_text

    return header_text, header_text

def norm_space(s: str) -> str:
    """
    Collapse ANY whitespace (including newlines/tabs) to single spaces and trim.
    """
    return re.sub(r"\s+", " ", (s or "").strip())

# ---------- Layout helpers ----------
def two_col_bboxes(page, top_frac=0.04, bottom_frac=0.96, gutter_frac=0.005):
    w, h = float(page.width), float(page.height)
    top = h * top_frac
    bottom = h * bottom_frac
    gutter = w * gutter_frac
    mid = w * 0.5
    return (0.0, top,  mid - gutter, bottom), (mid + gutter, top, w, bottom)

def extract_words_in_bbox(page, bbox, x_tol=3, y_tol=3):
    sub = page.within_bbox(bbox)
    return sub.extract_words(x_tolerance=x_tol, y_tolerance=y_tol,
                             keep_blank_chars=False, use_text_flow=True) or []

def group_words_into_lines(words: List[Dict[str, Any]], y_tol=3.0):
    if not words: return []
    ws = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines, cur, cur_top = [], [], None
    for w in ws:
        if cur_top is None or abs(w["top"] - cur_top) <= y_tol:
            cur.append(w)
            if cur_top is None: cur_top = w["top"]
        else:
            cur.sort(key=lambda x: x["x0"]); lines.append(cur)
            cur = [w]; cur_top = w["top"]
    if cur: cur.sort(key=lambda x: x["x0"]); lines.append(cur)
    return lines

def line_text(ln): return " ".join(w["text"] for w in ln).strip()
def line_x0(ln):   return min(w["x0"] for w in ln)

def lines_to_bbox(lines: List[List[Dict[str, Any]]], page_height: float):
    """Return a PDF-coordinate bbox (x0, top, x1, bottom) covering the provided lines."""

    xs0, xs1, tops, bottoms = [], [], [], []
    for ln in lines:
        for w in ln:
            x0 = w.get("x0")
            x1 = w.get("x1")
            top = w.get("top")
            bottom = w.get("bottom")
            if None in (x0, x1, top, bottom):
                continue
            xs0.append(float(x0))
            xs1.append(float(x1))
            tops.append(float(top))
            bottoms.append(float(bottom))

    if not xs0:
        return None

    min_x0 = min(xs0)
    max_x1 = max(xs1)

    # pdfplumber's words expose top/bottom as distances from the top edge; convert to PDF coords
    top_candidates = [page_height - b for b in bottoms]
    bottom_candidates = [page_height - t for t in tops]

    top_pdf = max(0.0, min(top_candidates))
    bottom_pdf = min(page_height, max(bottom_candidates))
    if top_pdf > bottom_pdf:
        top_pdf, bottom_pdf = bottom_pdf, top_pdf

    return (min_x0, top_pdf, max_x1, bottom_pdf)

def detect_col_leftmost(page, bbox) -> Optional[float]:
    sub = page.within_bbox(bbox)
    xs = []
    if sub.chars:
        xs = [c["x0"] for c in sub.chars if c.get("x0") is not None]
    if not xs:
        words = extract_words_in_bbox(page, bbox)
        xs = [w["x0"] for w in words if w.get("x0") is not None]
    if not xs: return None
    return float(min(xs))

# ---------- Margin-chunking + QA ----------
def chunk_column_by_margin(page, bbox, left_margin_x: Optional[float], tol: float,
                           y_tol=3.0) -> List[Dict[str, Any]]:
    if left_margin_x is None:
        return []
    words = extract_words_in_bbox(page, bbox)
    lines = group_words_into_lines(words, y_tol=y_tol)

    chunks, cur = [], []

    def flush():
        if not cur: return
        text = "\n".join(line_text(ln) for ln in cur).strip()
        if text:
            chunk_bbox = lines_to_bbox(cur, float(page.height))
            if chunk_bbox is None:
                chunk_bbox = bbox
            chunk = {"text": text}
            if chunk_bbox is not None:
                chunk["bbox"] = tuple(float(v) for v in chunk_bbox)
            chunks.append(chunk)
        cur.clear()

    for ln in lines:
        x0 = line_x0(ln)
        if abs(x0 - left_margin_x) <= tol:
            flush()
            cur.append(ln)
        else:
            if cur:
                cur.append(ln)
            else:
                pass
    flush()
    return chunks

def save_chunk_preview(page,
                       bbox,
                       preview_dir: str,
                       page_index: int,
                       column_tag: str,
                       column_chunk_idx: int,
                       global_idx: int,
                       dpi: int = 180,
                       pad: float = 2.0) -> Optional[str]:
    """Persist a JPG preview of the chunk defined by bbox. Returns absolute path or None."""

    if not bbox or not preview_dir:
        return None

    abs_dir = os.path.abspath(os.path.expanduser(preview_dir))
    try:
        os.makedirs(abs_dir, exist_ok=True)
    except OSError as exc:
        print(f"[warn] Failed to create preview directory '{abs_dir}': {exc}")
        return None

    width, height = float(page.width), float(page.height)
    x0, top, x1, bottom = map(float, bbox)
    pad = max(0.0, float(pad))
    padded = (
        max(0.0, x0 - pad),
        max(0.0, top - pad),
        min(width, x1 + pad),
        min(height, bottom + pad),
    )

    try:
        cropped_page = page.within_bbox(padded)
        page_image = cropped_page.to_image(resolution=int(dpi))
        pil = page_image.original.convert("RGB")
        del page_image
        filename = f"p{page_index:03d}_{column_tag}{column_chunk_idx:02d}_{global_idx:04d}.jpg"
        out_path = os.path.join(abs_dir, filename)
        pil.save(out_path, format="JPEG", quality=90)
        pil.close()
        return os.path.abspath(out_path)
    except Exception as exc:
        print(f"[warn] Failed to save preview for page {page_index} chunk {column_tag}{column_chunk_idx}: {exc}")
        return None

def extract_leading_qnum_and_clean(stem: str):
    """
    Return (qnum:int|None, cleaned_stem:str)
    Strips a leading question number token if present and returns it.
    """
    if not stem:
        return None, stem
    m = QUESTION_NUM_RE.match(stem)
    if not m:
        return None, stem
    digits = next((g for g in m.groups() if g), None)
    try:
        qnum = int(digits) if digits is not None else None
    except Exception:
        qnum = None
    cleaned = stem[m.end():].lstrip()
    return qnum, cleaned

def extract_qa_from_chunk_text(text: str):
    """
    Return (stem, options, dispute_bool, detected_qnum)
      - stem ends at first '①'
      - options split by circled numerals
      - detected_qnum comes from the beginning of the stem (e.g., '16.')
      - NEW: all whitespace collapsed to single spaces
    """
    if not text:
        return None, None, False, None

    first = text.find("①")
    if first == -1:
        return None, None, False, None

    stem = text[:first]
    opts_blob = text[first:]

    # dispute flag
    dispute = bool(DISPUTE_RE.search(stem))
    stem = DISPUTE_RE.sub("", stem)
    stem = norm_space(stem)

    # pull leading question number from stem (and remove it from stem)
    detected_qnum, stem = extract_leading_qnum_and_clean(stem)
    stem = norm_space(stem)  # normalize again after removing number

    # split options
    parts = [p for p in OPT_SPLIT_RE.split(opts_blob) if p]
    options = []
    i = 0
    while i < len(parts):
        sym = parts[i].strip()
        if sym and sym[0] in CIRCLED:
            raw_txt = parts[i+1] if (i+1) < len(parts) else ""
            # strip circled index from the option text; keep only in 'index'
            clean_txt = norm_space(CIRCLED_STRIP_RE.sub("", raw_txt))
            options.append({"index": sym[0], "text": clean_txt})
            i += 2
        else:
            i += 1

    options = [o for o in options if o["index"] in CIRCLED]
    if not options:
        return None, None, dispute, detected_qnum

    return stem, options, dispute, detected_qnum

def pdf_to_qa_margin_chunked(pdf_path: str,
                             subject: Optional[str], year: int, target: str,
                             start_num: int,
                             L_margin: Optional[float], R_margin: Optional[float],
                             tol: float,
                             top_frac=0.04, bottom_frac=0.96, gutter_frac=0.005,
                             y_tol=3.0,
                             auto_subject: bool = False,
                             subject_keywords: Optional[Dict[str, str]] = None,
                             chunk_preview_dir: Optional[str] = None,
                             chunk_preview_dpi: int = 180,
                             chunk_preview_pad: float = 2.0) -> List[Dict[str, Any]]:
    out = []
    qnum = start_num
    preview_dir = (os.path.abspath(os.path.expanduser(chunk_preview_dir))
                   if chunk_preview_dir else None)
    global_chunk_idx = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            effective_subject = subject
            if auto_subject or subject is None:
                detected, _header_text = detect_page_subject(page, subject_keywords=subject_keywords)
                if detected:
                    effective_subject = detected

            Lbbox, Rbbox = two_col_bboxes(page, top_frac, bottom_frac, gutter_frac)
            L_chunks = chunk_column_by_margin(page, Lbbox, L_margin, tol, y_tol=y_tol) if L_margin is not None else []
            R_chunks = chunk_column_by_margin(page, Rbbox, R_margin, tol, y_tol=y_tol) if R_margin is not None else []
            # reading order: L then R per page
            for column_tag, column_chunks in (("L", L_chunks), ("R", R_chunks)):
                for column_chunk_idx, ch in enumerate(column_chunks, start=1):
                    stem, options, dispute, detected_qnum = extract_qa_from_chunk_text(ch["text"])
                    if stem is None or not options:
                        continue

                    # Prefer detected number from the stem; otherwise use running counter
                    qno = detected_qnum if detected_qnum is not None else qnum

                    global_chunk_idx += 1
                    preview_path = None
                    if preview_dir and ch.get("bbox"):
                        preview_path = save_chunk_preview(
                            page,
                            ch.get("bbox"),
                            preview_dir,
                            page_index,
                            column_tag,
                            column_chunk_idx,
                            global_chunk_idx,
                            dpi=chunk_preview_dpi,
                            pad=chunk_preview_pad,
                        )

                    out.append({
                        "subject": (effective_subject if effective_subject is not None else ""),
                        "year": year,
                        "target": target,
                        "content": {
                            "question_number": qno,
                            "question_text": stem,  # already normalized (no newlines)
                            "dispute_bool": bool(dispute),
                            "dispute_site": None,
                            "options": options,     # each option text normalized too
                            "source": {
                                "page": page_index,
                                "column": column_tag,
                                "chunk_index": column_chunk_idx,
                            }
                        }
                    })

                    if preview_path:
                        out[-1]["content"]["preview_image"] = preview_path

                    # advance running counter only if we didn't detect a number
                    if detected_qnum is None:
                        qnum += 1
    return out

# ---------- GUI ----------
class MarginPreviewApp(tk.Tk):
    def __init__(self, pdf_path: str, out_path: str,
                 subject: Optional[str], year: int, target: str,
                 dpi=180, tol=1.5, top_frac=0.04, bottom_frac=0.96, gutter_frac=0.005,
                 auto_subject: bool = False,
                 subject_keywords: Optional[Dict[str, str]] = None,
                 chunk_preview_dir: Optional[str] = None,
                 chunk_preview_dpi: int = 180,
                 chunk_preview_pad: float = 2.0):
        super().__init__()
        self.title("QA Margin Preview & Parser")
        self.geometry("1200x850")
        self.minsize(900, 700)

        if isinstance(subject, str) and subject.strip().lower() == "auto":
            subject = None

        self.pdf_path = pdf_path
        self.out_path = out_path
        self.subject = subject
        self.year = year
        self.target = target
        self.auto_subject = auto_subject or (subject is None)
        self.subject_keywords = subject_keywords or {}

        self.chunk_preview_dir = (os.path.abspath(os.path.expanduser(chunk_preview_dir))
                                  if chunk_preview_dir else None)
        self.chunk_preview_dpi = max(1, int(chunk_preview_dpi))
        self.chunk_preview_pad = max(0.0, float(chunk_preview_pad))

        self.dpi = dpi
        self.tol = tk.DoubleVar(value=float(tol))
        self.top_frac = tk.DoubleVar(value=float(top_frac))
        self.bottom_frac = tk.DoubleVar(value=float(bottom_frac))
        self.gutter_frac = tk.DoubleVar(value=float(gutter_frac))

        self.L_margin = None
        self.R_margin = None

        self.tk_img = None
        self._tmp_png = None
        self.current_idx = 0

        # Layout
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # Left control panel
        panel = ttk.Frame(self, padding=8)
        panel.grid(row=0, column=0, sticky="ns")

        ttk.Label(panel, text="Navigation").pack(anchor="w", pady=(0,4))
        nav = ttk.Frame(panel); nav.pack(anchor="w", pady=(0,8))
        ttk.Button(nav, text="◀ Prev", command=self.prev_page, width=10).pack(side="left")
        ttk.Button(nav, text="Next ▶", command=self.next_page, width=10).pack(side="left", padx=(6,0))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(panel, text="Column crop (fractions)").pack(anchor="w", pady=(0,4))
        cf = ttk.Frame(panel); cf.pack(anchor="w", pady=(0,8))
        ttk.Label(cf, text="Top").grid(row=0, column=0, sticky="w")
        ttk.Entry(cf, textvariable=self.top_frac, width=7).grid(row=0, column=1, padx=(4,0))
        ttk.Label(cf, text="Bottom").grid(row=1, column=0, sticky="w")
        ttk.Entry(cf, textvariable=self.bottom_frac, width=7).grid(row=1, column=1, padx=(4,0))
        ttk.Label(cf, text="Gutter").grid(row=2, column=0, sticky="w")
        ttk.Entry(cf, textvariable=self.gutter_frac, width=7).grid(row=2, column=1, padx=(4,0))
        ttk.Button(cf, text="Apply crop", command=self.refresh).grid(row=3, column=0, columnspan=2, pady=(6,0), sticky="ew")

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(panel, text="Margin tolerance (pt)").pack(anchor="w", pady=(0,4))
        ttk.Entry(panel, textvariable=self.tol, width=8).pack(anchor="w")
        ttk.Label(panel, text="Detected margins").pack(anchor="w", pady=(8,4))
        self.marg_label = ttk.Label(panel, text="L: —   R: —")
        self.marg_label.pack(anchor="w")

        ttk.Button(panel, text="Use this page's margins for ALL pages", command=self.assign_margins, width=28).pack(anchor="w", pady=(10,0))
        ttk.Button(panel, text="Export margins.json…", command=self.export_margins, width=28).pack(anchor="w", pady=(6,0))
        ttk.Button(panel, text="Parse → QA JSON", command=self.parse_all, width=28).pack(anchor="w", pady=(12,0))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(panel, text="Output").pack(anchor="w")
        self.out_path_label = ttk.Label(panel, text=os.path.abspath(self.out_path), wraplength=220)
        self.out_path_label.pack(anchor="w")

        ttk.Label(panel, text="Chunk preview JPG dir").pack(anchor="w", pady=(8,0))
        preview_label_text = self.chunk_preview_dir if self.chunk_preview_dir else "(disabled)"
        self.preview_dir_label = ttk.Label(panel, text=preview_label_text, wraplength=220)
        self.preview_dir_label.pack(anchor="w")

        # Canvas
        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
               # ... snip highlight thickness kept ...
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.canvas.bind("<Configure>", lambda e: self.render_page())

        # Status
        status = ttk.Frame(self, padding=8)
        status.grid(row=1, column=0, columnspan=2, sticky="ew")
        status.columnconfigure(0, weight=1)
        self.info = ttk.Label(status, text="")
        self.info.grid(row=0, column=0, sticky="w")

        # Load PDF
        try:
            self.pdf = pdfplumber.open(self.pdf_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PDF:\n{e}")
            self.destroy(); return

        self.num_pages = len(self.pdf.pages)
        self.bind("<Left>", lambda e: self.prev_page())
        self.bind("<Right>", lambda e: self.next_page())

        self.render_page()

    # ---- GUI actions ----
    def page_bboxes_and_margins(self):
        page = self.pdf.pages[self.current_idx]
        Lbbox, Rbbox = two_col_bboxes(page, self.top_frac.get(), self.bottom_frac.get(), self.gutter_frac.get())
        Lx = detect_col_leftmost(page, Lbbox)
        Rx = detect_col_leftmost(page, Rbbox)
        return page, Lbbox, Rbbox, Lx, Rx

    def render_page(self):
        # render base page
        page, Lbbox, Rbbox, Lx, Rx = self.page_bboxes_and_margins()

        im = page.to_image(resolution=int(self.dpi))
        # draw column boxes (light blue)
        for (x0, top, x1, bottom) in (Lbbox, Rbbox):
            im.draw_rect((x0, top, x1, bottom), stroke="#66B2FF", stroke_width=3)
        # draw red margin lines across each column (short marker at 6% depth)
        def draw_margin_line(x, bbox):
            if x is None: return
            x0, top, x1, bottom = bbox
            y = top + (bottom - top) * 0.06
            im.draw_line([(x0, y), (x, y)], stroke="red", stroke_width=4)
        draw_margin_line(Lx, Lbbox)
        draw_margin_line(Rx, Rbbox)

        # replace temp file (ensure old is removed after the new image is not locked)
        new_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        new_path = new_tmp.name
        new_tmp.close()
        im.save(new_path)

        # --- Fit into canvas (safe, no file lock on disk file) ---
        with Image.open(new_path) as _pil:
            pil = _pil.copy()  # detach from file; allow deletion afterwards

        # delete previous tmp *now* (old PhotoImage already owns its pixels)
        if self._tmp_png and os.path.exists(self._tmp_png):
            try:
                os.remove(self._tmp_png)
            except PermissionError:
                pass
        self._tmp_png = new_path

        # scale image to canvas
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        scale = min(cw / pil.width, ch / pil.height, 1.0)
        tw = max(1, int(round(pil.width * scale)))
        th = max(1, int(round(pil.height * scale)))
        if (tw, th) != (pil.width, pil.height):
            pil = pil.resize((tw, th), Image.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img)

        # update info + detected margins readout
        self.info.config(text=f"Page {self.current_idx+1}/{self.num_pages}  |  tol = {self.tol.get():.2f} pt")
        ml = "—" if Lx is None else f"{Lx:.2f}"
        mr = "—" if Rx is None else f"{Rx:.2f}"
        self.marg_label.config(text=f"L: {ml}   R: {mr}")

    def refresh(self):
        self.render_page()

    def prev_page(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.render_page()

    def next_page(self):
        if self.current_idx < self.num_pages - 1:
            self.current_idx += 1
            self.render_page()

    def assign_margins(self):
        # use current page’s detected margins as defaults
        _, _, _, Lx, Rx = self.page_bboxes_and_margins()
        if Lx is None and Rx is None:
            messagebox.showwarning("No margins", "Could not detect margins on this page.")
            return
        self.L_margin = None if Lx is None else float(Lx)
        self.R_margin = None if Rx is None else float(Rx)
        messagebox.showinfo("Assigned", f"Assigned margins from page {self.current_idx+1}:\n"
                                        f"L = {self.L_margin if self.L_margin is not None else '—'}\n"
                                        f"R = {self.R_margin if self.R_margin is not None else '—'}")

    def export_margins(self):
        if self.L_margin is None and self.R_margin is None:
            if not messagebox.askyesno("No assigned margins",
                                       "You haven't assigned margins yet.\nExport detected values from this page instead?"):
                return
            _, _, _, Lx, Rx = self.page_bboxes_and_margins()
            L_export = Lx
            R_export = Rx
        else:
            L_export = self.L_margin
            R_export = self.R_margin

        if L_export is None and R_export is None:
            messagebox.showwarning("No margins", "No margins to export.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON","*.json")],
                                            initialfile="margins.json",
                                            title="Export margins.json")
        if not path:
            return
        data = {"global": {}, "pages": {}}
        if L_export is not None: data["global"]["L"] = float(L_export)
        if R_export is not None: data["global"]["R"] = float(R_export)
        data["global"]["tol"] = float(self.tol.get())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("Exported", f"Saved: {os.path.abspath(path)}")

    def parse_all(self):
        # ensure margins are set (fallback to current page detection)
        if self.L_margin is None and self.R_margin is None:
            _, _, _, Lx, Rx = self.page_bboxes_and_margins()
            self.L_margin = None if Lx is None else float(Lx)
            self.R_margin = None if Rx is None else float(Rx)
        if self.L_margin is None and self.R_margin is None:
            messagebox.showwarning("No margins", "No margins assigned/detected; cannot parse.")
            return

        # pick output path if the provided --out points to a directory
        out_path = self.out_path
        if os.path.isdir(out_path):
            out_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                    filetypes=[("JSON","*.json")],
                                                    initialfile="qa.json",
                                                    title="Save QA JSON")
            if not out_path:
                return
            self.out_path = out_path
            self.out_path_label.config(text=os.path.abspath(out_path))

        try:
            qa = pdf_to_qa_margin_chunked(
                pdf_path=self.pdf_path,
                subject=(None if self.auto_subject else self.subject), year=self.year, target=self.target,
                start_num=1,
                L_margin=self.L_margin, R_margin=self.R_margin,
                tol=float(self.tol.get()),
                top_frac=float(self.top_frac.get()),
                bottom_frac=float(self.bottom_frac.get()),
                gutter_frac=float(self.gutter_frac.get()),
                y_tol=3.0,
                auto_subject=self.auto_subject,
                subject_keywords=self.subject_keywords,
                chunk_preview_dir=self.chunk_preview_dir,
                chunk_preview_dpi=self.chunk_preview_dpi,
                chunk_preview_pad=self.chunk_preview_pad,
            )
        except Exception as e:
            messagebox.showerror("Parse failed", str(e))
            return

        try:
            with open(self.out_path, "w", encoding="utf-8") as f:
                json.dump(qa, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return

        preview_count = sum(
            1
            for item in qa
            if isinstance(item, dict)
            and isinstance(item.get("content"), dict)
            and item["content"].get("preview_image")
        )

        msg_lines = [
            f"Wrote {len(qa)} QA items →",
            os.path.abspath(self.out_path),
        ]
        if preview_count and self.chunk_preview_dir:
            msg_lines.extend([
                "",
                f"Saved {preview_count} chunk previews →",
                self.chunk_preview_dir,
            ])

        messagebox.showinfo("Done", "\n".join(msg_lines))

    def destroy(self):
        # clean temp image on exit (safe even if Windows still holds a handle; ignore errors)
        try:
            if self._tmp_png and os.path.exists(self._tmp_png):
                os.remove(self._tmp_png)
        except PermissionError:
            pass
        try: self.pdf.close()
        except: pass
        super().destroy()


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Preview margins, assign defaults, and parse QA from PDF.")
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--out", required=True, help="Path to write QA JSON (file path; if folder, dialog will prompt)")
    ap.add_argument("--subject", help="Subject (e.g., 형법). Use 'auto' to detect from page headers.")
    ap.add_argument("--year", type=int, required=True, help="Year (e.g., 2025)")
    ap.add_argument("--target", default="default", help="Target label")

    ap.add_argument("--dpi", type=int, default=180, help="Preview DPI (default 180)")
    ap.add_argument("--tol", type=float, default=1.5, help="Margin hit tolerance (points)")
    ap.add_argument("--top-frac", type=float, default=0.04, help="Top crop fraction")
    ap.add_argument("--bottom-frac", type=float, default=0.96, help="Bottom crop fraction")
    ap.add_argument("--gutter-frac", type=float, default=0.005, help="Half-gap around center")
    ap.add_argument("--subject-map", help="Optional JSON mapping of header substrings to canonical subjects.")
    ap.add_argument("--chunk-preview-dir", help="Directory to store JPG previews for each extracted chunk.")
    ap.add_argument("--chunk-preview-dpi", type=int, default=180,
                    help="DPI for chunk preview JPG crops (default 180).")
    ap.add_argument("--chunk-preview-pad", type=float, default=2.0,
                    help="Padding (points) added around each chunk preview crop (default 2).")

    args = ap.parse_args()

    subject_keywords = None
    if args.subject_map:
        try:
            with open(args.subject_map, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                subject_keywords = {str(k): str(v) for k, v in data.items()}
            else:
                raise ValueError("subject map JSON must be an object of substring -> subject")
        except Exception as exc:
            raise SystemExit(f"Failed to load subject map: {exc}")

    subject_arg = args.subject
    auto_subject = False
    if subject_arg is None or str(subject_arg).strip().lower() == "auto":
        auto_subject = True
        subject_arg = None

    chunk_preview_dir_arg = args.chunk_preview_dir.strip() if args.chunk_preview_dir else None

    app = MarginPreviewApp(
        pdf_path=args.pdf,
        out_path=args.out,
        subject=subject_arg,
        year=args.year,
        target=args.target,
        dpi=args.dpi,
        tol=args.tol,
        top_frac=args.top_frac,
        bottom_frac=args.bottom_frac,
        gutter_frac=args.gutter_frac,
        auto_subject=auto_subject,
        subject_keywords=subject_keywords,
        chunk_preview_dir=chunk_preview_dir_arg,
        chunk_preview_dpi=args.chunk_preview_dpi,
        chunk_preview_pad=args.chunk_preview_pad
    )
    app.mainloop()

if __name__ == "__main__":
    main()
