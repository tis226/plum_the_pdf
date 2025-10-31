import os, re, sys, json, argparse, tempfile
from decimal import Decimal
import pdfplumber

# GUI deps (only with --preview)
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from PIL import Image, ImageTk
except Exception:
    tk = ttk = messagebox = Image = ImageTk = None

# ============ Config ============
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"  # extend as needed
QUESTION_START_RE = re.compile(
    rf"^\s*(?:[{CIRCLED}]|[0-9]{{1,3}}[.)]|제\s*[0-9]{{1,3}}\s*문)",
    re.UNICODE
)

# ============ Utils ============
def clean_arg(p: str) -> str:
    import re, os
    m = re.fullmatch(r"[rR]([\"'])(.*)\1", p)
    if m: p = m.group(2)
    return os.path.normpath(p)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def to_float(x):
    return float(x) if isinstance(x, (int, float, Decimal)) else x

# ============ Layout ============
def two_col_bboxes(page, top_frac=0.04, bottom_frac=0.96, gutter_frac=0.005):
    w, h = float(page.width), float(page.height)
    top = h * top_frac
    bottom = h * bottom_frac
    gutter = w * gutter_frac
    mid = w * 0.5
    left_bbox  = (0.0, top,  mid - gutter, bottom)
    right_bbox = (mid + gutter, top,  w, bottom)
    return left_bbox, right_bbox

# ============ Text helpers ============
def group_words_into_lines(words, y_tol=3.0):
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines, cur = [], [ws[0]]
    cur_top = ws[0]["top"]
    for w in ws[1:]:
        if abs(w["top"] - cur_top) <= y_tol:
            cur.append(w)
        else:
            cur.sort(key=lambda x: x["x0"]); lines.append(cur)
            cur = [w]; cur_top = w["top"]
    cur.sort(key=lambda x: x["x0"]); lines.append(cur)
    return lines

def extract_lines_in_bbox(page, bbox, y_tol=3.0):
    x0b, y0b, x1b, y1b = bbox
    sub = page.within_bbox(bbox)
    words = sub.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False, use_text_flow=True) or []
    words_slim = []
    for w in words:
        wx0, wx1, wt, wb, tx = w.get("x0"), w.get("x1"), w.get("top"), w.get("bottom"), w.get("text", "")
        if None in (wx0, wx1, wt, wb): continue
        if not (y0b <= wt <= y1b or y0b <= wb <= y1b): continue
        words_slim.append({"x0": to_float(wx0), "x1": to_float(wx1),
                           "top": to_float(wt), "bottom": to_float(wb), "text": tx})
    lines = group_words_into_lines(words_slim, y_tol=y_tol)
    out = []
    for ln in lines:
        lx0 = min(w["x0"] for w in ln)
        lx1 = max(w["x1"] for w in ln)
        top = min(w["top"] for w in ln)
        bot = max(w["bottom"] for w in ln)
        txt = " ".join(w["text"] for w in ln if w.get("text"))
        out.append({"x0": lx0, "x1": lx1, "top": top, "bottom": bot, "y": 0.5*(top+bot), "text": txt})
    return out

def autodetect_margin_x(page, bbox):
    sub = page.within_bbox(bbox)
    chars = sub.chars or []
    if chars:
        return float(min(c["x0"] for c in chars if c.get("x0") is not None))
    words = sub.extract_words(x_tolerance=3, y_tolerance=3) or []
    if words:
        return float(min(w["x0"] for w in words if w.get("x0") is not None))
    return None

# ============ Starts / chunks ============
def detect_question_starts(lines, margin_x, col_left, tol=1.0):
    starts = []
    target_rel = None if margin_x is None else (margin_x - col_left)
    for i, ln in enumerate(lines):
        rel = ln["x0"] - col_left
        left_ok = True if target_rel is None else abs(rel - target_rel) <= tol
        text_ok = bool(QUESTION_START_RE.search(ln["text"]))
        if left_ok and text_ok:
            starts.append(i)
    return starts

def make_chunks_from_starts(lines, start_idxs, pad=2.0):
    chunks = []
    if not start_idxs: return chunks
    idxs = list(start_idxs) + [len(lines)]
    for a, b in zip(idxs[:-1], idxs[1:]):
        block = lines[a:b]
        x0 = min(l["x0"] for l in block)
        x1 = max(l["x1"] for l in block)
        top = min(l["top"] for l in block) - pad
        bot = max(l["bottom"] for l in block) + pad
        text = "\n".join(l["text"] for l in block)
        chunks.append({
            "x0": x0, "x1": x1, "top": top, "bottom": bot,
            "start_line": a, "end_line": b-1, "text": text
        })
    return chunks

# ============ FLOW across pages ============
def build_flow_segments(pdf, top_frac, bottom_frac, gutter_frac, y_tol):
    segs = []
    for i, page in enumerate(pdf.pages):
        L, R = two_col_bboxes(page, top_frac, bottom_frac, gutter_frac)
        L_lines = extract_lines_in_bbox(page, L, y_tol=y_tol)
        R_lines = extract_lines_in_bbox(page, R, y_tol=y_tol)
        segs.append((i, 'L', L, L_lines))
        segs.append((i, 'R', R, R_lines))
    return segs

def flow_chunk_all_pages(pdf, L_rel_offset, R_rel_offset, y_tol, tol, top_frac, bottom_frac, gutter_frac):
    segs = build_flow_segments(pdf, top_frac, bottom_frac, gutter_frac, y_tol)

    # margins per segment from global rel offsets
    seg_meta = []
    for (pi, col, bbox, lines) in segs:
        L, R = two_col_bboxes(pdf.pages[pi], top_frac, bottom_frac, gutter_frac)
        if col == 'L':
            margin_abs = None if L_rel_offset is None else (L[0] + L_rel_offset)
            col_left = L[0]
        else:
            margin_abs = None if R_rel_offset is None else (R[0] + R_rel_offset)
            col_left = R[0]
        seg_meta.append((pi, col, bbox, lines, margin_abs, col_left))

    seg_starts = []
    for (pi, col, bbox, lines, m_abs, col_left) in seg_meta:
        seg_starts.append(detect_question_starts(lines, m_abs, col_left, tol=tol))

    chunks = []
    current = None
    for seg_idx, (pi, col, bbox, lines, m_abs, col_left) in enumerate(seg_meta):
        starts = set(seg_starts[seg_idx])
        i = 0
        while i < len(lines):
            if i in starts:
                if current is not None:
                    chunks.append(current)
                current = {"pieces": [], "start": {"page": pi, "col": col, "line_idx": i}}
            if current is not None:
                next_mark = min((j for j in starts if j > i), default=None)
                end_idx = (next_mark-1) if next_mark is not None else (len(lines)-1)
                block = lines[i:end_idx+1]
                if block:
                    x0 = min(l["x0"] for l in block)
                    x1 = max(l["x1"] for l in block)
                    top = min(l["top"] for l in block) - 2.0
                    bot = max(l["bottom"] for l in block) + 2.0
                    text = "\n".join(l["text"] for l in block)
                    current["pieces"].append({
                        "page": pi, "col": col,
                        "box": {"x0": x0, "x1": x1, "top": top, "bottom": bot},
                        "start_line": i, "end_line": end_idx,
                        "text": text
                    })
                i = end_idx + 1
            else:
                i += 1
    if current is not None:
        chunks.append(current)

    per_page_boxes = {i: [] for i in range(len(pdf.pages))}
    for ch_id, ch in enumerate(chunks, start=1):
        for p in ch["pieces"]:
            b = p["box"].copy()
            b["chunk_id"] = ch_id
            b["col"] = p["col"]
            per_page_boxes[p["page"]].append(b)

    return chunks, per_page_boxes

# ============ Rendering ============
def draw_page_overlay(page, left_bbox, right_bbox, L_abs, R_abs, boxes, out_path, dpi=150, debug_bboxes=False):
    im = page.to_image(resolution=dpi)
    # show column outlines
    if debug_bboxes:
        for (x0,y0,x1,y1) in (left_bbox, right_bbox):
            im.draw_rect((x0,y0,x1,y1), stroke="#00FFFF", fill=None, stroke_width=1)
    # show global defaults (red) at top band
    if L_abs is not None:
        im.draw_line([(left_bbox[0], left_bbox[1]+20), (L_abs, left_bbox[1]+20)], stroke="red", stroke_width=5)
    if R_abs is not None:
        im.draw_line([(right_bbox[0], right_bbox[1]+20), (R_abs, right_bbox[1]+20)], stroke="red", stroke_width=5)
    # draw chunk boxes
    for b in boxes:
        im.draw_rect((b["x0"], b["top"], b["x1"], b["bottom"]), stroke="red", fill=None, stroke_width=3)
    im.save(out_path)

# ============ GUI ============
class App(tk.Tk):
    def __init__(self, pdf_path, out_dir=None, dpi=150, y_tol=3.0, tol=1.0,
                 top_frac=0.04, bottom_frac=0.96, gutter_frac=0.005, debug_bboxes=False):
        super().__init__()
        self.title("Flow-aware 2-column chunker (pick default margins from a page)")
        self.geometry("1200x880"); self.minsize(1000,700)

        self.pdf_path = clean_arg(pdf_path)
        self.dpi=dpi; self.y_tol=y_tol; self.tol=tol
        self.top_frac=top_frac; self.bottom_frac=bottom_frac; self.gutter_frac=gutter_frac
        self.debug_bboxes=debug_bboxes
        self.out_dir = out_dir or os.path.join(os.path.dirname(self.pdf_path), "chunks_out")
        ensure_dir(self.out_dir)

        # global margins (relative offsets from col-left); set via "Use this page’s AUTO margins"
        self.L_rel = None
        self.R_rel = None

        # UI
        top = ttk.Frame(self); top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Button(top, text="◀ Prev", command=self.prev).pack(side=tk.LEFT)
        ttk.Button(top, text="Next ▶", command=self.next).pack(side=tk.LEFT, padx=(6,0))
        ttk.Button(top, text="Use this page’s AUTO margins for ALL pages", command=self.use_this_page_auto).pack(side=tk.LEFT, padx=(12,0))
        ttk.Button(top, text="Export ALL (flow-aware)", command=self.export_all).pack(side=tk.RIGHT)
        self.info = ttk.Label(top, text="We auto-detect per-page margins (orange). Click the button to make this page’s autos the global defaults (red), then Export.")
        self.info.pack(side=tk.LEFT, padx=(12,0))

        self.canvas = tk.Canvas(self, bg="#222"); self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas.bind("<Configure>", lambda e: self.render())

        self.status = ttk.Label(self, text=""); self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)

        # data
        try:
            self.pdf = pdfplumber.open(self.pdf_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PDF:\n{e}"); self.destroy(); return

        self.page_idx = 0
        self.tk_img = None

        # Keys
        self.bind("<Left>", lambda e: self.prev())
        self.bind("<Right>", lambda e: self.next())

        self.render()

    def current_page(self):
        return self.pdf.pages[self.page_idx]

    def detect_auto_for_page(self, page):
        L, R = two_col_bboxes(page, self.top_frac, self.bottom_frac, self.gutter_frac)
        L_auto = autodetect_margin_x(page, L)
        R_auto = autodetect_margin_x(page, R)
        return L, R, L_auto, R_auto

    def use_this_page_auto(self):
        page = self.current_page()
        L, R, L_auto, R_auto = self.detect_auto_for_page(page)
        if L_auto is None or R_auto is None:
            messagebox.showwarning("No text", "Couldn’t auto-detect both margins on this page.")
            return
        self.L_rel = L_auto - L[0]
        self.R_rel = R_auto - R[0]
        messagebox.showinfo("Defaults set",
            f"This page’s auto margins applied as defaults for ALL pages:\n"
            f"Left rel = {self.L_rel:.2f} pt   Right rel = {self.R_rel:.2f} pt")
        self.render()

    def render(self):
        page = self.current_page()
        L, R, L_auto, R_auto = self.detect_auto_for_page(page)

        im = page.to_image(resolution=self.dpi)

        # Column outlines (cyan)
        for (x0,y0,x1,y1) in (L, R):
            im.draw_rect((x0,y0,x1,y1), stroke="#00FFFF", fill=None, stroke_width=1)

        # Auto margins (orange)
        if L_auto is not None:
            im.draw_line([(L[0], L[1]+8), (L_auto, L[1]+8)], stroke="orange", stroke_width=6)
        if R_auto is not None:
            im.draw_line([(R[0], R[1]+8), (R_auto, R[1]+8)], stroke="orange", stroke_width=6)

        # Global defaults (red) displayed on THIS page
        if self.L_rel is not None:
            im.draw_line([(L[0], L[1]+20), (L[0]+self.L_rel, L[1]+20)], stroke="red", stroke_width=5)
        if self.R_rel is not None:
            im.draw_line([(R[0], R[1]+20), (R[0]+self.R_rel, R[1]+20)], stroke="red", stroke_width=5)

        # push to canvas
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False); path = tmp.name; tmp.close()
        im.save(path)
        pil = Image.open(path)
        self.update_idletasks()
        cw = self.canvas.winfo_width() or 1000
        ch = self.canvas.winfo_height() or 700
        scale = min(cw / pil.width, ch / pil.height, 1.0)
        if scale < 1.0:
            pil = pil.resize((max(1,int(pil.width*scale)), max(1,int(pil.height*scale))), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self.tk_img)
        try: os.remove(path)
        except: pass

        self.status.config(text=f"Page {self.page_idx+1}/{len(self.pdf.pages)} | "
                                f"AUTO L={L_auto}  R={R_auto}  |  Defaults (rel pt): L={self.L_rel}  R={self.R_rel}")

    def prev(self):
        self.page_idx = max(0, self.page_idx-1); self.render()

    def next(self):
        self.page_idx = min(len(self.pdf.pages)-1, self.page_idx+1); self.render()

    def export_all(self):
        if self.L_rel is None or self.R_rel is None:
            messagebox.showwarning("Pick defaults", "Choose a page and click 'Use this page’s AUTO margins for ALL pages' first.")
            return
        ensure_dir(self.out_dir)

        # Build flow chunks across all pages
        chunks, per_page_boxes = flow_chunk_all_pages(
            self.pdf, self.L_rel, self.R_rel, self.y_tol, self.tol,
            self.top_frac, self.bottom_frac, self.gutter_frac
        )

        # Write JSON
        all_json = {"chunks": chunks, "note": "Chunks follow reading flow across columns/pages."}
        with open(os.path.join(self.out_dir, "chunks_flow.json"), "w", encoding="utf-8") as f:
            json.dump(all_json, f, ensure_ascii=False, indent=2)

        # Render per-page overlays
        for pi, page in enumerate(self.pdf.pages):
            L, R = two_col_bboxes(page, self.top_frac, self.bottom_frac, self.gutter_frac)
            L_abs = L[0] + self.L_rel
            R_abs = R[0] + self.R_rel
            boxes = per_page_boxes.get(pi, [])
            png_path = os.path.join(self.out_dir, f"page_{pi+1:03d}_chunks.png")
            draw_page_overlay(page, L, R, L_abs, R_abs, boxes, png_path, dpi=self.dpi, debug_bboxes=self.debug_bboxes)

        messagebox.showinfo("Done", f"Exported ALL pages to:\n{self.out_dir}")

# ============ CLI ============
def main():
    ap = argparse.ArgumentParser(description="Flow-aware chunking with 'use this page’s auto margins as defaults' button")
    ap.add_argument("pdf_path")
    ap.add_argument("--out", help="Output folder (default: <pdf_dir>/chunks_out)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--y-tol", type=float, default=3.0, help="Line grouping vertical tolerance (pt)")
    ap.add_argument("--tol", type=float, default=1.0, help="Margin match tolerance (pt)")
    ap.add_argument("--top-frac", type=float, default=0.04)
    ap.add_argument("--bottom-frac", type=float, default=0.96)
    ap.add_argument("--gutter-frac", type=float, default=0.005)
    ap.add_argument("--debug-bboxes", action="store_true")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    if not args.preview:
        print("Run with --preview to open the GUI.", file=sys.stderr); sys.exit(1)

    global tk, ttk, messagebox, Image, ImageTk
    if any(x is None for x in (tk, ttk, messagebox, Image, ImageTk)):
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox
            from PIL import Image, ImageTk
        except Exception as e:
            print("GUI preview requires tkinter + Pillow:", e, file=sys.stderr); sys.exit(1)

    app = App(
        pdf_path=clean_arg(args.pdf_path),
        out_dir=args.out, dpi=args.dpi, y_tol=args.y_tol, tol=args.tol,
        top_frac=args.top_frac, bottom_frac=args.bottom_frac, gutter_frac=args.gutter_frac,
        debug_bboxes=args.debug_bboxes
    )
    app.mainloop()

if __name__ == "__main__":
    main()
