# pip install pdfplumber pillow
from pathlib import Path
import pdfplumber

# ========== CONFIG ==========
PDF_PATH = Path(r"C:\Users\Jae\Desktop\HM\script\test_script\20년 1차_pdfplumber_test.pdf")
OUT_DIR  = PDF_PATH.parent / f"{PDF_PATH.stem}_above_top_rule_words"
DPI = 150

# line detection
TOL = 1.0          # tolerance to consider a line horizontal
MIN_FRAC = 0.5     # keep horizontals >= 50% of page width
PAD_DOWN = 3.0     # include a tiny band *below* the rule so boundary glyphs aren't dropped

# word extraction / layout
X_TOL, Y_TOL = 2, 4
LINE_MERGE_TOL = 2.0  # how similar "top" must be to treat words as same line (in px of top-origin coords)
# ===========================

def is_horizontal(L): 
    return abs(L["y0"] - L["y1"]) <= TOL

def words_in_bbox_toporigin(words, crop_rect_toporigin):
    """Filter words (which are already top-origin) by intersection with a top-origin rect."""
    cx0, ctop, cx1, cbottom = crop_rect_toporigin
    keep = []
    for w in words:
        # word rect: x0, top, x1, bottom in top-origin already
        if not (w["bottom"] < ctop or
                w["top"]    > cbottom or
                w["x1"]     < cx0 or
                w["x0"]     > cx1):
            keep.append(w)
    return keep

def words_to_text_in_reading_order(words, line_tol=LINE_MERGE_TOL):
    """Sort words by line (top) then x0, and join them into lines."""
    if not words:
        return ""
    # sort by (top, x0)
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines = []
    current = []
    prev_top = None
    for w in words:
        if prev_top is None or abs(w["top"] - prev_top) <= line_tol:
            current.append(w["text"])
        else:
            lines.append(" ".join(current))
            current = [w["text"]]
        prev_top = w["top"]
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)

def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    with pdfplumber.open(PDF_PATH) as pdf:
        for i, rp in enumerate(pdf.pages, start=1):
            # normalize rotation so coordinates are consistent
            page = rp.rotate(-rp.rotation) if rp.rotation else rp
            pw, ph = float(page.width), float(page.height)

            # ---- 1) detect the TOPMOST long horizontal line (>= MIN_FRAC * page width)
            candidates = [
                (max(L["y0"], L["y1"]), L)
                for L in (page.lines or [])
                if is_horizontal(L) and abs(L["x1"] - L["x0"]) >= MIN_FRAC * pw
            ]
            if not candidates:
                print(f"Page {i}: no long horizontal line found — skipping")
                continue

            y_div, _ = max(candidates, key=lambda t: t[0])  # highest y (closest to top in PDF coords)

            # ---- 2) define the PDF-space crop ABOVE the line
            # PDF coords use bottom-left origin: (x0, y0, x1, y1)
            y0_pdf = max(0.0, y_div - PAD_DOWN)  # extend a little *below* the rule
            bbox_pdf = (0, y0_pdf, pw, ph)

            # ---- 3) extract words and keep only those inside the crop (use top-origin for intersection)
            words = page.extract_words(x_tolerance=X_TOL, y_tolerance=Y_TOL, keep_blank_chars=False) or []

            # Convert our PDF-space bbox to top-origin for intersection with words
            crop_toporigin = {
                "x0": bbox_pdf[0],
                "top": ph - bbox_pdf[3],
                "x1": bbox_pdf[2],
                "bottom": ph - bbox_pdf[1],
            }
            words_crop = words_in_bbox_toporigin(
                words,
                (crop_toporigin["x0"], crop_toporigin["top"], crop_toporigin["x1"], crop_toporigin["bottom"])
            )

            text = words_to_text_in_reading_order(words_crop, line_tol=LINE_MERGE_TOL)

            # ---- 4) save TXT
            out_txt = OUT_DIR / f"{PDF_PATH.stem}_p{i:03d}_above.txt"
            with open(out_txt, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            print(f"Page {i}: saved text -> {out_txt}")

            # ---- 5) preview: red crop region + blue word boxes
            im = page.to_image(resolution=DPI)
            # red crop overlay
            im.draw_rects([crop_toporigin], fill=(255, 0, 0, 40), stroke="red", stroke_width=2)
            # blue word boxes
            blue_rects = [{"x0": w["x0"], "top": w["top"], "x1": w["x1"], "bottom": w["bottom"]} for w in words_crop]
            if blue_rects:
                im.draw_rects(blue_rects, stroke="blue", fill=None, stroke_width=2)

            out_png = OUT_DIR / f"{PDF_PATH.stem}_p{i:03d}_preview.png"
            im.save(out_png)
            print(f"Page {i}: saved preview -> {out_png}")

if __name__ == "__main__":
    main()
