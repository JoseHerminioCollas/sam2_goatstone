#!/usr/bin/env python3
"""Cut objects out of a photo with SAM 2 and save each one as a transparent PNG.

Interactive (opens a window):
    python cut.py photo.jpg -o out/

    left click   add a point on the object
    right click  add a point on something that is NOT the object
    u            undo last point
    r            reset points for the current object
    s / Enter    save the current object and start the next one
    q / Esc      quit

Headless (re-use clicks saved by an earlier run):
    python cut.py photo.jpg -o out/ --points out/points.json
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def load_predictor(model_path, config):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.set_grad_enabled(False)
    model = build_sam2(config, str(model_path), device="cpu")
    return SAM2ImagePredictor(model)


def predict(predictor, points, smooth=2):
    """Return alpha (0..1, float32) for the given [[x, y, label], ...] points."""
    pts = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
    labels = np.array([p[2] for p in points], dtype=np.int32)
    single = len(points) == 1
    logits, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=labels,
        multimask_output=single,  # one click is ambiguous: take the best of 3
        return_logits=True,
    )
    best = int(np.argmax(scores)) if single else 0
    return clean_mask(logits[best], smooth)


def clean_mask(logits, smooth=2):
    """Keep the largest piece, fill holes (e.g. glare), then smooth the outline."""
    hard = (logits > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        hard = (lab == biggest).astype(np.uint8)
    # fill holes: flood the background from a 1px border, everything not reached is inside
    flood = cv2.copyMakeBorder(hard, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood, None, (0, 0), 1)
    filled = hard | (1 - flood[1:-1, 1:-1])
    if smooth <= 0:
        return filled.astype(np.float32)
    return smooth_edge(filled, smooth)


def smooth_edge(mask, sigma, shrink=1, ss=4):
    """Round off bumps in the outline, trim `shrink` px of fringe, and anti-alias the edge.

    Works at `ss`x resolution on the object's bounding box, then scales back down,
    which gives a crisp edge with one pixel of anti-aliasing (like a cut path in GIMP).
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask.astype(np.float32)
    m = int(3 * sigma + shrink + 2)
    h, w = mask.shape
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, h)
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, w)
    crop = mask[y0:y1, x0:x1].astype(np.float32)
    big = cv2.resize(crop, None, fx=ss, fy=ss, interpolation=cv2.INTER_LINEAR)
    big = (cv2.GaussianBlur(big, (0, 0), sigma * ss) > 0.5).astype(np.uint8)
    if shrink:
        k = 2 * shrink * ss + 1
        big = cv2.erode(big, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    small = cv2.resize(big.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w), np.float32)
    out[y0:y1, x0:x1] = small
    return out


def save_cutout(rgb, alpha, path, pad):
    ys, xs = np.nonzero(alpha > 0.5)
    if len(ys) == 0:
        print(f"  empty mask, skipped {path.name}")
        return False
    h, w = alpha.shape
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    a = (alpha * 255).round().astype(np.uint8)
    rgba = np.dstack([rgb, a])[y0:y1, x0:x1]
    Image.fromarray(rgba, "RGBA").save(path)
    print(f"  saved {path}")
    return True


def save_preview(rgb, alphas, path):
    """Photo with every cut outline drawn and numbered, for checking the result."""
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for i, a in enumerate(alphas, 1):
        m = (a > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (40, 40, 255), 2)
        M = cv2.moments(m)
        if M["m00"]:
            c = (int(M["m10"] / M["m00"]) - 8, int(M["m01"] / M["m00"]) + 8)
            cv2.putText(out, str(i), c, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            cv2.putText(out, str(i), c, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
    cv2.imwrite(str(path), out)
    print(f"  preview {path}")


def run_headless(predictor, rgb, objects, outdir, prefix, pad, smooth):
    alphas = []
    for i, obj in enumerate(objects, 1):
        alpha = predict(predictor, obj["points"], smooth)
        name = obj.get("name") or f"{prefix}_{i:02d}"
        if save_cutout(rgb, alpha, outdir / f"{name}.png", pad):
            alphas.append(alpha)
    return alphas


def run_interactive(predictor, rgb, outdir, prefix, pad, max_side, smooth):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    objects, alphas, points = [], [], []
    state = {"alpha": None, "dirty": True}

    def on_mouse(event, x, y, flags, _):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
            points.append([int(x / scale), int(y / scale), label])
            state["dirty"] = True

    win = "cut  |  L-click: object  R-click: not object  u: undo  r: reset  s: save  q: quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        if state["dirty"]:
            state["alpha"] = predict(predictor, points, smooth) if points else None
            state["dirty"] = False
            view = bgr.copy()
            for a in alphas:  # already saved objects: dim them
                view[a > 0.5] = (view[a > 0.5] * 0.35).astype(np.uint8)
            if state["alpha"] is not None:
                m = state["alpha"] > 0.5
                view[m] = (0.55 * view[m] + 0.45 * np.array([255, 160, 0])).astype(np.uint8)
            for x, y, lab in points:
                cv2.circle(view, (x, y), 7, (0, 220, 0) if lab else (0, 0, 255), -1)
                cv2.circle(view, (x, y), 7, (255, 255, 255), 1)
            if scale < 1:
                view = cv2.resize(view, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("q"), 27):
            break
        if key == ord("u") and points:
            points.pop(); state["dirty"] = True
        elif key == ord("r"):
            points.clear(); state["dirty"] = True
        elif key in (ord("s"), 13, 10) and state["alpha"] is not None:
            name = f"{prefix}_{len(objects) + 1:02d}"
            if save_cutout(rgb, state["alpha"], outdir / f"{name}.png", pad):
                objects.append({"name": name, "points": [list(p) for p in points]})
                alphas.append(state["alpha"])
                (outdir / "points.json").write_text(json.dumps({"objects": objects}, indent=1))
            points.clear(); state["dirty"] = True
    cv2.destroyAllWindows()
    return alphas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--outdir", default="out")
    ap.add_argument("--points", help="JSON from an earlier run; cuts without opening a window")
    ap.add_argument("--prefix", default="object", help="output file name prefix (default: object)")
    ap.add_argument("--pad", type=int, default=6, help="transparent margin around each cut, px")
    ap.add_argument("--smooth", type=float, default=2, help="outline smoothing, 0 = off (default: 2)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--max-side", type=int, default=1000, help="max window size, px")
    args = ap.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"model not found: {args.model}\nrun ./download_model.sh first")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rgb = np.array(Image.open(args.image).convert("RGB"))  # writable copy for torch

    print("loading model ...")
    predictor = load_predictor(args.model, args.config)
    print("reading image (the slow step) ...")
    predictor.set_image(rgb)

    if args.points:
        objects = json.loads(Path(args.points).read_text())["objects"]
        alphas = run_headless(predictor, rgb, objects, outdir, args.prefix, args.pad, args.smooth)
    else:
        alphas = run_interactive(predictor, rgb, outdir, args.prefix, args.pad, args.max_side, args.smooth)
    if alphas:
        save_preview(rgb, alphas, outdir / "_preview.jpg")


if __name__ == "__main__":
    main()
