# sam2_goatstone

Cut objects out of a photo and save each one as its own transparent PNG.
You click on an object and [SAM 2](https://github.com/facebookresearch/sam2)
(Meta's Segment Anything 2, Apache 2.0) finds its outline. Runs locally on a CPU;
no GPU needed, and nothing is uploaded.

Uses the **SAM 2.1 tiny** model (156 MB, about 1 GB of RAM while running).

## Setup

Needs Python 3.10+ and `python3-venv` (`sudo apt install python3.12-venv` on Ubuntu 24.04).

```bash
python3 -m venv .venv
# PyTorch first (CPU build), so SAM 2 can build against it
.venv/bin/pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-build-isolation -r requirements.txt
./download_model.sh
```

## Use

```bash
.venv/bin/python cut.py photo.jpg -o out/
```

A window opens. Reading the image takes a few seconds; after that each click updates quickly.

| Input        | Action                                              |
|--------------|-----------------------------------------------------|
| left click   | point on the object (add more to grow the outline)  |
| right click  | point on something that is *not* the object         |
| `u`          | undo last point                                     |
| `r`          | clear points for the current object                 |
| `s` / Enter  | save the current object, start the next one         |
| `q` / Esc    | quit                                                |

Saved objects are dimmed so you can see what is left.

Output in `out/`:

- `object_01.png`, `object_02.png`, … cropped cut-outs with a transparent background
- `points.json` the clicks, so the same cuts can be made again
- `_preview.jpg` the photo with every outline drawn and numbered

### Re-run without clicking

```bash
.venv/bin/python cut.py photo.jpg -o out/ --points out/points.json
```

`points.json` can also be written by hand: each object is a list of
`[x, y, label]` points in image pixels, label `1` = on the object, `0` = not on it.

```json
{"objects": [
  {"name": "stone_01", "points": [[220, 320, 1]]},
  {"name": "stone_02", "points": [[465, 260, 1], [430, 330, 0]]}
]}
```

### Options

- `--prefix stone` names files `stone_01.png`, …
- `--pad 6` transparent margin around each cut, in pixels
- `--smooth 2` rounds off small bumps in the outline and gives a crisp, anti-aliased
  edge with 1px of fringe trimmed; `0` keeps SAM's raw outline, `3` is rounder
- `--max-side 1000` largest window size; large photos are shown scaled down
- `--model`, `--config` use another SAM 2.1 checkpoint (e.g. small: `configs/sam2.1/sam2.1_hiera_s.yaml`)

## Tips

- One click is usually enough for a clear object. If the outline grabs a
  neighbour, right-click on the neighbour.
- If the outline only covers part of the object (glare, a two-tone stone),
  left-click on the missing part.
- Holes inside an object (glare) are filled automatically; only the largest
  connected piece is kept.

## Angle views and 3D model of a smooth stone

Two scripts estimate what a smooth (tumbled, polished) stone looks like from other
angles, from **one** front photo. SAM 2 finds the outline; the stone is modelled as
a rounded shape whose widest edge is that outline, and the photo is wrapped over the
front. **The sides and back are invented** from the front texture, so the results are
for demos and testing only, never for selling a real stone. Works best on rounded,
tumbled stones; crystal clusters and rough stones come out wrong.

```bash
# PNG views (front, left, right, top, side, back, transparent) and a 360° GIF
.venv/bin/python stone360.py photo.jpg -o out/agate_a --name agate_a

# the same shape as a textured 3D model (.glb), at true size
.venv/bin/python stone_glb.py photo.jpg -o out/agate_a/agate_a.glb --size-mm 28
```

`stone360.py` writes `<name>_front.png`, `_left`, `_right`, `_top`, `_side`, `_back`
(25° turns, a 90° side, the back) and `<name>_360.gif`. `stone_glb.py` writes one
`.glb`: one mesh and one JPEG texture (front on top, back below), scaled so the
longest side is `--size-mm`.

Useful options (both scripts): `--thickness` (front depth / radius, default 0.45),
`--back` (back depth relative to the front, 0.8), `--exp` (shape: lower is boxier,
0.6), `--swirl` (how far the back texture is turned, radians, π).

To look at a `.glb`: `f3d model.glb` (`sudo apt install f3d`), or Google's
`<model-viewer>` in a page served with `python3 -m http.server`; a browser does not
load a `.glb` from a page opened from disk.

### Metals

For a metal such as a gold nugget, make the model reflect like metal:

```bash
.venv/bin/python stone_glb.py nugget.jpg -o out/gold_a/gold_a.glb --size-mm 16 --metallic 1 --roughness 0.4
```

## 3D model of a faceted (step-cut) gem

`gem_glb.py` builds an emerald-cut gem from a top-view cut-out and its 8 girdle
corners: the table and step facets on top are textured with the photo, the
underside (pavilion and keel) is invented and gets the same photo, darker. A
see-through glass material was tried and dropped: `<model-viewer>` showed the page
background through it, so the underside looked white.

```bash
.venv/bin/python gem_glb.py out/quartz_b/quartz_b1.png out/quartz_b/outline_level.json -o out/quartz_b/quartz_b.glb --length-mm 12
```

`outline.json` is `[[x, y], ...]`, the 8 corners in the cut-out's pixels, clockwise
from the top-left of the top edge. Options: `--depth` (total depth / width, 0.62),
`--ior` (refractive index, quartz 1.544).
