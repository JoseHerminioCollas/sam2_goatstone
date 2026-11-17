"""Textured .glb of a step-cut (emerald cut) gem from a top-view photo.

The girdle is the gem's outline (8 corners, image pixels). The crown is two step
rings and a flat table, textured with the photo by projecting it from above, so
the top view matches the photo. The pavilion steps down to a keel line, in a
the same photo, slightly darker (a see-through material shows the page
background through it, so the pavilion looked white). Flat facets.
The pavilion and the proportions are invented. Demo use only.

Usage: gem_glb.py cutout.png outline.json -o gem.glb --length-mm 12
  cutout.png    the transparent cut-out (top view)
  outline.json  [[x, y], ...] the 8 girdle corners in the cut-out's pixels
"""
import argparse
import io
import json
import struct

import cv2
import numpy as np
from PIL import Image

# step rings as (scale x, scale y, height as a fraction of the total depth), from girdle to keel
CROWN = [(1.00, 1.00, 0.020), (0.86, 0.80, 0.130), (0.64, 0.52, 0.220)]  # last ring = table
PAVILION = [(1.00, 1.00, -0.020), (0.80, 0.70, -0.300), (0.55, 0.36, -0.600), (0.30, 0.00, -0.780)]


def ring(P, c, sx, sy):
    return c + (P - c) * np.array([sx, sy])


def facets(rings_px, rings_z, close_top):
    """Triangles (as vertex triples in px + z) between consecutive rings, and a fan cap."""
    tris = []
    for (A, za), (B, zb) in zip(zip(rings_px, rings_z), zip(rings_px[1:], rings_z[1:])):
        n = len(A)
        for i in range(n):
            j = (i + 1) % n
            a0, a1 = (*A[i], za), (*A[j], za)
            b0, b1 = (*B[i], zb), (*B[j], zb)
            tris.append([a0, a1, b1])
            tris.append([a0, b1, b0])
    if close_top:
        T, zt = rings_px[-1], rings_z[-1]
        for i in range(1, len(T) - 1):
            tris.append([(*T[0], zt), (*T[i], zt), (*T[i + 1], zt)])
    return np.array(tris, np.float64)  # (n, 3, 3)


def to_mesh(tris_px, px_to_mm, centre, size_px, outward_ref):
    """Flat-shaded mesh: unique vertices per triangle, positions in metres (y up), UVs from px."""
    v_px = tris_px.reshape(-1, 3)
    pos = np.stack([(v_px[:, 0] - centre[0]) * px_to_mm,
                    -(v_px[:, 1] - centre[1]) * px_to_mm,
                    v_px[:, 2]], 1) / 1000.0
    tri = pos.reshape(-1, 3, 3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    # face outward: away from the gem's centre
    flip = (fn * (tri.mean(1) - outward_ref)).sum(1) < 0
    tri[flip] = tri[flip][:, ::-1]
    v_px3 = v_px.reshape(-1, 3, 3)
    v_px3[flip] = v_px3[flip][:, ::-1]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    keep = np.linalg.norm(fn, axis=1) > 1e-14  # drop degenerate triangles at the keel
    tri, v_px3, fn = tri[keep], v_px3[keep], fn[keep]
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    pos = tri.reshape(-1, 3)
    nrm = np.repeat(fn, 3, axis=0)
    uv = np.stack([v_px3.reshape(-1, 3)[:, 0] / size_px[0], v_px3.reshape(-1, 3)[:, 1] / size_px[1]], 1)
    idx = np.arange(len(pos), dtype=np.uint32)
    return pos.astype(np.float32), nrm.astype(np.float32), uv.astype(np.float32), idx


def write_glb(path, prims, materials, jpeg, name, extensions):
    chunks, views, accessors, offset = [], [], [], 0

    def add(data, target=None):
        nonlocal offset
        b = data.tobytes() if isinstance(data, np.ndarray) else data
        pad = (-len(b)) % 4
        v = {"buffer": 0, "byteOffset": offset, "byteLength": len(b)}
        if target:
            v["target"] = target
        views.append(v)
        chunks.append(b + b"\0" * pad)
        offset += len(b) + pad
        return len(views) - 1

    def acc(arr, typ, comp, target, minmax=False):
        a = {"bufferView": add(arr, target), "componentType": comp, "count": len(arr), "type": typ}
        if minmax:
            a["min"], a["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        accessors.append(a)
        return len(accessors) - 1

    gl_prims = []
    for (pos, nrm, uv, idx), mat in prims:
        attrs = {"POSITION": acc(pos, "VEC3", 5126, 34962, True), "NORMAL": acc(nrm, "VEC3", 5126, 34962)}
        if uv is not None:
            attrs["TEXCOORD_0"] = acc(uv, "VEC2", 5126, 34962)
        gl_prims.append({"attributes": attrs, "indices": acc(idx, "SCALAR", 5125, 34963), "material": mat})
    img_view = add(jpeg)
    gltf = {
        "asset": {"version": "2.0", "generator": "sam2_goatstone gem_glb.py"},
        "extensionsUsed": extensions,
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": gl_prims}],
        "materials": materials,
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": img_view, "mimeType": "image/jpeg"}],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": offset}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    binb = b"".join(chunks)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binb)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binb), 0x004E4942) + binb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cutout")
    ap.add_argument("outline")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--length-mm", type=float, default=12, help="long side of the girdle, mm")
    ap.add_argument("--depth", type=float, default=0.62, help="total depth / width of the girdle")
    ap.add_argument("--ior", type=float, default=1.544, help="refractive index (quartz 1.544)")
    ap.add_argument("--quality", type=int, default=90)
    a = ap.parse_args()

    im = Image.open(a.cutout).convert("RGBA")
    W, H = im.size
    P = np.array(json.load(open(a.outline)), np.float64)
    c = P.mean(0)
    px_to_mm = a.length_mm / (P[:, 0].max() - P[:, 0].min())
    width_mm = (P[:, 1].max() - P[:, 1].min()) * px_to_mm
    depth_mm = a.depth * width_mm

    # texture: the photo's colours, with the background filled by the nearest gem colour
    rgb = np.array(im.convert("RGB"))
    outside = (np.array(im)[..., 3] < 250).astype(np.uint8)
    outside = cv2.dilate(outside, np.ones((5, 5), np.uint8))  # also the soft edge
    rgb = cv2.inpaint(rgb, outside * 255, 7, cv2.INPAINT_TELEA)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=a.quality)

    crown_r = [ring(P, c, sx, sy) for sx, sy, _ in CROWN]
    crown_z = [z * depth_mm for _, _, z in CROWN]
    pav_r = [ring(P, c, sx, sy) for sx, sy, _ in PAVILION]
    pav_z = [z * depth_mm for _, _, z in PAVILION]

    centre_m = np.array([0, 0, (CROWN[-1][2] + PAVILION[-1][2]) / 2 * depth_mm / 1000])
    crown = facets(crown_r, crown_z, close_top=True)
    girdle = facets([crown_r[0], pav_r[0]], [crown_z[0], pav_z[0]], close_top=False)
    pavilion = facets(pav_r, pav_z, close_top=False)

    top = to_mesh(np.concatenate([crown, girdle]), px_to_mm, c, (W, H), centre_m)
    bottom = to_mesh(pavilion, px_to_mm, c, (W, H), centre_m)
    # tint for the pavilion: the mean colour of the gem in the photo
    alpha = np.array(im)[..., 3] > 200
    tint = (np.array(im)[..., :3][alpha].mean(0) / 255).tolist()
    tint_light = [min(1.0, 0.35 + 0.75 * t) for t in tint]

    materials = [
        {"name": "crown (photo)", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.06},
         "extensions": {"KHR_materials_ior": {"ior": a.ior}}},
        {"name": "pavilion (photo, darker)", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "baseColorFactor": [0.82, 0.82, 0.78, 1.0],
            "metallicFactor": 0.0, "roughnessFactor": 0.06},
         "extensions": {"KHR_materials_ior": {"ior": a.ior}}},
    ]
    exts = ["KHR_materials_ior"]
    write_glb(a.out, [(top, 0), (bottom, 1)], materials, buf.getvalue(),
              a.out.rsplit("/", 1)[-1].rsplit(".", 1)[0], exts)
    n = len(top[3]) // 3 + len(bottom[3]) // 3
    print(f"saved {a.out}: {n} facets (triangles), {a.length_mm:.1f} x {width_mm:.1f} x "
          f"{depth_mm * (CROWN[-1][2] - PAVILION[-1][2]):.1f} mm")


if __name__ == "__main__":
    main()
