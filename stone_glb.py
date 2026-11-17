"""Export the stone360 model of a smooth stone as a textured .glb (glTF binary).

Usage: .venv/bin/python stone_glb.py photo.jpg -o out/stone/stone_a.glb --size-mm 28

Same shape and texture as stone360.py: a closed body whose equator is the SAM 2
outline; front texture from the photo, back texture invented. One mesh, one
JPEG texture atlas (front on top, back below). Demo use only.
"""
import argparse
import io
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stone360 as s3  # noqa: E402

N_THETA, N_PHI = 256, 96          # mesh grid (vertices per hemisphere)
T_THETA, T_PHI = 2048, 1024       # texture grid (pixels per hemisphere)


def grids(big, amask, n_theta, n_phi, a):
    P, N, N_face, C, Rm, T = s3.build(big, amask, n_theta, n_phi, a.thickness, a.back, a.exp, a.swirl, 0.85)
    n = n_theta * n_phi
    shape = (n_phi, n_theta, 3)
    return P[:n].reshape(shape), P[n:].reshape(shape), C[:n].reshape(shape), C[n:].reshape(shape)


def hemisphere(P, v0, v1, flip):
    """Vertices (seam column repeated), UVs in [v0, v1] of the atlas, triangles."""
    n_phi, n_theta, _ = P.shape
    Pw = np.concatenate([P, P[:, :1]], 1)  # repeat the first column to close the seam
    u = np.arange(n_theta + 1) / n_theta + 0.5 / T_THETA
    # mesh row k sits at q = k/(n_phi-1), as does texture row q*(T_PHI-1)
    q = np.arange(n_phi) / (n_phi - 1)
    v = v0 + (v1 - v0) * (q * (T_PHI - 1) + 0.5) / T_PHI
    U, V = np.meshgrid(u, v)
    uv = np.stack([U, V], -1).reshape(-1, 2)
    W = n_theta + 1
    i = np.arange(n_phi - 1)[:, None] * W + np.arange(n_theta)[None, :]
    a, b, c, d = i, i + 1, i + W, i + W + 1
    if flip:
        tris = np.concatenate([np.stack([a, b, c], -1), np.stack([b, d, c], -1)])
    else:
        tris = np.concatenate([np.stack([a, c, b], -1), np.stack([b, c, d], -1)])
    return Pw.reshape(-1, 3), uv, tris.reshape(-1, 3)


def vertex_normals(V, F):
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n


def write_glb(path, V, Nrm, UV, F, jpeg, name):
    V = V.astype(np.float32)
    Nrm = Nrm.astype(np.float32)
    UV = UV.astype(np.float32)
    F = F.astype(np.uint32)
    chunks, views, offset = [], [], 0

    def add(data, target=None):
        nonlocal offset
        b = data.tobytes() if isinstance(data, np.ndarray) else data
        pad = (-len(b)) % 4
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(b)}
        if target:
            view["target"] = target
        views.append(view)
        chunks.append(b + b"\0" * pad)
        offset += len(b) + pad
        return len(views) - 1

    vP, vN, vT, vI = add(V, 34962), add(Nrm, 34962), add(UV, 34962), add(F, 34963)
    vImg = add(jpeg)
    gltf = {
        "asset": {"version": "2.0", "generator": "sam2_goatstone stone_glb.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3, "material": 0}]}],
        "materials": [{"name": "stone", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.35}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": vImg, "mimeType": "image/jpeg"}],
        "accessors": [
            {"bufferView": vP, "componentType": 5126, "count": len(V), "type": "VEC3",
             "min": V.min(0).tolist(), "max": V.max(0).tolist()},
            {"bufferView": vN, "componentType": 5126, "count": len(V), "type": "VEC3"},
            {"bufferView": vT, "componentType": 5126, "count": len(V), "type": "VEC2"},
            {"bufferView": vI, "componentType": 5125, "count": F.size, "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    binb = b"".join(chunks)
    total = 12 + 8 + len(js) + 8 + len(binb)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binb), 0x004E4942) + binb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--out", required=True, help="output .glb")
    ap.add_argument("--size-mm", type=float, default=28, help="longest side of the stone, mm")
    ap.add_argument("--scale", type=float, default=3)
    ap.add_argument("--thickness", type=float, default=0.45)
    ap.add_argument("--back", type=float, default=0.8)
    ap.add_argument("--exp", type=float, default=0.6)
    ap.add_argument("--swirl", type=float, default=np.pi)
    ap.add_argument("--quality", type=int, default=90)
    a = ap.parse_args()

    rgb = np.array(Image.open(a.image).convert("RGB"))
    alpha = s3.sam_mask(rgb)
    big = cv2.resize(rgb, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
    amask = cv2.resize(alpha, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_LINEAR)

    Pf, Pb, _, _ = grids(big, amask, N_THETA, N_PHI, a)
    _, _, Cf, Cb = grids(big, amask, T_THETA, T_PHI, a)

    # atlas: front on top half, back on bottom half (rows = equator -> pole)
    atlas = np.concatenate([Cf, Cb], 0).clip(0, 255).astype(np.uint8)
    atlas = cv2.resize(atlas, (2048, 2048), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(atlas).save(buf, "JPEG", quality=a.quality)

    Vf, UVf, Ff = hemisphere(Pf, 0.0, 0.5, flip=False)
    Vb, UVb, Fb = hemisphere(Pb, 0.5, 1.0, flip=True)
    V = np.concatenate([Vf, Vb])
    UV = np.concatenate([UVf, UVb])
    F = np.concatenate([Ff, Fb + len(Vf)])

    # image axes (x right, y down, z to the viewer) -> glTF (y up), metres, centred
    V = V * np.array([1, -1, 1])
    F = F[:, ::-1]  # the y flip mirrors, so turn the triangles round
    V -= (V.max(0) + V.min(0)) / 2
    V *= (a.size_mm / 1000) / (V.max(0) - V.min(0))[:2].max()

    # make every triangle face outward (centre is inside a convex-ish stone)
    fc = V[F].mean(1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    inward = (fn * fc).sum(1) < 0
    F[inward] = F[inward][:, ::-1]
    Nrm = vertex_normals(V, F)

    write_glb(a.out, V, Nrm, UV, F, buf.getvalue(), Path(a.out).stem)
    print(f"saved {a.out}: {len(V)} vertices, {len(F)} triangles, {Path(a.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
