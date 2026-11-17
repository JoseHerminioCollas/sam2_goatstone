"""Estimate all-round views of a smooth (tumbled) stone from one front photo.

Usage: .venv/bin/python stone360.py photo.jpg -o out/stone --name stone_a

1. SAM 2 finds the outline.
2. The stone is modelled as a closed, rounded body: the outline is its equator,
   front and back are superellipse domes (meridians from centre to outline).
3. Texture:
   - front: the photo, but towards the equator the texture keeps running at
     true arc length and folds back inward at the outline, so the side band
     is not stretched;
   - back: same mapping, swirled around the centre (more at the middle, none
     at the equator) and a little darker, so its banding differs from the
     front with no seam at the equator.
4. The body is turned and drawn with a z-buffer.

Everything not seen in the photo (sides, back) is invented. Demo use only.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cut  # noqa: E402


def sam_mask(rgb):
    pred = cut.load_predictor(cut.DEFAULT_MODEL, cut.DEFAULT_CONFIG)
    pred.set_image(rgb)
    h, w = rgb.shape[:2]
    pts = [[w // 2, h // 2, 1], [w // 3, h // 2, 1], [2 * w // 3, h // 2, 1]]
    return cut.predict(pred, pts, smooth=2)


def outline_radius(mask, n_theta):
    """Centroid and outline radius R(theta) (star-shaped outline)."""
    m = (mask > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    M = cv2.moments(m)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    ang = np.arctan2(c[:, 1] - cy, c[:, 0] - cx)
    rad = np.hypot(c[:, 0] - cx, c[:, 1] - cy)
    order = np.argsort(ang)
    ang, rad = ang[order], rad[order]
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    R = np.interp(theta, ang, rad, period=2 * np.pi)
    k = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
    k /= k.sum()
    R = np.convolve(np.concatenate([R[-9:], R, R[:9]]), k, mode="valid")
    return (cx, cy), theta, R


def spow(x, e):
    return np.sign(x) * np.abs(x) ** e


def build(big, mask, n_theta, n_phi, thick, back_ratio, ex, swirl, dark):
    (cx, cy), theta, R = outline_radius(mask, n_theta)
    Rm = float(R.mean())
    T = thick * Rm

    q = np.linspace(0, 1, n_phi)
    phi = (np.pi / 2) * (0.5 - 0.5 * np.cos(np.pi * q))  # 0 = equator, pi/2 = pole; dense at both ends
    f = spow(np.cos(phi), ex)  # radial fraction
    g = spow(np.sin(phi), ex)  # height fraction

    # arc length from the pole along a mean meridian, in units of Rm
    dr = np.diff(f)
    dz = np.diff(g) * T / Rm
    seg = np.hypot(dr, dz)
    s_from_eq = np.concatenate([[0], np.cumsum(seg)])  # from equator
    s = s_from_eq[-1] - s_from_eq  # from pole
    # texture radial fraction: photo where the surface faces the camera,
    # true arc length near the rim, folded back inward past the outline
    f0 = 0.7
    i0 = np.argmin(np.abs(f - f0))
    tr = np.where(f <= f[i0], f, f[i0] + (s - s[i0]))
    tr = np.where(tr > 1, 2 - tr, tr)
    tr = np.clip(tr, 0, 1)

    TH, TR = np.meshgrid(theta, tr)  # (n_phi, n_theta)
    Rg = np.broadcast_to(R[None, :], TH.shape)

    # photo with the glare painted out, for the back
    hsv0 = cv2.cvtColor(big, cv2.COLOR_RGB2HSV)
    gl = ((hsv0[..., 1] < 110) & (hsv0[..., 2] > 90)).astype(np.uint8)
    gl = cv2.dilate(gl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    # fill the glare with the texture from the opposite side of the stone
    h, w = big.shape[:2]
    opp = cv2.warpAffine(big, np.float32([[-1, 0, 2 * cx], [0, -1, 2 * cy]]), (w, h),
                         borderMode=cv2.BORDER_REFLECT)
    wgl = cv2.GaussianBlur(gl.astype(np.float32), (0, 0), 5)[..., None]
    big_back = (big * (1 - wgl) + opp * wgl).astype(np.uint8)

    def sample(th, src=big):
        mx = (cx + TR * Rg * np.cos(th)).astype(np.float32)
        my = (cy + TR * Rg * np.sin(th)).astype(np.float32)
        return cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    tex_front = sample(TH).astype(np.float32)
    # back: swirl grows from 0 at the equator to `swirl` at the pole
    tau = (phi / (np.pi / 2))[:, None]
    ramp = np.clip(tau / 0.5, 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)  # rigid turn over the middle, twist only in a ring
    tex_back = sample(TH + swirl * ramp, big_back)
    tex_back = tex_back.astype(np.float32) * dark
    # same at the equator so there is no seam
    w = np.clip(tau / 0.15, 0, 1)[..., None]
    tex_back = tex_front * (1 - w) + tex_back * w

    F = f[:, None]
    G = g[:, None]
    X = F * Rg * np.cos(TH)
    Y = F * Rg * np.sin(TH)
    Zf = G * T * np.ones_like(TH)
    Zb = -G * T * back_ratio * np.ones_like(TH)

    def normals(X, Y, Z):
        dXa, dYa, dZa = (np.gradient(a, axis=1) for a in (X, Y, Z))
        dXb, dYb, dZb = (np.gradient(a, axis=0) for a in (X, Y, Z))
        n = np.cross(np.stack([dXa, dYa, dZa], -1), np.stack([dXb, dYb, dZb], -1))
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        # point away from the body
        p = np.stack([X, Y, Z], -1)
        n *= np.sign((n * p).sum(-1, keepdims=True) + 1e-9)
        return n

    Pf = np.stack([X, Y, Zf], -1).reshape(-1, 3)
    Pb = np.stack([X, Y, Zb], -1).reshape(-1, 3)
    Nf = normals(X, Y, Zf).reshape(-1, 3)
    Nb = normals(X, Y, Zb).reshape(-1, 3)
    P = np.concatenate([Pf, Pb]).astype(np.float32)
    N = np.concatenate([Nf, Nb]).astype(np.float32)
    C = np.concatenate([tex_front.reshape(-1, 3), tex_back.reshape(-1, 3)]).astype(np.float32)
    # "lighting already in the texture": the light each point had when facing the camera
    N_face = N.copy()
    N_face[:, 2] = np.abs(N_face[:, 2])
    return P, N, N_face, C, Rm, T


def render(P, N, N_face, C, yaw, pitch, size, scale, light=(-0.5, -0.6, 0.65)):
    cyw, syw = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    Ry = np.array([[cyw, 0, syw], [0, 1, 0], [-syw, 0, cyw]], np.float32)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    Rot = Rx @ Ry
    Pr = P @ Rot.T
    Nr = N @ Rot.T
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    before = np.clip(N_face @ L, 0.1, 1)
    after = np.clip(Nr @ L, 0.0, 1)
    col = np.clip(C * (0.6 + 0.4 * after / before)[:, None], 0, 255)

    vis = Nr[:, 2] > -0.05  # back-face cull
    u = np.round(Pr[vis, 0] * scale + size / 2).astype(np.int32)
    v = np.round(Pr[vis, 1] * scale + size / 2).astype(np.int32)
    d = Pr[vis, 2]
    col = col[vis]
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    u, v, d, col = u[ok], v[ok], d[ok], col[ok]
    o = np.argsort(d)
    img = np.zeros((size, size, 3), np.float32)
    cov = np.zeros((size, size), np.uint8)
    img[v[o], u[o]] = col[o]
    cov[v[o], u[o]] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled = cv2.morphologyEx(cov, cv2.MORPH_CLOSE, k)
    holes = (filled == 1) & (cov == 0)
    if holes.any():
        s = cv2.blur(img * cov[..., None], (5, 5))
        w = cv2.blur(cov.astype(np.float32), (5, 5))[..., None]
        img[holes] = (s / np.maximum(w, 1e-3))[holes]
    return img, filled


def finish(img, cov, out_size):
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    a = cut.smooth_edge((cv2.resize(cov.astype(np.float32), (out_size, out_size),
                                    interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8), 1.5)
    return img.clip(0, 255).astype(np.uint8), a


def crop_rgba(img, alpha, pad=8):
    a = (alpha * 255).round().astype(np.uint8)
    ys, xs = np.nonzero(a > 127)
    y0, y1 = max(ys.min() - pad, 0), ys.max() + pad + 1
    x0, x1 = max(xs.min() - pad, 0), xs.max() + pad + 1
    return Image.fromarray(np.dstack([img, a])[y0:y1, x0:x1], "RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--name", default="stone")
    ap.add_argument("--scale", type=float, default=3, help="upscale the photo first")
    ap.add_argument("--thickness", type=float, default=0.45, help="front half-thickness / radius")
    ap.add_argument("--back", type=float, default=0.8, help="back depth relative to front")
    ap.add_argument("--exp", type=float, default=0.6, help="superellipse exponent, <1 = boxier")
    ap.add_argument("--swirl", type=float, default=np.pi, help="back texture turn at the centre, radians")
    ap.add_argument("--ss", type=int, default=2, help="supersampling")
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rgb = np.array(Image.open(a.image).convert("RGB"))
    alpha = sam_mask(rgb)
    big = cv2.resize(rgb, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
    amask = cv2.resize(alpha, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_LINEAR)

    P, N, N_face, C, Rm, T = build(big, amask, 3000, 1400, a.thickness, a.back, a.exp, a.swirl, 0.85)
    out_size = int(max(np.abs(P).max(0)) * 2.3)
    ss = a.ss

    views = {"front": (0, 0), "left": (25, 0), "right": (-25, 0), "top": (0, -25),
             "side": (90, 0), "back": (180, 0)}
    for label, (yaw, pitch) in views.items():
        img, cov = render(P, N, N_face, C, yaw, pitch, out_size * ss, ss)
        crop_rgba(*finish(img, cov, out_size)).save(out / f"{a.name}_{label}.png")
        print("saved", label, flush=True)

    frames = []
    for yaw in range(0, 360, 10):
        img, cov = render(P, N, N_face, C, yaw, -10, out_size * ss, ss)
        im, al = finish(img, cov, out_size)
        bg = np.full_like(im, 245)
        comp = (im * al[..., None] + bg * (1 - al[..., None])).astype(np.uint8)
        frames.append(Image.fromarray(comp).resize((out_size // 2, out_size // 2), Image.LANCZOS))
    frames[0].save(out / f"{a.name}_360.gif", save_all=True, append_images=frames[1:], duration=90, loop=0)
    print("saved 360 gif", flush=True)


if __name__ == "__main__":
    main()
