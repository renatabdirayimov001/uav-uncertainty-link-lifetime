"""Remove a uniform background from a stock illustration and split it into its isolated objects."""
import sys, numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
def remove_bg(path, tol=38, maxside=None):
    im = Image.open(path).convert('RGB')
    if maxside and max(im.size) > maxside:
        s = maxside / max(im.size); im = im.resize((int(im.size[0]*s), int(im.size[1]*s)), Image.LANCZOS)
    a = np.asarray(im).astype(float)
    border = np.r_[a[0], a[-1], a[:, 0], a[:, -1]]
    bgc = np.median(border, axis=0)
    dist = np.sqrt(((a - bgc) ** 2).sum(2))
    cand = dist < tol
    lab, n = ndimage.label(cand)
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    bg = np.isin(lab, list(edge))
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    # soft edge: partially transparent where colour is close to background
    soft = np.clip((dist - tol * 0.5) / (tol * 0.5), 0, 1)
    edgeband = ndimage.binary_dilation(bg, iterations=2) & ~bg
    alpha = np.where(edgeband, (soft * 255).astype(np.uint8), alpha)
    alpha = np.asarray(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(0.8)))
    return Image.fromarray(np.dstack([a.astype(np.uint8), alpha]), 'RGBA'), bgc
def components(rgba, min_area=0.002, gap=6):
    al = np.asarray(rgba)[:, :, 3] > 40
    lab, n = ndimage.label(ndimage.binary_dilation(al, iterations=gap))
    objs = ndimage.find_objects(lab); out = []
    H, W = al.shape
    for k, sl in enumerate(objs):
        if sl is None: continue
        area = (lab[sl] == k + 1).sum()
        if area < min_area * H * W: continue
        crop = rgba.crop((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
        m = (lab[sl] == k + 1)
        c = np.asarray(crop).copy(); c[:, :, 3] = np.where(m, c[:, :, 3], 0)
        out.append(((sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2, Image.fromarray(c, 'RGBA')))
    out.sort(key=lambda o: (round(o[0] / (H / 8)), o[1]))
    return out
