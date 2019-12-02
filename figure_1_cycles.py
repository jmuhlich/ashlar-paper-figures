import warnings
import sys
import os
import itertools
import pathlib
import concurrent.futures
import gc
import numpy as np
import skimage.color
import skimage.feature
import skimage.io
import skimage.morphology
import skimage.transform


def pool_apply_blocked(pool, w, func, img1, img2, *args):
    assert img1.shape[:2] == img2.shape[:2]
    range_y = range(0, img1.shape[0], w)
    range_x = range(0, img2.shape[1], w)
    futures = []
    for y, x in itertools.product(range_y, range_x):
        f = pool.submit(
            func, img1[y:y+w, x:x+w], img2[y:y+w, x:x+w], *args
        )
        futures.append(f)
    show_progress(futures)
    results = [f.result() for f in futures]
    results = np.reshape(results, (len(range_y), len(range_x), -1))
    return results


def show_progress(futures):
    n = len(futures)
    while True:
        done, not_done = concurrent.futures.wait(futures, timeout=1)
        c = len(done)
        percent = int(np.round(c / n * 100))
        print(f"\r        {percent}% completed ({c}/{n})", end="")
        if c == n:
            print()
            break


def optical_flow(img1, img2, w, pool):
    assert img1.dtype == img2.dtype
    results = pool_apply_blocked(
        pool, w, skimage.feature.register_translation, img1, img2, 10
    )
    shifts = results[..., 0]
    # Unwrap inner-most nested numpy arrays (register_translation shifts).
    shifts = np.stack(shifts.ravel()).reshape(shifts.shape + (-1,))
    angle = np.arctan2(shifts[..., 0], shifts[..., 1])
    distance = np.linalg.norm(shifts, axis=2)
    return angle, distance


def colorize(angle, distance):
    assert angle.shape == distance.shape
    assert np.min(distance) >= 0
    assert np.max(distance) <= 1
    img_lch = np.zeros(angle.shape[:2] + (3,), np.float32)
    img_lch[..., 0] = distance * 30 + 20
    img_lch[..., 1] = distance * 100
    img_lch[..., 2] = angle
    img = skimage.color.lab2rgb(skimage.color.lch2lab(img_lch))
    img = skimage.img_as_ubyte(img)
    return img


def compose(base, img, pool):
    in_range = skimage.exposure.exposure.intensity_range(img)
    pool_apply_blocked(pool, 1000, compose_block, base, img, in_range)


def compose_block(base, img, in_range):
    img = skimage.exposure.rescale_intensity(img, in_range=in_range)
    block = skimage.img_as_float(base) + skimage.img_as_float(img)[..., None]
    block = skimage.img_as_ubyte(np.clip(block, 0, 1))
    base[:] = block


def build_panel(img1, img2, bmask, w, out_scale, dmax, pool):
    print("    computing optical flow")
    angle, distance = optical_flow(img1, img2, w, pool)
    print("    colorizing")
    dnorm = np.clip(distance, 0, dmax) / dmax
    heatmap_small = colorize(angle, dnorm) * bmask[..., None]
    heatmap = heatmap_small.repeat(w, axis=1).repeat(w, axis=0)
    print("    composing image")
    panel = heatmap[:img1.shape[0], :img1.shape[1], :]
    compose(panel, img1, pool)
    panel = skimage.transform.rescale(
        panel, 1 / out_scale, multichannel=True, anti_aliasing=False
    )
    panel = skimage.img_as_ubyte(panel)
    return panel


warnings.filterwarnings(
    'ignore', 'Possible precision loss', UserWarning, '^skimage\.util\.dtype'
)

# First cycle Ashlar image.
path1 = pathlib.Path(sys.argv[1])
# Later cycle comparison image, independently stitched (with any algorithm).
path2a = pathlib.Path(sys.argv[2])
# Later cycle Ashlar image, simultaneously stitched with first cycle.
path2b = pathlib.Path(sys.argv[3])

# Small crop
# x1, x2 = 100, 10100
# y1, y2 = 8000, 16000
# ax1, ax2 = 3000, 4000
# ay1, ay2 = 3000, 4000

# Medium crop
# x1, x2 = 100, 20100
# y1, y2 = 8000, 24000
# ax1, ax2 = 11000, 12000
# ay1, ay2 = 4000, 5000

# Full image
x1, x2 = 100, 35000
y1, y2 = 100, 24000
ax1, ax2 = 18000-400, 18000+400
ay1, ay2 = 7600-400, 7600+100

bsize = 200
out_scale = 20
#dmax = 25
dmax = 5
block_threshold = 8000

assert bsize % out_scale == 0, "bsize must be a multiple of out_scale"

print("Performing global image alignment")
c1 = skimage.io.imread(str(path1))[y1:y2, x1:x2]
img2a = skimage.io.imread(str(path2a))
c2a = img2a[y1:y2, x1:x2]
shift, _, _ = skimage.feature.register_translation(
    c1[ay1:ay2, ax1:ax2], c2a[ay1:ay2, ax1:ax2]
)
shift = np.round(shift).astype(int)
c2a = img2a[y1-shift[0]:y2-shift[0], x1-shift[1]:x2-shift[1]]

bmax = skimage.measure.block_reduce(c1, (bsize, bsize), np.max)
bmask = bmax > block_threshold
bmask = skimage.morphology.remove_small_objects(bmask, min_size=4)
bmask = skimage.morphology.remove_small_holes(bmask, area_threshold=100)

pool = concurrent.futures.ThreadPoolExecutor(len(os.sched_getaffinity(0)))

print("Building first panel")
panel_a = build_panel(c1, c2a, bmask, bsize, out_scale, dmax, pool)
print("    saving")
skimage.io.imsave('Figure_1E.tif', panel_a, check_contrast=False)
del img2a, c2a, panel_a
gc.collect()

try:
    c2b = skimage.io.imread(str(path2b))[y1:y2, x1:x2]
    print("Building second panel")
    panel_b = build_panel(c1, c2b, bmask, bsize, out_scale, dmax, pool)
    print("    saving")
    skimage.io.imsave('Figure_1F.tif', panel_b, check_contrast=False)
except IOError:
    print("Skipping second panel (could not read file)")

pool.shutdown()
