import os
import argparse
import re
import itertools
import pathlib
import concurrent.futures
import tqdm
import numpy as np
import skimage.color
import skimage.registration
import skimage.morphology
import skimage.measure
import skimage.util
import tifffile
import sklearn.linear_model
import threadpoolctl
import zarr


def pool_apply_blocked(pool, w, func, img1, img2, desc, *args, **kwargs):
    assert img1.shape[:2] == img2.shape[:2]
    range_y = range(0, img1.shape[0], w)
    range_x = range(0, img2.shape[1], w)
    futures = []
    for y, x in itertools.product(range_y, range_x):
        f = pool.submit(
            func, img1[y:y+w, x:x+w], img2[y:y+w, x:x+w], *args, **kwargs
        )
        futures.append(f)
    show_progress(futures, desc)
    results = [f.result() for f in futures]
    results = np.reshape(results, (len(range_y), len(range_x), -1))
    return results


def show_progress(futures, desc=None):
    n = len(futures)
    progress = tqdm.tqdm(
        concurrent.futures.as_completed(futures), total=n, desc=desc
    )
    for _ in progress:
        pass


def optical_flow(img1, img2, w, pool):
    assert img1.dtype == img2.dtype
    results = pool_apply_blocked(
        pool, w, skimage.registration.phase_cross_correlation, img1, img2,
        "    computing optical flow", upsample_factor=10,
    )
    shifts = results[..., 0]
    # Unwrap inner-most nested numpy arrays (register_translation shifts).
    shifts = np.stack(shifts.ravel()).reshape(shifts.shape + (-1,))
    # skimage phase correlation tells us how to shift the second image to match
    # the first, but we want the opposite i.e. how the second image has been
    # shifted with respect to the first. So we reverse the shifts.
    shifts = -shifts
    return shifts


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
    pool_apply_blocked(
        pool, 1000, compose_block, base, img, "    composing image", in_range,
    )


def compose_block(base, img, in_range):
    img = skimage.exposure.rescale_intensity(img, in_range=in_range)
    block = skimage.img_as_float(base) + skimage.img_as_float(img)[..., None]
    block = skimage.img_as_ubyte(np.clip(block, 0, 1))
    base[:] = block


def build_panel(img1, img2, bmask, w, out_scale, dmax, pool):
    assert w % out_scale == 0
    assert np.all(np.mod(img1.shape, w) == 0)
    shifts = optical_flow(img1, img2, w, pool)
    p1 = np.dstack(np.meshgrid(
        range(shifts.shape[0]), range(shifts.shape[1]), indexing='ij'
    ))
    p1 *= w
    p2 = p1 + shifts
    lr = sklearn.linear_model.LinearRegression()
    lr.fit(p1[bmask], p2[bmask])
    shifts = (p2 - lr.intercept_) @ np.linalg.inv(lr.coef_.T) - p1
    print("    mean shift:", np.mean(shifts, axis=(0, 1)))
    print("    median shift:", np.median(shifts, axis=(0,1)))
    angle = np.arctan2(shifts[..., 0], shifts[..., 1])
    distance = np.linalg.norm(shifts, axis=2)
    print("    colorizing")
    dnorm = np.clip(distance, 0, dmax) / dmax
    heatmap_small = colorize(angle, dnorm) * bmask[..., None]
    hs = w // out_scale
    panel = heatmap_small.repeat(hs, axis=1).repeat(hs, axis=0)
    img1_scaled = downscale(img1, out_scale)
    compose(panel, img1_scaled, pool)
    return panel, distance, shifts


def crop_to(img, shape, offset=None):
    begin = np.array(offset if offset is not None else [0, 0])
    end = begin + shape
    if any(end > img.shape):
        raise ValueError("offset+shape is larger than image size")
    out = img[begin[0]:end[0], begin[1]:end[1]]
    # Above check should have handled this, but let's be sure.
    assert out.shape == tuple(shape)
    return out


def mean_round_sametype(a, axis=None):
    """Compute mean, round, and cast back to original dtype (typically int)."""
    return a.mean(axis=axis).round().astype(a.dtype)


def block_reduce_nopad(image, block_size, func=np.sum, cval=0):
    """Like block_reduce but requires image.shape is multiple of block_size"""
    if len(block_size) != image.ndim:
        raise ValueError(
            "`block_size` must have the same length as `image.shape`."
        )
    # This check lets us skip calling np.pad, which always returns a copy.
    if (np.array(image.shape) % block_size).any():
        raise ValueError(
            "`image.shape` must be an integer multiple of `block_size`."
        )
    blocked = skimage.util.view_as_blocks(image, block_size)
    return func(blocked, axis=tuple(range(image.ndim, blocked.ndim)))


def downscale(image, block_width):
    """Downscale 2D or 3D image using as little extra memory as possible."""
    if image.shape[0] % block_width or image.shape[1] % block_width:
        raise ValueError(
            "`image` width and height must be a multiple of `block_width`."
        )
    block_size = (block_width, block_width)
    if image.ndim == 2:
        pass
    elif image.ndim == 3:
        block_size = block_size + (1,)
    else:
        raise ValueError("`image` must be 2-D or 3-D")
    return block_reduce_nopad(image, block_size, func=mean_round_sametype)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Compute block-based optical flow analysis on two images",
)
parser.add_argument(
    "image1_path", metavar="image1.tif", type=pathlib.Path,
    help="First (fixed) image path",
)
parser.add_argument(
    "image2_path", metavar="image2.tif", type=pathlib.Path,
    help="Second (moving) image TIFF path",
)
parser.add_argument(
    "output_path", metavar="output.tif", type=pathlib.Path,
    help="Output image path",
)
parser.add_argument(
    "--data-output", metavar="data.npy", type=pathlib.Path,
    help="Output flow magnitudes NumPy array (.npy) path",
)
parser.add_argument(
    "--crop", metavar="LEFT,RIGHT,TOP,BOTTOM",
    help="Pixel coordinates for cropping the input images before processing"
)
parser.add_argument(
    "--registration-downsample", type=int, default=10,
    help="Factor by which to downsample the images before performing the"
    " initial global rigid registration step. Increasing this value reduces"
    " RAM and CPU usage at the expense of registration accuracy.",
)
parser.add_argument(
    "--block-size", type=int, default=200,
    help="Width in pixels of the square blocks used to compute the coarse"
    " optical flow field. Each block will be assigned a single flow direction"
    " and magnitude. Must be a multiple of output-downsample (see below).",
)
parser.add_argument(
    "--output-downsample", type=int, default=20,
    help="Factor by which to downsample the final output image. Must be an"
    " integer factor of block-size (see above)."
)
parser.add_argument(
    "--max-distance", type=float, default=6,
    help="Large flow magnitudes (outliers) will be clamped down to this value"
    " for the purpose of rendering the output image. The values saved to the"
    " .npy file will not be clamped.",
)
parser.add_argument(
    "--intensity-threshold", type=float, default=8000,
    help="Image blocks with no pixel brighter than this value will be considered"
    " background and excluded from the optical flow analysis. These blocks will"
    " be fully black in the output image and will not be represented in the"
    " data-output file (if used).",
)
parser.add_argument(
    "--area-threshold", type=int, default=7,
    help="Isolated regions of the image smaller than this number of blocks will"
    " be considered background and excluded from analysis (see the description"
    " of intensity-threshold above for what this exclusion entails).",
)
args = parser.parse_args()

threadpoolctl.threadpool_limits(1)

ga_downscale = args.registration_downsample
bsize = args.block_size
out_scale = args.output_downsample
dmax = args.max_distance
block_threshold = args.intensity_threshold

assert args.image1_path.suffix.lower().endswith(".tif")
assert args.image2_path.suffix.lower().endswith(".tif")
assert args.output_path.suffix.lower().endswith(".tif")
if args.data_output:
    assert args.data_output.suffix.endswith(".npy")
assert bsize % out_scale == 0, "block_size must be a multiple of output_downsample"
if args.crop:
    cmatch = re.match(r"(\d+),(\d+),(\d+),(\d+)$", args.crop)
    assert cmatch, "crop must be 4 integer values separated by commas (no spaces)"
    args.crop = [int(x) for x in cmatch.groups()]

print("Loading images")
tiff1 = tifffile.TiffFile(args.image1_path)
tiff2 = tifffile.TiffFile(args.image2_path)
z1 = zarr.open(tiff1.aszarr())
z2 = zarr.open(tiff2.aszarr())
if args.crop:
    cxmin, cxmax, cymin, cymax = args.crop
    assert cxmin >= 0 and cymin >= 0, "LEFT and TOP crop values must be >= 0"
    for za, iname in (z1, 'image1'), (z2, 'image2'):
        assert cxmax < za.shape[1], f"RIGHT crop value exceeds {iname} width"
        assert cymax < za.shape[0], f"BOTTOM crop value exceeds {iname} height"
    img1 = z1[cymin:cymax+1, cxmin:cxmax+1]
    img2 = z2[cymin:cymax+1, cxmin:cxmax+1]
else:
    img1 = z1[:]
    img2 = z2[:]
its = np.minimum(img1.shape, img2.shape)
its_round = its // ga_downscale * ga_downscale
c1 = crop_to(img1, its_round)
c2 = crop_to(img2, its_round)

print("Performing global image alignment")
r1 = downscale(c1, ga_downscale)
r2 = downscale(c2, ga_downscale)
shift = skimage.registration.phase_cross_correlation(
    r1, r2, upsample_factor=ga_downscale, return_error=False,
)
shift = (shift * ga_downscale).astype(int)
print(f"Global shift for independent stitch is x,y={shift[::-1]}")

border = np.abs(shift)
offset1 = np.zeros(2, int)
offset2 = -shift
for d in 0, 1:
    if offset2[d] < 0:
        offset1[d] = shift[d]
        offset2[d] = 0

shape = (its - border) // bsize * bsize
c1 = crop_to(img1, shape, offset1)
c2 = crop_to(img2, shape, offset2)

bmax = skimage.measure.block_reduce(c1, (bsize, bsize), np.max)
bmask = bmax > block_threshold
bmask = skimage.morphology.remove_small_objects(bmask, min_size=args.area_threshold)

pool = concurrent.futures.ThreadPoolExecutor(len(os.sched_getaffinity(0)))

print()
print("Building output image")
panel, dist, shifts = build_panel(c1, c2, bmask, bsize, out_scale, dmax, pool)
print("    saving")
skimage.io.imsave(args.output_path, panel, check_contrast=False)

if args.data_output:
    np.save(args.data_output, dist[bmask])

pool.shutdown()
