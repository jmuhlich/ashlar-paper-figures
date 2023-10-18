import os
import sys
import argparse
import re
import itertools
import pathlib
import concurrent.futures
import tqdm
import matplotlib.pyplot as plt
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
    results = np.array([f.result() for f in futures], dtype=object)
    results = results.reshape(len(range_y), len(range_x), -1)
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


def compose(base, img, pool, brightness, in_range="image"):
    pool_apply_blocked(
        pool,
        1000,
        compose_block,
        base,
        img,
        "    composing image",
        brightness,
        in_range,
    )


def compose_block(base, img, brightness, in_range):
    img = skimage.exposure.rescale_intensity(
        img, in_range=in_range, out_range=float
    )
    block = skimage.img_as_float(base) + img[..., None] * brightness
    block = skimage.img_as_ubyte(np.clip(block, 0, 1))
    base[:] = block


def build_panel(
    img1, img2, bmask, w, out_scale, dmax, brightness, intensity_pct, pool
):
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
    det = np.linalg.det(lr.coef_)
    assert det != 0, "Degenerate matrix"
    (a, b), (c, d) = lr.coef_
    scale_y = np.linalg.norm([a, b])
    scale_x = det / scale_y
    shear = (a * c + b * d) / det
    rotation = np.arctan2(b, a)
    print("    recovered affine transform: ")
    print(f"      scale = [{scale_y:.3g} {scale_x:.3g}]")
    print(f"      shear = {shear:.3g}")
    print(f"      rotation = {rotation:.3g}")
    if np.allclose(lr.coef_, np.eye(2), atol=1e-4, rtol=1e-4):
        print("      (affine correction is trivial)")
    else:
        print("      (affine correction is non-trivial)")

    shifts = (p2 - lr.intercept_) @ np.linalg.inv(lr.coef_.T) - p1
    with np.printoptions(precision=3):
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
    kwargs = {}
    if intensity_pct:
        kwargs["in_range"] = tuple(np.percentile(img1_scaled, intensity_pct))
    compose(panel, img1_scaled, pool, brightness=brightness, **kwargs)
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


def read_tiff_channel(path, channel):
    tiff = tifffile.TiffFile(path)
    try:
        zstore = tiff.aszarr(level=0, key=channel)
    except IndexError:
        raise ValueError(
            f"Selected channel ({channel}) out of range: {path}"
        ) from None
    img = zarr.open(zstore)
    if img.ndim != 2:
        raise ValueError(f"TIFF must be 2-D or 3-D: {path}")
    return img


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
    "--image1-channel", metavar="CHANNEL", type=int, default=0,
    help="Channel number to read from image1",
)
parser.add_argument(
    "--image2-channel", metavar="CHANNEL", type=int, default=0,
    help="Channel number to read from image2",
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
    "--output-contrast-percentile", type=float, nargs=2, default=None,
    metavar=("LOWER", "UPPER"),
    help="Lower and upper brightness percentile for contrast rescaling. The"
    " reference image's brightness and contrast will be scaled to place these"
    " values at the bottom and top end of the brightness scale, respectively."
    " If not specified, the image's actual dynamic range will be used."
)
parser.add_argument(
    "--output-brightness-scale", type=float, default=1.0, metavar="SCALE",
    help="Multipler for reference image intensity in the output image.",
)
parser.add_argument(
    "--pixel-size", type=float, default=1.0, metavar="MICRONS",
    help="Size of input image pixels, in microns."
)
parser.add_argument(
    "--max-distance", type=float, default=4, metavar="MICRONS",
    help="Large flow magnitudes (outliers) will be clamped down to this value"
    " for the purpose of rendering the output image. The values saved to the"
    " .npy file will not be clamped. The value must be specified in microns"
    " (see pixel-size), NOT in pixels.",
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
parser.add_argument(
    "--display-threshold-only", action="store_true",
    help="Visualizes the foreground mask in a window, then skips the optical"
    " flow computation and all file output. The foreground is colored cyan and"
    " the background colored red."
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
z1 = read_tiff_channel(args.image1_path, args.image1_channel)
z2 = read_tiff_channel(args.image2_path, args.image2_channel)
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

if not args.display_threshold_only:
    print()
    print("Performing global image alignment")
r1 = downscale(c1, ga_downscale)
r2 = downscale(c2, ga_downscale)
if not args.display_threshold_only:
    shift = skimage.registration.phase_cross_correlation(
        r1, r2, upsample_factor=ga_downscale, return_error=False,
    )
    shift = (shift * ga_downscale).astype(int)
    print(f"    shift (y,x)={shift}")
else:
    shift = np.zeros(2, dtype=int)

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

print()
print("Computing foreground image mask")
bmax = skimage.measure.block_reduce(c1, (bsize, bsize), np.max)
bmask = bmax > block_threshold
bmask = skimage.morphology.remove_small_objects(bmask, min_size=args.area_threshold)

if args.display_threshold_only:
    s = bsize // ga_downscale
    bmask_upscaled = bmask.repeat(s, axis=1).repeat(s, axis=0)
    threshold_img = skimage.util.img_as_float(crop_to(r1, bmask_upscaled.shape))
    k = threshold_img.max() / 2
    threshold_img = (threshold_img[..., None]).repeat(3, axis=2)
    threshold_img[..., 0] += ~bmask_upscaled * k
    threshold_img[..., 1] += bmask_upscaled * k
    threshold_img[..., 2] += bmask_upscaled * k
    threshold_img = skimage.exposure.rescale_intensity(threshold_img)
    plt.imshow(threshold_img.clip(0, 1))
    plt.show()
    sys.exit()
assert np.any(bmask), "Entire image was discarded as background; please" \
    " adjust --intensity-threshold and/or --area-threshold values"

pool = concurrent.futures.ThreadPoolExecutor(len(os.sched_getaffinity(0)))

print()
print("Building output image")
panel, dist, shifts = build_panel(
    c1,
    c2,
    bmask,
    bsize,
    out_scale,
    dmax,
    args.output_brightness_scale,
    args.output_contrast_percentile,
    pool,
)
print("    saving")
skimage.io.imsave(args.output_path, panel, check_contrast=False)

if args.data_output:
    np.save(args.data_output, dist[bmask] * args.pixel_size)

pool.shutdown()
