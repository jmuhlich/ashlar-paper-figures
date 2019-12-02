import warnings
import sys
import pathlib
import numpy as np
import skimage.color, skimage.filters, skimage.exposure
import attr
import ashlar, ashlar.reg
from ashlar.bioformats import BioformatsReader


def extent(plane):
    v1 = plane.bounds.vector1
    v2 = plane.bounds.vector2
    return (v1.x, v2.x, v2.y, v1.y)

def colorize(img, hue):
    if not np.issubdtype(img.dtype, np.floating) or img.ndim != 2:
        raise ValueError("img must be a 2-D floating point image")
    img_lch = np.empty(img.shape + (3,))
    img_lch[..., 0] = img * 50
    img_lch[..., 1] = img * 100
    img_lch[..., 2] = hue
    img = skimage.color.lab2rgb(skimage.color.lch2lab(img_lch))
    return img

def idx2hue(i, n):
    return i / n * (2 * np.pi) * 0.8 + 1.8


warnings.filterwarnings(
    'ignore', 'Possible precision loss', UserWarning, '^skimage\.util\.dtype'
)

input_dir = pathlib.Path(sys.argv[1])
file_path = input_dir / 'Scan_20190510_220028_01x4x00696.rcpnl'
if not file_path.exists():
    print(f"Could not find {file_path.name} in {input_dir} (expected Z122_PickSeq_2 dir)")
    sys.exit()

tiles = [506, 507, 477, 478]

r1 = BioformatsReader.from_path(file_path)
ts1 = r1.tileset
rp1 = ashlar.RegistrationProcess(
    tileset=ts1, channel_number=0, overlap_minimum_size=40
)

tcs = [rp1.get_tile(t) for t in tiles]

origin = np.min(ts1.positions[tiles], axis=0)
scale = tcs[0].plane.pixel_size
mshape = tuple(np.array(tcs[0].plane.image.shape) * 2) + (3,)
ccenter = tcs[0].plane.intersection(tcs[3].plane).bounds.center - origin
cshape = ashlar.Vector(200, 250)
cmin = ((ccenter - cshape) / scale).astype(int)
cmax = ((ccenter + cshape) / scale).astype(int)

alignments = [
    rp1.compute_neighbor_intersection(tiles[a], tiles[b])
    for a, b in [[0, 1], [0, 2], [2, 3]]
]
ashift_1 = alignments[0].get_shift(tiles[1])
ashift_2 = alignments[1].get_shift(tiles[2])
ashift_3 = alignments[2].get_shift(tiles[3])
shifts = [ashlar.Vector(0, 0), ashift_1, ashift_2, ashift_2 + ashift_3]

rmax = np.percentile(
    np.dstack([skimage.img_as_float32(t.plane.image) for t in tcs]),
    99.99
)

mosaic1 = np.zeros(mshape, np.float32)
mosaic2 = np.zeros_like(mosaic1)
mosaic2_gray = np.zeros(mshape[:2], np.float32)

for i, (tile, shift) in enumerate(zip(tcs, shifts)):
    hue = idx2hue(i, len(tiles))
    img = skimage.exposure.rescale_intensity(
        skimage.img_as_float32(tile.plane.image), in_range=(0, rmax)
    )
    img_c = colorize(img, hue)
    pos_nominal = tile.plane.bounds.vector1 - origin
    pos_corrected = pos_nominal + shift
    #if i == 2:
    #    import pdb; pdb.set_trace()
    ashlar.reg.paste(mosaic1, img_c, pos_nominal / scale, np.add)
    ashlar.reg.paste(mosaic2, img_c, pos_corrected / scale, np.add)
    ashlar.reg.paste(
        mosaic2_gray, img, pos_corrected / scale, ashlar.reg.pastefunc_blend
    )

crops = {}
for img, name in ((mosaic1, 'B'), (mosaic2, 'C')):
    crop = img[cmin[0]:cmax[0],cmin[1]:cmax[1]]
    crop = skimage.exposure.rescale_intensity(crop, in_range=(0.05, 0.9))
    crop = skimage.exposure.adjust_gamma(crop, 1/2.2)
    crops[name] = crop
    skimage.io.imsave(f'Figure_1{name}.png', crop, check_contrast=False)


# shifts_other = []
# rmax_other = []
# mosaic_gray_other = []
# for rp in (rp2, rp3):
#     alignments = [
#         rp.compute_neighbor_intersection(tiles[a], tiles[b])
#         for a, b in [[0, 1], [0, 2], [2, 3]]
#     ]
#     cycle_shifts = [
#         ashlar.Vector(0, 0),
#         alignments[0].alignment.shift,
#         alignments[1].alignment.shift,
#         alignments[1].alignment.shift + alignments[2].alignment.shift
#     ]
#     cycle_tcs = [rp.get_tile(t) for t in tiles]
#     layer_alignment = ashlar.align.register_planes(
#         tcs[0].plane, cycle_tcs[0].plane
#     )
#     layer_shift = layer_alignment.shift
#     cycle_rmax = np.percentile(
#         np.dstack([skimage.img_as_float32(t.plane.image) for t in cycle_tcs]),
#         99.99
#     )
#     mg = np.zeros_like(mosaic2_gray)
#     for i, (tile, shift) in enumerate(zip(cycle_tcs, cycle_shifts)):
#         hue = idx2hue(i, len(tiles))
#         img = skimage.exposure.rescale_intensity(
#             skimage.img_as_float32(tile.plane.image), in_range=(0, cycle_rmax)
#         )
#         pos_nominal = tile.plane.bounds.vector1 - origin
#         pos_corrected = pos_nominal + shift + layer_shift
#         if i != 0:
#             pos_corrected += (np.random.rand(2) - 0.5) * 3
#         ashlar.reg.paste(
#             mg, img, pos_corrected / scale, np.maximum
#         )
#     shifts_other.append(cycle_shifts)
#     rmax_other.append(cycle_rmax)
#     mosaic_gray_other.append(mg)
