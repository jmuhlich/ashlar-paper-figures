import sys
import pathlib
import numpy as np
import networkx as nx
import skimage.util, skimage.color, skimage.exposure
import ashlar, ashlar.reg
from ashlar.bioformats import BioformatsReader


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


# Curated tile lists that produce a good figure.
known_datasets = {
    'COLNOR69MW2-cycle-0.ome.tif': [
        '???', [351, 352, 380, 381]
    ],
    'COLNOR69MW2@20191106_172134_384761.rcpnl': [
        'TNP_PILOT2', [386, 387, 357, 358]
    ],
    'Scan_20170408_105927_01x4x00024.rcpnl': [
        'ZM_CYCIF1_059_ITOX ... BP40', [13, 14, 19, 20]
    ],
}

if len(sys.argv) == 1:
    print(f"Usage: {sys.argv[0]} in_path [tile1 tile2 tile3 tile4]")
    sys.exit()

file_path = pathlib.Path(sys.argv[1])
if len(sys.argv) == 2 and file_path.name in known_datasets:
    tile_numbers = known_datasets[file_path.name][1]
else:
    tile_numbers = list(map(int, sys.argv[2:]))
    if len(tile_numbers) != 4:
        print("Please provide the numbers of 4 mutually overlapping tiles,")
        print("or specify one of the following datasets:")
        for name, (loc, _) in known_datasets.items():
            print(f"  {loc} ... {name}")
        sys.exit(1)

reader = BioformatsReader.from_path(str(file_path))
tileset = reader.tileset.subset(tile_numbers)
rp = ashlar.RegistrationProcess(
    tileset=tileset, channel_number=0, max_permutations=20
)
tiles = [rp.get_tile(t) for t in range(len(tileset))]

origin = np.min(tileset.positions, axis=0)
scale = tiles[0].plane.pixel_size
mshape = tuple(np.array(tiles[0].plane.image.shape) * 2) + (3,)
tile_intersection = tiles[0].plane.bounds
for tile in tiles[1:]:
    tile_intersection = tile_intersection.intersection(tile.plane.bounds)
if tile_intersection.area == 0:
    print(f"Tiles {tile_numbers} are not mutually overlapping")
    sys.exit()
ccenter = tile_intersection.center - origin
cshape = ashlar.Vector(100, 100)
cmin = ((ccenter - cshape) / scale).astype(int)
cmax = ((ccenter + cshape) / scale).astype(int)

executor = ashlar.RegistrationProcessExecutor(process=rp)
executor.neighbor_alignments()
alignments = executor.neighbor_alignments_
alignments = sorted(alignments, key=lambda x: x.plane_alignment.error)[:3]
edge_shifts = {a.tile_indexes: a.get_shift(a.tile_index_2) for a in alignments}
positions = tileset.positions - origin
new_positions = positions.copy()
#positions *= 0.995
g = nx.DiGraph(list(edge_shifts))
for target, path in nx.algorithms.single_source_shortest_path(g, 0).items():
    for u, v in zip(path[:-1], path[1:]):
        new_positions[target] += edge_shifts[u, v]

rmax = np.percentile(
    np.dstack([
        skimage.util.img_as_float32(rp.get_tile(t).plane.image)
        for t in range(len(tileset))
    ]),
    99.99
)

mosaic1 = np.zeros(mshape, np.float32)
mosaic2 = np.zeros_like(mosaic1)
#mosaic2_gray = np.zeros(mshape[:2], np.float32)

data = zip(tiles, positions, new_positions)
for i, (tile, pos_nominal, pos_corrected) in enumerate(data):
    hue = idx2hue(i, len(tiles))
    img = skimage.exposure.rescale_intensity(
        skimage.util.img_as_float32(tile.plane.image), in_range=(0, rmax)
    )
    img_c = colorize(img, hue)
    ashlar.reg.paste(mosaic1, img_c, pos_nominal / scale, np.add)
    ashlar.reg.paste(mosaic2, img_c, pos_corrected / scale, np.add)
    # ashlar.reg.paste(
    #     mosaic2_gray, img, pos_corrected / scale, ashlar.reg.pastefunc_blend
    # )

crops = {}
for img, name in ((mosaic1, 'B'), (mosaic2, 'C')):
    crop = img[cmin[0]:cmax[0],cmin[1]:cmax[1]]
    crop = skimage.exposure.rescale_intensity(crop, in_range=(0.05, 0.9))
    crop = skimage.exposure.adjust_gamma(crop, 1/2.2)
    crop = skimage.util.img_as_ubyte(crop)
    crops[name] = crop
    skimage.io.imsave(f'Figure_1{name}.png', crop, check_contrast=False)
