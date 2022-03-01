import argparse
import numpy as np
import pathlib
import tifffile
import zarr


parser = argparse.ArgumentParser(
    description="Extract and invert the blue channel from an RGB TIFF",
)
parser.add_argument(
    "in_path", metavar="input.tif", type=pathlib.Path,
    help="Input TIFF path",
)
parser.add_argument(
    "out_path", metavar="output.tif", type=pathlib.Path,
    help="Output TIFF path",
)
args = parser.parse_args()

tiff = tifffile.TiffFile(args.in_path)
assert tiff.series[0].axes == "YXS", "Input TIFF must be RGB"
assert tiff.series[0].dtype == "uint8", "Input TIFF must be 8-bit"
z = zarr.open(tiff.series[0].aszarr())
print("Reading blue channel")
img = z[0][..., 2]
print("Inverting image")
np.subtract(255, img, out=img)
print(f"Saving to {args.out_path}")
tifffile.imwrite(args.out_path, img, tile=(1024,1024))
