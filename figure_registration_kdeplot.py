import argparse
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import seaborn as sns
import sys

colors = ["#1e90ff", "#ee5665", "#1fa23f", "#ee50ff"]

parser = argparse.ArgumentParser(
    description="Plot kernel density estimates of several 1-D arrays stored in"
    " .npy files",
)
parser.add_argument(
    "-i", metavar="NAME:input.npy", nargs="+", required=True, dest="inputs",
    help="Series name and input data file path, separated by a colon",
)
parser.add_argument(
    "-o", metavar="output.pdf", type=pathlib.Path, required=True, dest="output",
    help="Ouput plot path",
)
parser.add_argument(
    "-m", metavar="MAX", type=float, required=True, dest="max",
    help="Maximum X-axis value for the plot",
)
args = parser.parse_args()
assert all(':' in x for x in args.inputs), "Malformed -i value"
assert len(args.inputs) <= len(colors), "Script needs more color values"
args.inputs = {k: pathlib.Path(v) for k, v in (x.split(':') for x in args.inputs)}

fig, ax = plt.subplots(figsize=(6, 2.5))
for (name, path), color in zip(args.inputs.items(), colors):
    data = np.load(path)
    kwargs = dict(
        data=np.where(data < args.max, data, args.max),
        clip=[0, args.max],
        color=color,
        ax=ax,
    )
    sns.kdeplot(**kwargs, fill=True, lw=0, alpha=0.2)
    sns.kdeplot(**kwargs)
    line = ax.lines[-1]
    xs = line.get_xdata()
    ys = line.get_ydata()
    mx = np.median(data)
    my = np.interp(mx, xs, ys)
    ax.vlines(mx, 0, my, color=color, linestyle="--")
    ax.annotate(
        f"{name}\nmedian = {mx:#.3g}",
        (mx, my),
        color=color,
        xytext=(0, 10),
        textcoords="offset points",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontweight="bold",
    )
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Registration error (\u00B5m)")
ax.set_ylabel("Density")
fig.tight_layout()
fig.savefig(args.output)
print(f"Wrote figure to: {args.output}")
