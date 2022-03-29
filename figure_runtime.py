import ashlar.reg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import scipy.stats
import sklearn.cluster
import sys

labels = {
    "COLNOR69MW2": "Colon",
    "HBM355.JDLK.244": "Spleen",
    "TMA11": "TMA",
    "Tonsil-WD-76845-02": "Tonsil",
    "SM-243": "Brain",
    "LSP10407-InCell6000": "CRC",
}

base_path = pathlib.Path(sys.argv[0]).parent

df = pd.read_csv(base_path / "input/times.csv")
df = df[df.Tool=='Ashlar']
dfs = df[['Sample']].drop_duplicates()
dfs["Name"] = dfs["Sample"].str.replace(r'-cycle.*', '')
dfs["Path"] = (
    'input/raw' / (dfs["Sample"] + ".ome.tif").apply(pathlib.Path)
)
dfs["Size"] = dfs.Path.apply(lambda x: x.stat().st_size)

def get_overlap(path):
    reader = ashlar.reg.BioformatsReader(str(path))
    aligner = ashlar.reg.EdgeAligner(reader)
    pos = reader.metadata.positions
    pos -= np.min(pos, axis=0)
    ndist = np.abs(np.array([
        pos[u] - pos[v] for u, v in aligner.neighbors_graph.edges
    ]))
    centroids, _, _ = sklearn.cluster.k_means(ndist, 2)
    overlap = np.mean(1 - np.max(centroids, axis=0) / reader.metadata.size)
    return overlap

def get_pixels(path):
    reader = ashlar.reg.BioformatsReader(str(path))
    return reader.metadata.num_images * np.prod(reader.metadata.size)

def get_tile_pixels(path):
    reader = ashlar.reg.BioformatsReader(str(path))
    return np.prod(reader.metadata.size)

dfs["Overlap"] = dfs["Path"].map(get_overlap)
dfs["Pixels"] = dfs["Path"].map(get_pixels)
dfs["Tile_Pixels"] = dfs["Path"].map(get_tile_pixels)

df = pd.merge(df, dfs)
dfm = df.groupby("Name").mean()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 8))

rx = np.hstack([[0], np.unique(df["Pixels"])])

res_s = scipy.stats.linregress(df["Pixels"], df["Stitch_Time"])
ax1.scatter(
    df["Pixels"], df["Stitch_Time"], color="none", edgecolors="tab:blue", lw=0.5
)
ax1.plot(rx, res_s.intercept + res_s.slope * rx, "-", c="darkgray", zorder=0)
for i, r in dfm.iterrows():
    s = labels.get(i, i)
    xy = r.Pixels, r.Stitch_Time
    ax1.annotate(s, xy, (5, -3), textcoords="offset points")
ax1.text(
    0.02, 1, f'R\u00B2 = {res_s.rvalue**2:.4}', transform=ax1.transAxes, va="top"
)
ax1.set_xlim(xmin=0)
ax1.set_ylim(ymin=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
#ax1.set_xlabel("Total pixels")
ax1.set_ylabel("Stitching time (s)")

res_r = scipy.stats.linregress(df["Pixels"], df["Register_Time"])
ax2.scatter(
    df["Pixels"], df["Register_Time"], color="none", edgecolors="tab:blue", lw=0.5
)
ax2.plot(rx, res_r.intercept + res_r.slope * rx, "-", c="darkgray", zorder=0)
for i, r in dfm.iterrows():
    s = labels.get(i, i)
    xy = r.Pixels, r.Register_Time
    ax2.annotate(s, xy, (5, -3), textcoords="offset points")
ax2.text(
    0.02, 1, f'R\u00B2 = {res_r.rvalue**2:.4}', transform=ax2.transAxes, va="top"
)
ax2.set_xlim(xmin=0)
ax2.set_ylim(ymin=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlabel("Total pixels")
ax2.set_ylabel("Registration time (s)")

fig.tight_layout()

fpath = base_path / "output" / "runtime.pdf"
fig.savefig(fpath)
print(f"Wrote figure to: {fpath}")
