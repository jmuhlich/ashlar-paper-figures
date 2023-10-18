from ashlar import reg
import numpy as np
import pandas as pd
import pathlib
import sys
import time

files = (
    (
        'COLNOR69MW2-cycle-1.ome.tif',
        'COLNOR69MW2-cycle-2.ome.tif',
    ),
    (
        'Tonsil-Codex-1.ome.tif',
        'Tonsil-Codex-2.ome.tif',
    ),
    (
        'HBM355.JDLK.244-cycle-1.ome.tif',
        'HBM355.JDLK.244-cycle-2.ome.tif',
    ),
    (
        'TMA11-cycle-1.ome.tif',
        'TMA11-cycle-2.ome.tif',
    ),
    (
        'Tonsil-WD-76845-02-cycle-1.ome.tif',
        'Tonsil-WD-76845-02-cycle-2.ome.tif',
    ),
    (
        'LSP10407-InCell6000-cycle-1.ome.tif',
        'LSP10407-InCell6000-cycle-2.ome.tif'
    ),
    # (
    #     'SM-243-cycle-1.ome.tif',
    #     'SM-243-cycle-2.ome.tif',
    # ),
    # (
    #     'LSP10407-CyteFinder-cropped-cycle-1.ome.tif',
    #     'LSP10407-CyteFinder-cropped-cycle-2.ome.tif',
    # ),
    # (
    #     'LSP10407-CyteFinder-cycle-1.ome.tif',
    #     'LSP10407-CyteFinder-cycle-2.ome.tif',
    # ),
)

files = files[1:2]
print(files)

base_path = pathlib.Path(sys.argv[0]).parent.resolve()
data_path = base_path / 'input' / 'raw'
times = pd.DataFrame()

for i, (f1, f2) in enumerate(files):
    print(f"Processing: {f1}, {f2}\n===\n")

    t_begin1 = time.perf_counter()
    reader1 = reg.BioformatsReader(str(data_path / f1))
    aligner1 = reg.EdgeAligner(reader1, verbose=True)
    ftimes1 = aligner1.run()
    t_end1 = time.perf_counter()
    t1 = t_end1 - t_begin1

    t_begin2 = time.perf_counter()
    reader2 = reg.BioformatsReader(str(data_path / f2))
    aligner2 = reg.LayerAligner(reader2, aligner1, verbose=True)
    ftimes2 = aligner2.run()
    t_end2 = time.perf_counter()
    t2 = t_end2 - t_begin2

    times.loc[f1, "Stitch_Time"] = t1
    times.loc[f1, "Register_Time"] = t2
    df1 = pd.DataFrame(ftimes1, index=[f1])
    df2 = pd.DataFrame(ftimes2, index=[f1])
    df1.iloc[0, 1:] = np.diff(df1.values)
    df2.iloc[0, 1:] = np.diff(df2.values)
    df1["init"] = df1["begin"] - t_begin1
    df2["init"] = df2["begin"] - t_begin2
    del df1["begin"]
    del df2["begin"]
    df2.columns += "2"
    if i == 0:
        times = pd.concat([times, df1, df2], axis=1)
    else:
        times.loc[[f1], df1.columns] = df1
        times.loc[[f1], df2.columns] = df2
    print("\n\n")

times["Tool"] = "Ashlar"
times.index.name = "Sample"
print("\n\n\n\n\n")
try:
    times.to_csv(sys.argv[1])
    print(f"Wrote results to: {sys.argv[1]}")
except Exception as e:
    if len(sys.argv == 2):
        print(f"Failed writing to {sys.argv[1]}:")
        print(e)
        print("Writing to stdout instead")
        print()
    print("RESULTS\n=======")
    print(times.to_csv())
