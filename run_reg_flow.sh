python figure_registration_flow.py input/COLNOR69MW2_ashlar_cycle_* output/COLNOR69MW2_ashlar_flow.tif --data-output output/COLNOR69MW2_ashlar_dist.npy --max-dist 6
python figure_registration_flow.py input/COLNOR69MW2_mist_cycle_* output/COLNOR69MW2_mist_flow.tif --data-output output/COLNOR69MW2_mist_dist.npy --max-dist 6

python figure_registration_flow.py input/HBM355.JDLK.244_ashlar_cycle_* output/HBM355.JDLK.244_ashlar_flow.tif --data-output output/HBM355.JDLK.244_ashlar_dist.npy --max-dist 2
python figure_registration_flow.py input/HBM355.JDLK.244_mist_cycle_* output/HBM355.JDLK.244_mist_flow.tif --data-output output/HBM355.JDLK.244_mist_dist.npy --max-dist 2
python figure_registration_flow.py input/HBM355.JDLK.244_akoya_cycle_* output/HBM355.JDLK.244_akoya_flow.tif --data-output output/HBM355.JDLK.244_akoya_dist.npy --max-dist 2

python figure_registration_flow.py input/TMA11_ashlar_cycle_* output/TMA11_ashlar_flow.tif --data-output output/TMA11_ashlar_dist.npy --max-dist 6
python figure_registration_flow.py input/TMA11_mist_cycle_* output/TMA11_mist_flow.tif --data-output output/TMA11_mist_dist.npy --max-dist 6

python figure_registration_flow.py input/Tonsil-WD-76845-02_ashlar_cycle_* output/Tonsil-WD-76845-02_ashlar_flow.tif --data-output output/Tonsil-WD-76845-02_ashlar_dist.npy --max-dist 6
python figure_registration_flow.py input/Tonsil-WD-76845-02_mist_cycle_* output/Tonsil-WD-76845-02_mist_flow.tif --data-output output/Tonsil-WD-76845-02_mist_dist.npy --max-dist 6

python \
    figure_registration_flow.py \
    input/SM-243_ashlar_cycle_1.tif \
    input/SM-243_ashlar_cycle_2.tif \
    output/SM-243_ashlar_flow.tif \
    --data-output output/SM-243_ashlar_dist.npy \
    --max-dist 2
python \
    figure_registration_flow.py \
    input/SM-243_mist_cycle_1.tif \
    input/SM-243_mist_cycle_2.tif \
    output/SM-243_mist_flow.tif \
    --data-output output/SM-243_mist_dist.npy \
    --max-dist 2
python \
    figure_registration_flow.py \
    input/SM-243_zen_cycle_1.tif \
    input/SM-243_zen_cycle_2.tif \
    output/SM-243_zen_flow.tif \
    --data-output output/SM-243_zen_dist.npy \
    --max-dist 2

python figure_registration_flow.py input/WD-76845-097_ashlar_cycle_* output/WD-76845-097_ashlar_flow.tif --data-output output/WD-76845-097_ashlar_dist.npy --max-dist 6


python \
    figure_registration_flow.py \
    input/Maric-brain-cycle-1.tif \
    input/Maric-brain-cycle-2.tif \
    output/Maric-brain_zen_flow.tif \
    --data-output output/Maric-brain_zen_dist.npy \
    --max-dist 6

# The --max-distance values below all correspond to 1.5 microns, using the
# following pixel sizes taken from the original files:
# InCell6000 : 0.325
# GT450      : 0.263687
# Versa      : 0.2762
# Hamamatsu  : 0.225810093711189
python \
    figure_registration_flow.py \
    input/LSP10407-InCell6000-ashlar-1.tif \
    input/LSP10407-InCell6000-ashlar-2.tif \
    output/LSP10407-InCell6000-ashlar-flow.tif \
    --data-output output/LSP10407-InCell6000-ashlar-dist.npy \
    --intensity-threshold 65250 \
    --crop 1350,28200,4900,28230 \
    --output-contrast-percentile 5 95 \
    --output-brightness-scale 0.2 \
    --max-distance 4.62
python \
    figure_registration_flow.py \
    input/LSP10407-GT450-1.tif \
    input/LSP10407-GT450-2.tif \
    output/LSP10407-GT450_flow.tif \
    --data-output output/LSP10407-GT450_dist.npy \
    --intensity-threshold 80 \
    --crop 64013,97413,32715,61265 \
    --output-contrast-percentile 5 95 \
    --output-brightness-scale 0.2 \
    --max-distance 5.69
python \
    figure_registration_flow.py \
    input/LSP10407-Versa-1.tif \
    input/LSP10407-Versa-2.tif \
    output/LSP10407-Versa_flow.tif \
    --data-output output/LSP10407-Versa_dist.npy \
    --intensity-threshold 80 \
    --crop 17316,49120,24517,51646 \
    --output-contrast-percentile 5 95 \
    --output-brightness-scale 0.2 \
    --max-distance 5.43
python \
    figure_registration_flow.py \
    input/LSP10407-Hamamatsu-1.tif \
    input/LSP10407-Hamamatsu-2.tif \
    output/LSP10407-Hamamatsu_flow.tif \
    --data-output output/LSP10407-Hamamatsu_dist.npy \
    --intensity-threshold 80 \
    --crop 29000,67800,35000,68000 \
    --output-contrast-percentile 5 95 \
    --output-brightness-scale 0.2 \
    --max-distance 6.64
# python \
#     figure_registration_flow.py \
#     input/LSP10407-InCell6000-old-ashlar-1.tif \
#     input/LSP10407-InCell6000-old-ashlar-2.tif \
#     output/LSP10407-InCell6000-old-ashlar-flow.tif \
#     --data-output output/LSP10407-InCell6000-old-ashlar-dist.npy \
#     --intensity-threshold 65300 \
#     --crop 1350,28200,4900,28230 \
#     --output-contrast-percentile 5 95 \
#     --output-brightness-scale 0.2 \
#     --max-distance 4.62
# python \
    #     figure_registration_flow.py \
    #     input/LSP10407-CyteFinder-ashlar-1.tif \
    #     input/LSP10407-CyteFinder-ashlar-2.tif \
    #     output/LSP10407-CyteFinder-ashlar-flow.tif \
    #     --data-output output/LSP10407-CyteFinder-ashlar-dist.npy \
    #     --intensity-threshold 150 \
    #     --crop 0,26630,0,23550 \
    #     --max-distance 2
