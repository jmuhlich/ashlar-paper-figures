python \
    figure_registration_kdeplot.py \
    -i \
    ASHLAR:output/COLNOR69MW2_ashlar_dist.npy \
    MIST:output/COLNOR69MW2_mist_dist.npy \
    BigStitcher:output/COLNOR69MW2_bigstitcher_dist.npy \
    -o output/COLNOR69MW2_kdeplot.pdf \
    -m 4

python \
    figure_registration_kdeplot.py \
    -i \
    ASHLAR:output/COLNOR69MW2_ashlar_dist.npy \
    MIST:output/COLNOR69MW2_mist_dist.npy \
    -o output/COLNOR69MW2_kdeplot_ashlar_mist_only.pdf \
    -m 4

python \
    figure_registration_kdeplot.py \
    -i \
    ASHLAR:output/HBM355.JDLK.244_ashlar_dist.npy \
    MIST:output/HBM355.JDLK.244_mist_dist.npy \
    BigStitcher:output/HBM355.JDLK.244_bigstitcher_dist.npy \
    Akoya:output/HBM355.JDLK.244_akoya_dist.npy \
    -o output/HBM355.JDLK.244_kdeplot.pdf \
    -m 0.75

python \
    figure_registration_kdeplot.py \
    -i \
    ASHLAR:output/TMA11_ashlar_dist.npy \
    MIST:output/TMA11_mist_dist.npy \
    -o output/TMA11_kdeplot.pdf \
    -m 4

python \
    figure_registration_kdeplot.py \
    -i \
    ASHLAR:output/Tonsil-WD-76845-02_ashlar_dist.npy \
    MIST:output/Tonsil-WD-76845-02_mist_dist.npy \
    -o output/Tonsil-WD-76845-02_kdeplot.pdf \
    -m 4

python \
    figure_registration_kdeplot.py \
    -i \
    ASHLAR:output/SM-243_ashlar_dist.npy \
    MIST:output/SM-243_mist_dist.npy \
    -o output/SM-243_kdeplot.pdf \
    -m 0.15
    #'Zeiss Zen':output/SM-243_zen_dist.npy \

# python \
#     figure_registration_kdeplot.py \
#     -i ASHLAR:output/WD-76845-097_ashlar_dist.npy \
#     -o output/WD-76845-097_kdeplot.pdf -m 6


python \
    figure_registration_kdeplot.py \
    -i 'Zeiss Zen':output/Maric-brain_zen_dist.npy \
    -o output/Maric-brain_kdeplot.pdf \
    -m 2

python \
    figure_registration_kdeplot.py \
    -i \
    'Ashlar (GE IN Cell 6000)':output/LSP10407-InCell6000-ashlar-dist.npy \
    'Leica GT450':output/LSP10407-GT450_dist.npy \
    'Leica Versa':output/LSP10407-Versa_dist.npy \
    'Hamamatsu Nanozoomer':output/LSP10407-Hamamatsu_dist.npy \
    -o output/LSP10407-slide-scanners-kdeplot.pdf \
    -m 1.5
