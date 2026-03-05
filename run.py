import os
import sys

files = ['1900', '1901', '1902', '1903', '1904',
         '2000', '2001', '2002', '2003', '2004']

for file in files:
    os.system(f"python cluster/visualize_clusters.py \
      --pkl_dir data/peaks_origin/peaks_202505{file}_00_veh.pkl \
      --clusters_csv data/ddpm_test_samples_origin_cluster/peaks_202505{file}_00_veh/clusters_norm.csv \
      --out_png data/ddpm_test_samples_origin_cluster/pca_norm_{file}.png \
      --length_mode crop \
      --mode raw")
