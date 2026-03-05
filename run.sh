
python diffusion/diffusion_train.py \
--pkl_dir data/peaks_agc \
--out_dir data/ddpm_out_agc \
  --normalize per_sample \
  --timesteps 1000 \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-4




python diffusion/diffusion_sample.py \
  --ckpt data/ddpm_out_agc/ckpt_best.pt \
  --out_dir data/ddpm_test_samples_agc \
  --num 8 \
  --batch_size 8 \
  --save_grouped_pk



python vehicle_signal/plot_synth_signals.py \
  --npz_path data/ddpm_test_samples_agc/synth_signals.npz \
  --num 8 \
  --out_path data/ddpm_test_samples_agc_plot/my_first8.png




python cluster/cluster_vehicle_signals.py \
  --pkl_dir data/peaks_agc/peaks_2025051900_00_veh.pkl \
  --out_dir data/ddpm_test_samples_agc_cluster \
  --length_mode crop \
  --use_pca \
  --algo hdbscan \
  --min_cluster_size 30 \
  --min_samples 10 \
  --k_range 3,6
