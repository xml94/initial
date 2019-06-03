set -ex
python train.py  \
--dataroot ./datasets/horse2zebra \
--name horse2zebra_v8_May30_spectral \
--model cycle_gan \
--netG unet_256 \
--display_port 8090 \
--gpu_ids 1 \
--batch_size 4 \
--norm_G spectral \
--norm_D spectral
