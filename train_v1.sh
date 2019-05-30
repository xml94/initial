set -ex
python train.py  \
--dataroot ./datasets/tomato \
--name tomato_v8_May30_spectral \
--model cycle_gan \
--netG unet_256 \
--display_port 8097 \
--gpu_ids 0 \
--batch_size 4 \
--norm_G spectral \
--norm_D spectral
