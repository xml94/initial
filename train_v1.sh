set -ex
python train.py  \
--dataroot ./datasets/tomato \
--name tomato_June04_v3 \
--model cycle_gan \
--netG unet_256 \
--display_port 8096 \
--gpu_ids 1 \
--batch_size 6 \
--norm_G spectral \
--norm_D spectral
