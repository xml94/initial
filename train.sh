set -ex
python train.py  \
--dataroot ./datasets/tomato \
--name tomato_June05_v2 \
--model cycle_gan \
--netG unet_256 \
--display_port 8098 \
--gpu_ids 2 \
--norm_G spectral \
--norm_D spectral \
--batch_size 6
