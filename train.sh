set -ex
python train.py  \
--dataroot ./datasets/tomato \
--name tomato_June05_v1 \
--model cycle_gan \
--netG unet_256 \
--display_port 8097 \
--gpu_ids 1 \
--norm_G spectral \
--norm_D spectral \
--batch_size 4
