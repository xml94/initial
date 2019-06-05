set -ex
python train.py  \
--dataroot ./datasets/portrait \
--name portrait_June04_v6_catcontenate_attention \
--model cycle_gan \
--netG unet_256 \
--display_port 8098 \
--gpu_ids 2 \
--norm_G spectral \
--norm_D spectral \
--batch_size 3
