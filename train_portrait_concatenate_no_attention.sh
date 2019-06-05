set -ex
python train.py  \
--dataroot ./datasets/portrait \
--name portrait_June04_v7_catcontenate_no_attention \
--model cycle_gan \
--netG unet_256 \
--display_port 8096 \
--gpu_ids 0 \
--norm_G spectral \
--norm_D spectral \
--batch_size 4
