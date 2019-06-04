set -ex
python test.py  \
--dataroot ./datasets/tomato \
--name tomato_June04_v4_no_attention \
--model cycle_gan \
--netG unet_256 \
--gpu_ids 2 \
--norm_G spectral \
--norm_D spectral \
--batch_size 6
