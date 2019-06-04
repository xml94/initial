set -ex
python test.py  \
--dataroot ./datasets/tomato \
--name tomato_June04_v2 \
--model cycle_gan \
--netG unet_256 \
--gpu_ids 0 \
--norm_G spectral \
--norm_D spectral \
--batch_size 6
