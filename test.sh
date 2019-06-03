set -ex
python test.py  \
--dataroot ./datasets/tomato \
--name tomato_v9_May31_batch \
--model cycle_gan \
--netG unet_256 \
--gpu_ids 1 \
--norm_G batch \
--norm_D batch \
--batch_size 4
