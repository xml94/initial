set -ex
python train.py  \
--dataroot ./datasets/tomato \
--name tomato_v9_May31_batch \
--model cycle_gan \
--netG unet_256 \
--display_port 8098 \
--gpu_ids 2 \
--batch_size 4 \
--norm_G batch \
--norm_D batch
