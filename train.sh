set -ex
python train.py  \
--dataroot ./datasets/tomato \
--name tomato_v7_May30 \
--model cycle_gan \
--netG unet_256 \
--display_port 8096 \
--gpu_ids 2 \
--batch_size 6
