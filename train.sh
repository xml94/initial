set -ex
python train.py  \
--dataroot ./datasets/tomato \
--name tomato_June03_v1 \
--model cycle_gan \
--netG unet_256 \
--display_port 8096 \
--gpu_ids 0 \
--batch_size 6
