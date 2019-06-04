set -ex
python train.py  \
--dataroot ./datasets/tomato \
--name tomato_June04_v4_no_attention \
--model cycle_gan \
--netG resnet_9blocks \
--display_port 8096 \
--gpu_ids 1 \
--norm_G instance \
--norm_D instance \
--batch_size 2
