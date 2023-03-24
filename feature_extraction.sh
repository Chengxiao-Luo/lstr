#CONFIG=configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py
CHECKPOINT=checkpoints/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth

CUDA_VISIBLE_DEVICES=0 python feature_extraction.py --data-prefix data/thumos14/rawframes --data-list data/thumos14/video.txt --output-prefix ./rgb_feat --modality RGB --ckpt $CHECKPOINT
