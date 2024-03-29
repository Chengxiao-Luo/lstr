PATH_TO_CONFIG_FILE=configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x.yaml
PATH_TO_CHECKPOINT=checkpoints/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x/epoch-25.pth
CUDA_VISIBLE_DEVICES=3
#VIDEO_NAME=video_test_0000004
VIDEO_NAME=video_validation_0000051

#python inference.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
#    MODEL.CHECKPOINT $PATH_TO_CHECKPOINT MODEL.LSTR.INFERENCE_MODE batch

python inference.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
    MODEL.CHECKPOINT $PATH_TO_CHECKPOINT MODEL.LSTR.INFERENCE_MODE stream DATA.TEST_SESSION_SET "['$VIDEO_NAME']"