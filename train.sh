PATH_TO_CONFIG_FILE=configs/THUMOS/LSTR/lstr_long_512_work_8_kinetics_1x.yaml
CUDA_VISIBLE_DEVICES=1

#python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES
python train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES