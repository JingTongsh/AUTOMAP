
# export CUDA_VISIBLE_DEVICES=2 && python automap_main_train.py -c configs/train_64x64_ex.json
# export CUDA_VISIBLE_DEVICES=3 && python automap_main_train.py -c configs/train_64x64_ourdata.json
export CUDA_VISIBLE_DEVICES=0 && python automap_main_train.py -c configs/train_128x128_ourdata.json
