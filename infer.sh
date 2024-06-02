
export CUDA_VISIBLE_DEVICES=4,5
# python automap_main_inference.py -c configs/inference_64x64_ex.json
python automap_main_inference.py -c configs/inference_64x64_ourdata.json
