import numpy as np
from sklearn.utils import Bunch
import time
import tensorflow as tf

from models.automap_model_pytorch import AUTOMAP_Basic_Model 
from data_loader.automap_inference_data_generator import InferenceDataGenerator
from trainers.automap_inferencer import AUTOMAP_Inferencer
import pynvml

# Initialize pynvml
pynvml.nvmlInit()


def print_gpu_memory_info():
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU {i}: Memory Used: {info.used / 1e6} MB, Memory Total: {info.total / 1e6} MB")
    return info.used

def release_pynvml():
    pynvml.nvmlShutdown()
    

def fake_config(size: int):
    config = Bunch()
    config.batch_size = 1
    config.im_h = size
    config.im_w = size
    square = size * size
    config.fc_input_dim = square * 2
    config.fc_hidden_dim = square
    config.fc_output_dim = square
    config.save_inference_output = 'foo/bar.mat'
    return config


class FakeDataGenerator:
    def __init__(self, q, h, w):
        self.len = q
        
        self.input = np.random.rand(q, 2 * h * w)
        self.output = np.random.rand(q, h * w)
    
    def next_batch(self, ind_start, batch_size):
        idx = np.arange(ind_start,ind_start+batch_size)
        yield self.input[idx], self.output[idx]
        

def print_cuda_memory_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            memory_info = tf.config.experimental.get_memory_info(gpu.name)
            print(f"GPU: {gpu.name}")
            print(f"Memory Allocated: {memory_info['current'] / 1e6} MB")
            print(f"Memory Peak: {memory_info['peak'] / 1e6} MB")
    else:
        print("No GPU devices found.")
        
        
def main():
    size = 64
    config = fake_config(size)
    data = FakeDataGenerator(16, size, size)
    
    # AUTOMAP reconstruction
    print('before model init')
    mem_start = print_gpu_memory_info()
    model = AUTOMAP_Basic_Model(config)
    inferencer = AUTOMAP_Inferencer(model, data, config)
    print('initialized')
    print_gpu_memory_info()
    start = time.time()
    # with tf.GradientTape(watch_accessed_variables=False) as tape:
        # tape.watch(model.trainable_variables)
        # raw_data_input, output = next(data.next_batch(0, 1))
        # c_2, predictions = model(raw_data_input, training=False)
    output = inferencer.inference()  # (N, H*W)
    duration = time.time() - start
    print('done')
    mem_end = print_gpu_memory_info()
    print(f"Memory used: {(mem_end - mem_start)/1e6}) MB")
    
    output = output.reshape(-1, config.im_h, config.im_w)
    
    print(f'Inference time: {duration:.4f} seconds')
    
    release_pynvml()
    


if __name__ == '__main__':
    main()
