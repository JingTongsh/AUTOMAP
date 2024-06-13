from sklearn.utils import Bunch
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pynvml


class AUTOMAP_Basic_Model(nn.Module):
    def __init__(self, config):
        super(AUTOMAP_Basic_Model, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(config.fc_input_dim, config.fc_hidden_dim)
        self.fc2 = nn.Linear(config.fc_hidden_dim, config.fc_output_dim)
        
        # Reshape and zero padding
        self.im_h = config.im_h
        self.im_w = config.im_w
        self.zero_padding = nn.ZeroPad2d(4)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.deconv = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=1, padding=3)
        
    def forward(self, x):
        # Forward pass through fully connected layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # Reshape to (batch_size, 1, im_h, im_w)
        x = x.view(-1, 1, self.im_h, self.im_w)
        
        # Zero padding
        x = self.zero_padding(x)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(x))
        
        # Transpose convolutional layer
        x = self.deconv(c2)
        
        # Flatten the output
        output = x.view(-1, (self.im_h + 8) * (self.im_w + 8))
        
        return output


# Initialize pynvml
pynvml.nvmlInit()


def print_gpu_memory_info(watch_list=[0]):
    if watch_list is None:
        device_count = pynvml.nvmlDeviceGetCount()
        watch_list = list(range(device_count))
    for i in watch_list:
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


class FakeData(Dataset):
    def __init__(self, q, h, w):
        self.len = q
        
        self.input = torch.rand(q, 2 * h * w)
        self.output = torch.rand(q, h * w)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]
        
        
def main():
    
    ########## params to change ##########
    device = 0
    ql = 256
    size = 128
    ######################################
    torch.cuda.set_device(device)
    watch_list = [device]
    config = fake_config(size)
    data = FakeData(ql, size, size)
    loader = DataLoader(data, batch_size=1, shuffle=True)
    
    # AUTOMAP reconstruction
    print('before model init')
    mem_start = print_gpu_memory_info(watch_list)
    model = AUTOMAP_Basic_Model(config).cuda()
    print('initialized model')
    print_gpu_memory_info(watch_list)
    
    with torch.no_grad():
        start = time.time()
        for x, y in loader:
            x = x.cuda()
            output = model(x)
        duration = time.time() - start
        print('done')
        mem_end = print_gpu_memory_info(watch_list)
        print(f"Memory used: {(mem_end - mem_start)/1e6} MB")
        
        print(output.shape)
        
        # output = output.reshape(-1, config.im_h, config.im_w)
        
        print(f'Inference time: {duration:.4f} seconds')
        
        release_pynvml()
        
        print('results:')
        print('{}x{}x{}: {} MB, {} sec'.format(ql, size, size, (mem_end - mem_start)/1e6, duration))
    


if __name__ == '__main__':
    main()
