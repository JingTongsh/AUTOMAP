import os
import numpy as np
from scipy.io import loadmat
import mat73


def load_mat(file):
    try:
        file = loadmat(file)
    except:
        file = mat73.loadmat(file)
    keys = list(file.keys())
    keys_useful = [key for key in keys if not key.startswith('__')]
    key = keys_useful[0]
    print(key)
    data = file.get(key)
    data = np.array(data).transpose()
    return data


class DataGenerator:
    def __init__(self, config):
        self.config = config

        train_in_file = os.path.join(self.config.data_dir,self.config.train_input)
        train_out_file = os.path.join(self.config.data_dir,self.config.train_output)

        self.input = load_mat(train_in_file)
        self.output = load_mat(train_out_file)
        
        self.len = self.input.shape[0]
        
       
    def next_batch(self, batch_size):
        idx = np.random.choice(self.len, batch_size)
        yield self.input[idx], self.output[idx]

class ValDataGenerator:
    def __init__(self, config):
        self.config = config

        test_in_file = os.path.join(self.config.data_dir, self.config.test_input)
        test_out_file = os.path.join(self.config.data_dir, self.config.test_output)

        self.input = load_mat(test_in_file)
        self.output = load_mat(test_out_file)

        self.len = self.input.shape[0]

    def next_batch(self, batch_size):
        idx = np.random.choice(self.len, batch_size)
        yield self.input[idx], self.output[idx]