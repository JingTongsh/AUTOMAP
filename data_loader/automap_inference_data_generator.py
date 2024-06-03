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
    print(data.shape)
    return data


class InferenceDataGenerator:
    def __init__(self, config):
        self.config = config

        inference_in_file = os.path.join(self.config.data_dir, self.config.inference_input)
        inference_out_file = os.path.join(self.config.data_dir, self.config.inference_target_output)

        self.input = load_mat(inference_in_file)
        self.output = load_mat(inference_out_file)

        self.len = self.input.shape[0]

    def next_batch(self, ind_start, batch_size):
        idx = np.arange(ind_start,ind_start+batch_size)
        yield self.input[idx], self.output[idx]
